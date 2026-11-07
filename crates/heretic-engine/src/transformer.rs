// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Generic decoder-only transformer (Llama, Mistral, Qwen2, Qwen3, ...).
//!
//! The numerics follow the Transformers implementations as closely as practical:
//! RMSNorm and rotary embeddings are computed in float32 and cast back to the model
//! dtype in the same places, and attention probabilities are computed in float32.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::Result;
use candle_core::{D, DType, Device, IndexOp, Tensor};

use crate::config::{Activation, RopeScaling, TextConfig};
use crate::lora::{Linear, LoraLinear, matmul_t};
use crate::weights::WeightStore;

/// Name of the attention output projection component.
pub const ATTN_O_PROJ: &str = "attn.o_proj";
/// Name of the MLP down projection component.
pub const MLP_DOWN_PROJ: &str = "mlp.down_proj";

/// The list contains one element per layer. Each element maps from the component name
/// to a (possibly sparse) mapping from the module index to an (input, output) tuple
/// containing the I/O tensors of shape (prompt, component), stored on the CPU.
pub type ModuleIo = Vec<BTreeMap<String, BTreeMap<usize, (Tensor, Tensor)>>>;

/// What to record during a forward pass, in addition to the logits.
#[derive(Debug, Clone, Copy, Default)]
pub struct CaptureRequest {
    /// Hidden states at the last position: the embeddings, then the output of each layer,
    /// with the final norm applied to the last one (like Transformers' `hidden_states`).
    pub hidden_states: bool,
    /// Inputs and outputs of the modifiable modules at the last position.
    pub module_io: bool,
}

/// Values recorded during a forward pass.
#[derive(Debug, Default)]
pub struct Captured {
    /// Each tensor has shape (batch, hidden_size), in the model dtype.
    pub hidden_states: Vec<Tensor>,
    pub module_io: ModuleIo,
}

/// Key/value cache for incremental decoding.
#[derive(Debug, Clone)]
pub struct KvCache {
    layers: Vec<Option<(Tensor, Tensor)>>,
}

impl KvCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: vec![None; num_layers],
        }
    }

    pub fn seq_len(&self) -> usize {
        self.layers
            .first()
            .and_then(|l| l.as_ref())
            .map_or(0, |(k, _)| k.dims()[2])
    }
}

/// Inputs of a forward pass.
pub struct ForwardInput<'a> {
    /// Token IDs, shape (batch, seq), u32.
    pub input_ids: &'a Tensor,
    /// Position IDs, shape (batch, seq), f32.
    pub positions: &'a Tensor,
    /// Additive attention mask, shape (batch, 1, seq, kv_seq), f32.
    pub mask: &'a Tensor,
    /// Mask for sliding window layers (if any layer uses a window that the sequence exceeds).
    pub sliding_mask: Option<&'a Tensor>,
    /// Whether to apply the LoRA adapters (false evaluates the original model).
    pub use_adapters: bool,
    /// Whether to return logits for all positions instead of only the last one.
    pub all_logits: bool,
}

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x32 = x.to_dtype(DType::F32)?;
        let variance = x32.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        Ok(normed.to_dtype(dtype)?.broadcast_mul(&self.weight)?)
    }
}

struct Rope {
    /// Shape (1, 1, rotary_dim / 2), f32.
    inv_freq: Tensor,
    attention_scaling: f64,
    rotary_dim: usize,
}

impl Rope {
    fn new(config: &TextConfig, device: &Device) -> Result<Self> {
        let dim = config.rotary_dim();
        let base = config.rope_theta;
        let half = dim / 2;

        let default: Vec<f32> = (0..half)
            .map(|i| 1.0 / (base as f32).powf((2 * i) as f32 / dim as f32))
            .collect();

        let (inv_freq, attention_scaling): (Vec<f32>, f64) = match &config.rope_scaling {
            RopeScaling::Default => (default, 1.0),
            RopeScaling::Linear { factor } => (default.iter().map(|f| f / *factor as f32).collect(), 1.0),
            RopeScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            } => {
                let factor = *factor as f32;
                let low_freq_factor = *low_freq_factor as f32;
                let high_freq_factor = *high_freq_factor as f32;
                let old_context_len = *original_max_position_embeddings as f32;
                let low_freq_wavelen = old_context_len / low_freq_factor;
                let high_freq_wavelen = old_context_len / high_freq_factor;
                let inv_freq = default
                    .iter()
                    .map(|&f| {
                        let wavelen = 2.0 * std::f32::consts::PI / f;
                        let llama = if wavelen > low_freq_wavelen { f / factor } else { f };
                        let smooth =
                            (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
                        let smoothed = (1.0 - smooth) * llama / factor + smooth * llama;
                        let is_medium = wavelen >= high_freq_wavelen && wavelen <= low_freq_wavelen;
                        if is_medium { smoothed } else { llama }
                    })
                    .collect();
                (inv_freq, 1.0)
            }
            RopeScaling::Yarn {
                factor,
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
                attention_factor,
                mscale,
                mscale_all_dim,
                truncate,
            } => {
                // Some models (e.g. DeepSeek-V3) modify max_position_embeddings and keep the
                // pretrained value in original_max_position_embeddings; Transformers then uses
                // the ratio between the two as the factor.
                let factor = if *original_max_position_embeddings > 0.0 {
                    let ratio = config.max_position_embeddings as f64 / original_max_position_embeddings;
                    if ratio > 1.0 { ratio } else { *factor }
                } else {
                    *factor
                };
                let get_mscale = |scale: f64, m: f64| if scale <= 1.0 { 1.0 } else { 0.1 * m * scale.ln() + 1.0 };
                let attention_factor = attention_factor.unwrap_or_else(|| match (mscale, mscale_all_dim) {
                    (Some(m), Some(all)) if *m != 0.0 && *all != 0.0 => {
                        get_mscale(factor, *m) / get_mscale(factor, *all)
                    }
                    _ => get_mscale(factor, 1.0),
                });
                let correction_dim = |rotations: f64| {
                    (dim as f64 * (original_max_position_embeddings / (rotations * 2.0 * std::f64::consts::PI)).ln())
                        / (2.0 * base.ln())
                };
                let (mut low, mut high) = (correction_dim(*beta_fast), correction_dim(*beta_slow));
                if *truncate {
                    low = low.floor();
                    high = high.ceil();
                }
                let low = low.max(0.0);
                let mut high = high.min(dim as f64 - 1.0);
                if low == high {
                    high += 0.001;
                }
                let inv_freq = (0..half)
                    .map(|i| {
                        let pos_freq = (base as f32).powf((2 * i) as f32 / dim as f32);
                        let extrapolation = 1.0 / pos_freq;
                        let interpolation = 1.0 / (factor as f32 * pos_freq);
                        let ramp = ((i as f32 - low as f32) / (high as f32 - low as f32)).clamp(0.0, 1.0);
                        let extrapolation_factor = 1.0 - ramp;
                        interpolation * (1.0 - extrapolation_factor) + extrapolation * extrapolation_factor
                    })
                    .collect();
                (inv_freq, attention_factor)
            }
        };

        Ok(Self {
            inv_freq: Tensor::from_vec(inv_freq, (1, 1, half), device)?,
            attention_scaling,
            rotary_dim: dim,
        })
    }

    /// Returns (cos, sin) of shape (batch, 1, seq, rotary_dim) in the given dtype.
    fn cos_sin(&self, positions: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let freqs = positions.unsqueeze(D::Minus1)?.broadcast_mul(&self.inv_freq)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        let mut cos = emb.cos()?;
        let mut sin = emb.sin()?;
        if self.attention_scaling != 1.0 {
            cos = (cos * self.attention_scaling)?;
            sin = (sin * self.attention_scaling)?;
        }
        Ok((cos.to_dtype(dtype)?.unsqueeze(1)?, sin.to_dtype(dtype)?.unsqueeze(1)?))
    }

    /// Applies rotary embeddings to x of shape (batch, heads, seq, head_dim).
    fn apply(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let head_dim = x.dim(D::Minus1)?;
        let (rot, pass) = if self.rotary_dim < head_dim {
            (
                x.narrow(D::Minus1, 0, self.rotary_dim)?,
                Some(x.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?),
            )
        } else {
            (x.clone(), None)
        };

        let half = self.rotary_dim / 2;
        let x1 = rot.narrow(D::Minus1, 0, half)?;
        let x2 = rot.narrow(D::Minus1, half, half)?;
        let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        let embedded = (rot.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?)?;

        Ok(match pass {
            Some(pass) => Tensor::cat(&[&embedded, &pass], D::Minus1)?,
            None => embedded,
        })
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Arc<LoraLinear>,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scaling: f64,
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Arc<LoraLinear>,
    activation: Activation,
}

struct Layer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

/// Records the last-position input/output of a module.
fn record_io(captured: &mut Captured, layer: usize, component: &str, input: &Tensor, output: &Tensor) -> Result<()> {
    let last = |t: &Tensor| -> Result<Tensor> {
        let seq = t.dim(1)?;
        Ok(t.i((.., seq - 1, ..))?.to_device(&Device::Cpu)?)
    };
    while captured.module_io.len() <= layer {
        captured.module_io.push(BTreeMap::new());
    }
    let modules = captured.module_io[layer].entry(component.to_string()).or_default();
    let index = modules.len();
    modules.insert(index, (last(input)?, last(output)?));
    Ok(())
}

fn activate(x: &Tensor, activation: Activation) -> Result<Tensor> {
    Ok(match activation {
        Activation::Silu => x.silu()?,
        Activation::GeluTanh => x.gelu()?,
        Activation::Gelu => x.gelu_erf()?,
    })
}

fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, kv_heads, seq, head_dim) = x.dims4()?;
    Ok(x.unsqueeze(2)?
        .expand((b, kv_heads, n_rep, seq, head_dim))?
        .reshape((b, kv_heads * n_rep, seq, head_dim))?)
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        rope: &Rope,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        cache: Option<&mut Option<(Tensor, Tensor)>>,
        use_adapters: bool,
        capture: Option<(&mut Captured, usize)>,
    ) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let dtype = x.dtype();

        let shape = |t: Tensor, heads: usize| -> Result<Tensor> { Ok(t.reshape((b, seq, heads, self.head_dim))?) };

        let mut q = shape(self.q_proj.forward(x)?, self.num_heads)?;
        let mut k = shape(self.k_proj.forward(x)?, self.num_kv_heads)?;
        let v = shape(self.v_proj.forward(x)?, self.num_kv_heads)?;

        if let (Some(q_norm), Some(k_norm)) = (&self.q_norm, &self.k_norm) {
            q = q_norm.forward(&q)?;
            k = k_norm.forward(&k)?;
        }

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let q = rope.apply(&q, cos, sin)?;
        let k = rope.apply(&k, cos, sin)?;

        let (k, v) = match cache {
            Some(slot) => {
                let (k, v) = match slot.take() {
                    Some((past_k, past_v)) => (Tensor::cat(&[&past_k, &k], 2)?, Tensor::cat(&[&past_v, &v], 2)?),
                    None => (k, v),
                };
                *slot = Some((k.clone(), v.clone()));
                (k, v)
            }
            None => (k, v),
        };

        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, n_rep)?.contiguous()?;
        let v = repeat_kv(&v, n_rep)?.contiguous()?;

        let scores = (q.matmul(&k.t()?.contiguous()?)? * self.scaling)?;
        let scores = scores.to_dtype(DType::F32)?.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(dtype)?;
        let attn = probs.matmul(&v)?;

        let attn = attn
            .transpose(1, 2)?
            .reshape((b, seq, self.num_heads * self.head_dim))?;
        let out = self.o_proj.forward_with(&attn, use_adapters)?;

        if let Some((captured, layer)) = capture {
            record_io(captured, layer, ATTN_O_PROJ, &attn, &out)?;
        }

        Ok(out)
    }
}

impl Mlp {
    fn forward(&self, x: &Tensor, use_adapters: bool, capture: Option<(&mut Captured, usize)>) -> Result<Tensor> {
        let hidden = (activate(&self.gate_proj.forward(x)?, self.activation)? * self.up_proj.forward(x)?)?;
        let out = self.down_proj.forward_with(&hidden, use_adapters)?;
        if let Some((captured, layer)) = capture {
            record_io(captured, layer, MLP_DOWN_PROJ, &hidden, &out)?;
        }
        Ok(out)
    }
}

/// A decoder-only transformer with modifiable modules.
pub struct Transformer {
    pub config: TextConfig,
    embed_tokens: Tensor,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Tensor,
    rope: Rope,
    dtype: DType,
    device: Device,
}

impl Transformer {
    pub fn load(config: TextConfig, weights: &WeightStore, dtype: DType, device: &Device) -> Result<Self> {
        let load = |name: &str| -> Result<Tensor> { Ok(weights.load_text(name, dtype, device)?.1) };
        let load_optional = |name: &str| weights.load_text_optional(name, dtype, device);

        let linear = |prefix: &str, bias: bool| -> Result<Linear> {
            let weight = load(&format!("{prefix}.weight"))?;
            let bias = if bias {
                Some(load(&format!("{prefix}.bias"))?)
            } else {
                load_optional(&format!("{prefix}.bias"))?
            };
            Ok(Linear::new(weight, bias))
        };

        let lora_linear = |prefix: &str, bias: bool| -> Result<Arc<LoraLinear>> {
            let (name, weight) = weights.load_text(&format!("{prefix}.weight"), dtype, device)?;
            let bias = if bias {
                Some(load(&format!("{prefix}.bias"))?)
            } else {
                load_optional(&format!("{prefix}.bias"))?
            };
            Ok(Arc::new(LoraLinear::new(
                name.trim_end_matches(".weight").to_string(),
                weight,
                bias,
            )))
        };

        let norm = |name: &str| -> Result<RmsNorm> {
            Ok(RmsNorm {
                weight: load(name)?,
                eps: config.rms_norm_eps,
            })
        };

        let embed_tokens = load("embed_tokens.weight")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("layers.{i}");
            let (q_norm, k_norm) = if config.qk_norm {
                (
                    Some(norm(&format!("{p}.self_attn.q_norm.weight"))?),
                    Some(norm(&format!("{p}.self_attn.k_norm.weight"))?),
                )
            } else {
                (None, None)
            };

            layers.push(Layer {
                input_layernorm: norm(&format!("{p}.input_layernorm.weight"))?,
                self_attn: Attention {
                    q_proj: linear(&format!("{p}.self_attn.q_proj"), config.qkv_bias)?,
                    k_proj: linear(&format!("{p}.self_attn.k_proj"), config.qkv_bias)?,
                    v_proj: linear(&format!("{p}.self_attn.v_proj"), config.qkv_bias)?,
                    o_proj: lora_linear(&format!("{p}.self_attn.o_proj"), config.o_bias)?,
                    q_norm,
                    k_norm,
                    num_heads: config.num_attention_heads,
                    num_kv_heads: config.num_key_value_heads,
                    head_dim: config.head_dim,
                    scaling: (config.head_dim as f64).powf(-0.5),
                },
                post_attention_layernorm: norm(&format!("{p}.post_attention_layernorm.weight"))?,
                mlp: Mlp {
                    gate_proj: linear(&format!("{p}.mlp.gate_proj"), config.mlp_bias)?,
                    up_proj: linear(&format!("{p}.mlp.up_proj"), config.mlp_bias)?,
                    down_proj: lora_linear(&format!("{p}.mlp.down_proj"), config.mlp_bias)?,
                    activation: config.hidden_act,
                },
            });
        }

        let norm = norm("norm.weight")?;

        // Checkpoints without a separate head always tie it to the embeddings.
        let lm_head = match weights.lm_head_name() {
            Some(name) if !config.tie_word_embeddings => weights.load(&name, dtype, device)?,
            _ => embed_tokens.clone(),
        };

        let rope = Rope::new(&config, device)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            dtype,
            device: device.clone(),
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// The modifiable modules of a layer, by component name.
    pub fn layer_modules(&self, layer: usize) -> BTreeMap<String, Vec<Arc<LoraLinear>>> {
        let layer = &self.layers[layer];
        let mut modules = BTreeMap::new();
        modules.insert(ATTN_O_PROJ.to_string(), vec![layer.self_attn.o_proj.clone()]);
        modules.insert(MLP_DOWN_PROJ.to_string(), vec![layer.mlp.down_proj.clone()]);
        modules
    }

    /// Runs the model. Returns logits of shape (batch, vocab) for the last position
    /// (or (batch, seq, vocab) if `all_logits` is set), in the model dtype.
    pub fn forward(
        &self,
        input: &ForwardInput,
        mut cache: Option<&mut KvCache>,
        request: CaptureRequest,
        captured: &mut Captured,
    ) -> Result<Tensor> {
        let (b, seq) = input.input_ids.dims2()?;

        let mut x = self
            .embed_tokens
            .index_select(&input.input_ids.flatten_all()?, 0)?
            .reshape((b, seq, self.config.hidden_size))?;

        let (cos, sin) = self.rope.cos_sin(input.positions, self.dtype)?;
        let last = |t: &Tensor| -> Result<Tensor> { Ok(t.i((.., seq - 1, ..))?) };

        for (index, layer) in self.layers.iter().enumerate() {
            if request.hidden_states {
                captured.hidden_states.push(last(&x)?);
            }

            let slot = cache.as_deref_mut().map(|c| &mut c.layers[index]);

            let mask = match (self.config.sliding_window[index], input.sliding_mask) {
                (Some(_), Some(sliding_mask)) => sliding_mask,
                _ => input.mask,
            };

            let residual = x.clone();
            let h = layer.input_layernorm.forward(&x)?;
            let h = layer.self_attn.forward(
                &h,
                &self.rope,
                &cos,
                &sin,
                mask,
                slot,
                input.use_adapters,
                request.module_io.then_some((&mut *captured, index)),
            )?;
            x = (residual + h)?;

            let residual = x.clone();
            let h = layer.post_attention_layernorm.forward(&x)?;
            let h = layer.mlp.forward(
                &h,
                input.use_adapters,
                request.module_io.then_some((&mut *captured, index)),
            )?;
            x = (residual + h)?;
        }

        let x = self.norm.forward(&x)?;

        if request.hidden_states {
            captured.hidden_states.push(last(&x)?);
        }

        if input.all_logits {
            Ok(matmul_t(&x, &self.lm_head)?)
        } else {
            Ok(matmul_t(&last(&x)?, &self.lm_head)?)
        }
    }
}

/// Builds position IDs and the additive attention mask for a left-padded prompt batch,
/// like Transformers does for generation: `position_ids = cumsum(mask) - 1`, with padding
/// positions set to 1, and a causal mask that hides padding keys.
pub fn prefill_inputs(
    attention_mask: &[Vec<u32>],
    sliding_window: Option<usize>,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let b = attention_mask.len();
    let seq = attention_mask.first().map_or(0, Vec::len);

    let mut positions = Vec::with_capacity(b * seq);
    for row in attention_mask {
        let mut count = 0i64;
        for &m in row {
            count += m as i64;
            positions.push(if m == 0 { 1.0 } else { (count - 1) as f32 });
        }
    }

    let mut mask = vec![f32::MIN; b * seq * seq];
    for (row_index, row) in attention_mask.iter().enumerate() {
        for i in 0..seq {
            for j in 0..=i {
                let in_window = sliding_window.is_none_or(|w| i - j < w);
                if row[j] == 1 && in_window {
                    mask[row_index * seq * seq + i * seq + j] = 0.0;
                }
            }
        }
    }

    Ok((
        Tensor::from_vec(positions, (b, seq), device)?,
        Tensor::from_vec(mask, (b, 1, seq, seq), device)?,
    ))
}

/// Builds position IDs and the attention mask for one decoding step, given the
/// attention mask of all previous tokens (which is extended by one valid token).
pub fn decode_step_inputs(
    attention_mask: &[Vec<u32>],
    next_positions: &[f32],
    sliding_window: Option<usize>,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let b = attention_mask.len();
    let kv = attention_mask.first().map_or(0, Vec::len);
    let mut mask = vec![f32::MIN; b * kv];
    for (row_index, row) in attention_mask.iter().enumerate() {
        let i = kv - 1;
        for (j, &m) in row.iter().enumerate() {
            let in_window = sliding_window.is_none_or(|w| i - j < w);
            if m == 1 && in_window {
                mask[row_index * kv + j] = 0.0;
            }
        }
    }
    Ok((
        Tensor::from_vec(next_positions.to_vec(), (b, 1), device)?,
        Tensor::from_vec(mask, (b, 1, 1, kv), device)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_positions_and_mask() -> Result<()> {
        let (positions, mask) = prefill_inputs(&[vec![0, 1, 1], vec![1, 1, 1]], None, &Device::Cpu)?;
        assert_eq!(
            positions.to_vec2::<f32>()?,
            vec![vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 2.0]]
        );
        let mask = mask.to_vec3::<f32>().err();
        assert!(mask.is_some()); // rank 4
        let (_, mask) = prefill_inputs(&[vec![0, 1, 1]], None, &Device::Cpu)?;
        let rows = mask.squeeze(0)?.squeeze(0)?.to_vec2::<f32>()?;
        assert_eq!(rows[2], vec![f32::MIN, 0.0, 0.0]);
        assert_eq!(rows[1], vec![f32::MIN, 0.0, f32::MIN]);
        Ok(())
    }

    #[test]
    fn sliding_window_mask() -> Result<()> {
        let (_, mask) = prefill_inputs(&[vec![1, 1, 1, 1]], Some(2), &Device::Cpu)?;
        let rows = mask.squeeze(0)?.squeeze(0)?.to_vec2::<f32>()?;
        assert_eq!(rows[3], vec![f32::MIN, f32::MIN, 0.0, 0.0]);
        Ok(())
    }
}
