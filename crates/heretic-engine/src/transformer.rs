// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Generic decoder-only transformer: Llama, Mistral, Qwen2, Qwen3, Qwen3.5 (hybrid),
//! Gemma, Gemma 2, Gemma 3 and Gemma 4.
//!
//! The numerics follow the Transformers implementations as closely as practical:
//! RMSNorm and rotary embeddings are computed in float32 and cast back to the model
//! dtype in the same places, and attention probabilities are computed in float32.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context as _, Result};
use candle_core::{D, DType, Device, IndexOp, Tensor};

use crate::config::{
    Activation, KvSharing, LayerKind, LinearAttentionConfig, NormStyle, RopeScaling, RopeSpec, TextConfig,
};
use crate::deltanet;
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

/// Cache for incremental decoding. For attention layers a slot holds (keys, values);
/// for linear attention layers it holds (convolution state, recurrent state).
#[derive(Debug, Clone)]
pub struct KvCache {
    layers: Vec<Option<(Tensor, Tensor)>>,
    seq_len: usize,
}

impl KvCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: vec![None; num_layers],
            seq_len: 0,
        }
    }

    /// Number of tokens processed so far.
    pub fn seq_len(&self) -> usize {
        self.seq_len
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
    /// Llama style: in the model dtype; float32 styles: the effective float32 factor
    /// (`1 + w` for the offset style). `None` = no scale (Gemma 4 `v_norm`).
    weight: Option<Tensor>,
    eps: f64,
    style: NormStyle,
}

impl RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x32 = x.to_dtype(DType::F32)?;
        let variance = x32.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        Ok(match (&self.weight, self.style) {
            (None, _) => normed.to_dtype(dtype)?,
            (Some(w), NormStyle::Llama) => normed.to_dtype(dtype)?.broadcast_mul(w)?,
            (Some(w), _) => normed.broadcast_mul(w)?.to_dtype(dtype)?,
        })
    }
}

struct Rope {
    /// Shape (1, 1, rotary_dim / 2), f32.
    inv_freq: Tensor,
    attention_scaling: f64,
    rotary_dim: usize,
}

impl Rope {
    fn new(spec: &RopeSpec, max_position_embeddings: usize, device: &Device) -> Result<Self> {
        let dim = spec.rotary_dim();
        let base = spec.theta;
        let half = dim / 2;

        let default: Vec<f32> = (0..half)
            .map(|i| 1.0 / (base as f32).powf((2 * i) as f32 / dim as f32))
            .collect();

        let (inv_freq, attention_scaling): (Vec<f32>, f64) = match &spec.scaling {
            RopeScaling::Default => (default, 1.0),
            RopeScaling::Linear { factor } => (default.iter().map(|f| f / *factor as f32).collect(), 1.0),
            RopeScaling::Proportional { factor } => {
                // `_compute_proportional_rope_parameters`: exponents over the full head
                // dimension, zero frequencies for the non-rotated angles.
                let head_dim = spec.head_dim;
                let rope_angles = (spec.partial_rotary_factor * head_dim as f64 / 2.0).floor() as usize;
                let inv_freq = (0..head_dim / 2)
                    .map(|i| {
                        if i < rope_angles {
                            1.0 / (base as f32).powf((2 * i) as f32 / head_dim as f32) / *factor as f32
                        } else {
                            0.0
                        }
                    })
                    .collect();
                (inv_freq, 1.0)
            }
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
                    let ratio = max_position_embeddings as f64 / original_max_position_embeddings;
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
    /// `None` for Gemma 4 layers that reuse the keys/values of an earlier layer.
    k_proj: Option<Linear>,
    /// `None` when shared, or when the key projection doubles as the value projection.
    v_proj: Option<Linear>,
    o_proj: Arc<LoraLinear>,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    v_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scaling: f64,
    softcap: Option<f64>,
    output_gate: bool,
    kv_sharing: KvSharing,
}

/// Qwen3.5 `Qwen3_5GatedDeltaNet`.
struct GatedDeltaNet {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    /// Shape (conv_dim, kernel), float32.
    conv_weight: Tensor,
    /// Float32, shape (value heads).
    dt_bias: Tensor,
    /// `-exp(A_log)`, float32.
    a_neg: Tensor,
    /// `Qwen3_5RMSNormGated` weight, in the model dtype.
    norm_weight: Tensor,
    out_proj: Arc<LoraLinear>,
    dims: LinearAttentionConfig,
    eps: f64,
}

enum Mixer {
    Attention(Box<Attention>),
    Linear(Box<GatedDeltaNet>),
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Arc<LoraLinear>,
    activation: Activation,
}

/// Gemma 4 per-layer input block.
struct PerLayerBlock {
    gate: Linear,
    projection: Linear,
    norm: RmsNorm,
}

struct Layer {
    input_layernorm: RmsNorm,
    mixer: Mixer,
    /// Standard layout: the norm before the MLP. Sandwich layout: the norm after attention.
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: Option<RmsNorm>,
    post_feedforward_layernorm: Option<RmsNorm>,
    mlp: Mlp,
    per_layer: Option<PerLayerBlock>,
    /// Gemma 4 `layer_scalar` buffer.
    layer_scalar: Option<Tensor>,
    sliding: bool,
    rope: usize,
}

/// Gemma 4 per-layer embeddings (model level).
struct PerLayerInputs {
    /// Packed per-layer embedding table, kept on the CPU (it is as large as the decoder).
    embed: Tensor,
    embed_scale: f64,
    model_projection: Tensor,
    projection_norm: RmsNorm,
    dim: usize,
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

fn softcap(x: &Tensor, cap: f64) -> Result<Tensor> {
    Ok(((x / cap)?.tanh()? * cap)?)
}

/// Rounds a scalar to the model dtype, like `torch.tensor(scale).to(dtype)`.
fn round_to_dtype(value: f64, dtype: DType) -> Result<f64> {
    Ok(Tensor::new(value as f32, &Device::Cpu)?
        .to_dtype(dtype)?
        .to_dtype(DType::F32)?
        .to_scalar::<f32>()? as f64)
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
        shared: &mut Option<(Tensor, Tensor)>,
        use_adapters: bool,
        capture: Option<(&mut Captured, usize)>,
    ) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let dtype = x.dtype();

        let shape = |t: Tensor, heads: usize| -> Result<Tensor> { Ok(t.reshape((b, seq, heads, self.head_dim))?) };

        let (mut q, gate) = if self.output_gate {
            // Qwen3.5: the query projection holds (query, gate) per head.
            let both = self
                .q_proj
                .forward(x)?
                .reshape((b, seq, self.num_heads, 2 * self.head_dim))?;
            let q = both.narrow(D::Minus1, 0, self.head_dim)?;
            let gate = both.narrow(D::Minus1, self.head_dim, self.head_dim)?.reshape((
                b,
                seq,
                self.num_heads * self.head_dim,
            ))?;
            (q, Some(gate))
        } else {
            (shape(self.q_proj.forward(x)?, self.num_heads)?, None)
        };
        if let Some(q_norm) = &self.q_norm {
            q = q_norm.forward(&q)?;
        }
        let q = rope
            .apply(&q.transpose(1, 2)?.contiguous()?, cos, sin)?
            .force_contiguous()?;

        let (k, v) = if self.kv_sharing == KvSharing::Reuse {
            shared
                .clone()
                .context("Gemma 4 key/value sharing: the storing layer did not run before a reusing layer")?
        } else {
            let k_proj = self.k_proj.as_ref().context("missing k_proj")?;
            let k_raw = shape(k_proj.forward(x)?, self.num_kv_heads)?;
            let v_raw = match &self.v_proj {
                Some(v_proj) => shape(v_proj.forward(x)?, self.num_kv_heads)?,
                None => k_raw.clone(),
            };
            let k = match &self.k_norm {
                Some(norm) => norm.forward(&k_raw)?,
                None => k_raw,
            };
            let v = match &self.v_norm {
                Some(norm) => norm.forward(&v_raw)?,
                None => v_raw,
            };
            let k = rope
                .apply(&k.transpose(1, 2)?.contiguous()?, cos, sin)?
                .force_contiguous()?;
            let v = v.transpose(1, 2)?.force_contiguous()?;

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
            if self.kv_sharing == KvSharing::Store {
                *shared = Some((k.clone(), v.clone()));
            }
            (k, v)
        };

        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, n_rep)?.force_contiguous()?;
        let v = repeat_kv(&v, n_rep)?.force_contiguous()?;

        let mut scores = (q.matmul(&k.t()?.contiguous()?)? * self.scaling)?;
        if let Some(cap) = self.softcap {
            scores = softcap(&scores, cap)?;
        }
        let scores = scores.to_dtype(DType::F32)?.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(dtype)?;
        let attn = probs.matmul(&v)?;

        let mut attn = attn
            .transpose(1, 2)?
            .reshape((b, seq, self.num_heads * self.head_dim))?;
        if let Some(gate) = gate {
            attn = (attn * candle_nn::ops::sigmoid(&gate)?)?;
        }
        let out = self.o_proj.forward_with(&attn, use_adapters)?;

        if let Some((captured, layer)) = capture {
            record_io(captured, layer, ATTN_O_PROJ, &attn, &out)?;
        }

        Ok(out)
    }
}

impl GatedDeltaNet {
    /// `Qwen3_5GatedDeltaNet.forward`. `padding` is the (batch, seq) float mask of valid tokens.
    fn forward(
        &self,
        x: &Tensor,
        padding: Option<&Tensor>,
        cache: Option<&mut Option<(Tensor, Tensor)>>,
        use_adapters: bool,
        capture: Option<(&mut Captured, usize)>,
    ) -> Result<Tensor> {
        let dtype = x.dtype();
        // apply_mask_to_padding_states
        let x = match padding {
            Some(mask) => x.broadcast_mul(&mask.to_dtype(dtype)?.unsqueeze(2)?)?,
            None => x.clone(),
        };
        let (b, seq, _) = x.dims3()?;
        let d = &self.dims;
        let key_dim = d.num_key_heads * d.key_head_dim;
        let value_dim = d.num_value_heads * d.value_head_dim;

        let mixed = self
            .in_proj_qkv
            .forward(&x)?
            .transpose(1, 2)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let z = self.in_proj_z.forward(&x)?;
        let beta_logits = self.in_proj_b.forward(&x)?;
        let a = self.in_proj_a.forward(&x)?;

        let previous = cache.as_ref().and_then(|slot| slot.as_ref().cloned());
        let (conv_state, recurrent_state) = match &previous {
            Some((conv, rec)) => (Some(conv), Some(rec)),
            None => (None, None),
        };

        let (mixed, new_conv) = deltanet::causal_conv1d(&mixed, &self.conv_weight, conv_state)?;
        let mixed = mixed.to_dtype(dtype)?.transpose(1, 2)?.contiguous()?;

        let query = mixed
            .narrow(2, 0, key_dim)?
            .reshape((b, seq, d.num_key_heads, d.key_head_dim))?;
        let key = mixed
            .narrow(2, key_dim, key_dim)?
            .reshape((b, seq, d.num_key_heads, d.key_head_dim))?;
        let value = mixed
            .narrow(2, 2 * key_dim, value_dim)?
            .reshape((b, seq, d.num_value_heads, d.value_head_dim))?;

        let beta = candle_nn::ops::sigmoid(&beta_logits)?;
        // If the model is loaded in fp16, A could become -inf without the float32 cast.
        let g =
            deltanet::softplus(&a.to_dtype(DType::F32)?.broadcast_add(&self.dt_bias)?)?.broadcast_mul(&self.a_neg)?;

        let (query, key) = if d.num_value_heads > d.num_key_heads {
            let r = d.num_value_heads / d.num_key_heads;
            let interleave = |t: &Tensor| -> Result<Tensor> {
                Ok(t.unsqueeze(3)?
                    .expand((b, seq, d.num_key_heads, r, d.key_head_dim))?
                    .reshape((b, seq, d.num_value_heads, d.key_head_dim))?)
            };
            (interleave(&query)?, interleave(&key)?)
        } else {
            (query, key)
        };

        let (core, state) = if recurrent_state.is_some() && seq == 1 {
            deltanet::recurrent_gated_delta_rule(&query, &key, &value, &g, &beta, recurrent_state)?
        } else {
            deltanet::chunk_gated_delta_rule(&query, &key, &value, &g, &beta, recurrent_state)?
        };

        if let Some(slot) = cache {
            *slot = Some((new_conv, state));
        }

        // Qwen3_5RMSNormGated: norm before gate.
        let core = core
            .to_dtype(dtype)?
            .reshape((b * seq * d.num_value_heads, d.value_head_dim))?;
        let z = z.reshape((b * seq * d.num_value_heads, d.value_head_dim))?;
        let core32 = core.to_dtype(DType::F32)?;
        let variance = core32.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = core32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let normed = normed.to_dtype(dtype)?.broadcast_mul(&self.norm_weight)?;
        let gated = (normed.to_dtype(DType::F32)? * z.to_dtype(DType::F32)?.silu()?)?.to_dtype(dtype)?;
        let gated = gated.reshape((b, seq, value_dim))?;

        let out = self.out_proj.forward_with(&gated, use_adapters)?;
        if let Some((captured, layer)) = capture {
            record_io(captured, layer, ATTN_O_PROJ, &gated, &out)?;
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
    /// Embedding multiplier rounded to the model dtype (Gemma), if any.
    embed_scale: Option<f64>,
    per_layer_inputs: Option<PerLayerInputs>,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Tensor,
    ropes: Vec<Rope>,
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

        let style = config.norm_style;
        let eps = config.rms_norm_eps;
        let norm = |name: &str| -> Result<RmsNorm> {
            // Like Transformers, the weight is first cast to the model dtype.
            let w = load(name)?;
            let weight = match style {
                NormStyle::Llama => w,
                NormStyle::OffsetF32 => (w.to_dtype(DType::F32)? + 1.0)?,
                NormStyle::PlainF32 => w.to_dtype(DType::F32)?,
            };
            Ok(RmsNorm {
                weight: Some(weight),
                eps,
                style,
            })
        };

        let embed_tokens = load("embed_tokens.weight")?;
        let embed_scale = if config.embed_scale {
            Some(round_to_dtype((config.hidden_size as f64).sqrt(), dtype)?)
        } else {
            None
        };

        let per_layer_inputs = match config.per_layer_input {
            Some((dim, _)) => Some(PerLayerInputs {
                embed: weights
                    .load_text("embed_tokens_per_layer.weight", dtype, &Device::Cpu)?
                    .1,
                embed_scale: round_to_dtype((dim as f64).sqrt(), dtype)?,
                model_projection: load("per_layer_model_projection.weight")?,
                projection_norm: norm("per_layer_projection_norm.weight")?,
                dim,
            }),
            None => None,
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (i, spec) in config.layers.iter().enumerate() {
            let p = format!("layers.{i}");

            let mixer = match spec.kind {
                LayerKind::Attention => {
                    let a = format!("{p}.self_attn");
                    let shared = spec.kv_sharing == KvSharing::Reuse;
                    Mixer::Attention(Box::new(Attention {
                        q_proj: linear(&format!("{a}.q_proj"), config.qkv_bias)?,
                        k_proj: if shared {
                            None
                        } else {
                            Some(linear(&format!("{a}.k_proj"), config.qkv_bias)?)
                        },
                        v_proj: if shared || spec.k_eq_v {
                            None
                        } else {
                            Some(linear(&format!("{a}.v_proj"), config.qkv_bias)?)
                        },
                        o_proj: lora_linear(&format!("{a}.o_proj"), config.o_bias)?,
                        q_norm: if config.qk_norm {
                            Some(norm(&format!("{a}.q_norm.weight"))?)
                        } else {
                            None
                        },
                        k_norm: if config.qk_norm && !shared {
                            Some(norm(&format!("{a}.k_norm.weight"))?)
                        } else {
                            None
                        },
                        v_norm: if config.v_norm && !shared {
                            Some(RmsNorm {
                                weight: None,
                                eps,
                                style,
                            })
                        } else {
                            None
                        },
                        num_heads: config.num_attention_heads,
                        num_kv_heads: spec.num_key_value_heads,
                        head_dim: spec.head_dim,
                        scaling: config.attn_scaling.unwrap_or((spec.head_dim as f64).powf(-0.5)),
                        softcap: config.attn_softcap,
                        output_gate: config.attn_output_gate,
                        kv_sharing: spec.kv_sharing,
                    }))
                }
                LayerKind::LinearAttention => {
                    let l = format!("{p}.linear_attn");
                    let dims = config
                        .linear_attention
                        .clone()
                        .context("linear attention layer without linear_* settings in config.json")?;
                    let conv = load(&format!("{l}.conv1d.weight"))?;
                    let (channels, _, kernel) = conv.dims3()?;
                    let a_log = weights.load_text(&format!("{l}.A_log"), DType::F32, device)?.1;
                    Mixer::Linear(Box::new(GatedDeltaNet {
                        in_proj_qkv: linear(&format!("{l}.in_proj_qkv"), false)?,
                        in_proj_z: linear(&format!("{l}.in_proj_z"), false)?,
                        in_proj_b: linear(&format!("{l}.in_proj_b"), false)?,
                        in_proj_a: linear(&format!("{l}.in_proj_a"), false)?,
                        conv_weight: conv.reshape((channels, kernel))?.to_dtype(DType::F32)?,
                        dt_bias: load(&format!("{l}.dt_bias"))?.to_dtype(DType::F32)?,
                        a_neg: a_log.exp()?.neg()?,
                        norm_weight: load(&format!("{l}.norm.weight"))?,
                        out_proj: lora_linear(&format!("{l}.out_proj"), false)?,
                        dims,
                        eps,
                    }))
                }
            };

            let per_layer = if config.per_layer_input.is_some() {
                Some(PerLayerBlock {
                    gate: linear(&format!("{p}.per_layer_input_gate"), false)?,
                    projection: linear(&format!("{p}.per_layer_projection"), false)?,
                    norm: norm(&format!("{p}.post_per_layer_input_norm.weight"))?,
                })
            } else {
                None
            };

            let sandwich = config.sandwich_norms;
            layers.push(Layer {
                input_layernorm: norm(&format!("{p}.input_layernorm.weight"))?,
                mixer,
                post_attention_layernorm: norm(&format!("{p}.post_attention_layernorm.weight"))?,
                pre_feedforward_layernorm: if sandwich {
                    Some(norm(&format!("{p}.pre_feedforward_layernorm.weight"))?)
                } else {
                    None
                },
                post_feedforward_layernorm: if sandwich {
                    Some(norm(&format!("{p}.post_feedforward_layernorm.weight"))?)
                } else {
                    None
                },
                mlp: Mlp {
                    gate_proj: linear(&format!("{p}.mlp.gate_proj"), config.mlp_bias)?,
                    up_proj: linear(&format!("{p}.mlp.up_proj"), config.mlp_bias)?,
                    down_proj: lora_linear(&format!("{p}.mlp.down_proj"), config.mlp_bias)?,
                    activation: config.hidden_act,
                },
                per_layer,
                layer_scalar: load_optional(&format!("{p}.layer_scalar"))?,
                sliding: spec.sliding_window.is_some(),
                rope: spec.rope,
            });
        }

        let norm = norm("norm.weight")?;

        // Checkpoints without a separate head always tie it to the embeddings.
        let lm_head = match weights.lm_head_name() {
            Some(name) if !config.tie_word_embeddings => weights.load(&name, dtype, device)?,
            _ => embed_tokens.clone(),
        };

        let ropes = config
            .ropes
            .iter()
            .map(|spec| Rope::new(spec, config.max_position_embeddings, device))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            config,
            embed_tokens,
            embed_scale,
            per_layer_inputs,
            layers,
            norm,
            lm_head,
            ropes,
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

    /// The modifiable modules of a layer, by component name. Linear attention output
    /// projections count as "attn.o_proj", like in the Python implementation.
    pub fn layer_modules(&self, layer: usize) -> BTreeMap<String, Vec<Arc<LoraLinear>>> {
        let layer = &self.layers[layer];
        let mut modules = BTreeMap::new();
        let o_proj = match &layer.mixer {
            Mixer::Attention(attention) => attention.o_proj.clone(),
            Mixer::Linear(linear) => linear.out_proj.clone(),
        };
        modules.insert(ATTN_O_PROJ.to_string(), vec![o_proj]);
        modules.insert(MLP_DOWN_PROJ.to_string(), vec![layer.mlp.down_proj.clone()]);
        modules
    }

    /// Valid-token mask (batch, seq) for a prefill pass, read from the diagonal of the
    /// additive attention mask (a padding position cannot even attend to itself).
    fn padding_mask(mask: &Tensor) -> Result<Option<Tensor>> {
        let (_, _, seq, kv) = mask.dims4()?;
        if seq != kv || seq < 2 {
            return Ok(None);
        }
        let eye = Tensor::eye(seq, DType::F32, mask.device())?;
        let diagonal = mask.squeeze(1)?.broadcast_mul(&eye)?.sum(D::Minus1)?;
        // 0 for valid tokens, f32::MIN for padding.
        Ok(Some(diagonal.ge(-1.0f32)?.to_dtype(DType::F32)?))
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
        if let Some(scale) = self.embed_scale {
            x = (x * scale)?;
        }

        // Gemma 4 per-layer embeddings: (batch, seq, layers, dim).
        let per_layer_inputs = match &self.per_layer_inputs {
            Some(ple) => {
                let layers = self.layers.len();
                let ids = input.input_ids.flatten_all()?.to_device(&Device::Cpu)?;
                let token = ple
                    .embed
                    .index_select(&ids, 0)?
                    .to_device(&self.device)?
                    .reshape((b, seq, layers, ple.dim))?;
                let token = (token * ple.embed_scale)?;
                let projection = (matmul_t(&x, &ple.model_projection)? * (self.config.hidden_size as f64).powf(-0.5))?
                    .reshape((b, seq, layers, ple.dim))?;
                let projection = ple.projection_norm.forward(&projection)?;
                Some(((projection + token)? * 2f64.powf(-0.5))?)
            }
            None => None,
        };

        let cos_sin = self
            .ropes
            .iter()
            .map(|rope| rope.cos_sin(input.positions, self.dtype))
            .collect::<Result<Vec<_>>>()?;
        let padding = if self.config.is_hybrid() {
            Self::padding_mask(input.mask)?
        } else {
            None
        };
        let mut shared: Vec<Option<(Tensor, Tensor)>> = vec![None; self.ropes.len()];
        let last = |t: &Tensor| -> Result<Tensor> { Ok(t.i((.., seq - 1, ..))?) };

        for (index, layer) in self.layers.iter().enumerate() {
            if request.hidden_states {
                captured.hidden_states.push(last(&x)?);
            }

            let slot = cache.as_deref_mut().map(|c| &mut c.layers[index]);
            let capture = request.module_io.then_some((&mut *captured, index));

            let residual = x.clone();
            let h = layer.input_layernorm.forward(&x)?;
            let h = match &layer.mixer {
                Mixer::Attention(attention) => {
                    let mask = match (layer.sliding, input.sliding_mask) {
                        (true, Some(sliding_mask)) => sliding_mask,
                        _ => input.mask,
                    };
                    let (cos, sin) = &cos_sin[layer.rope];
                    attention.forward(
                        &h,
                        &self.ropes[layer.rope],
                        cos,
                        sin,
                        mask,
                        slot,
                        &mut shared[layer.rope],
                        input.use_adapters,
                        capture,
                    )?
                }
                Mixer::Linear(linear) => linear.forward(&h, padding.as_ref(), slot, input.use_adapters, capture)?,
            };

            let capture = request.module_io.then_some((&mut *captured, index));
            if let (Some(pre), Some(post)) = (&layer.pre_feedforward_layernorm, &layer.post_feedforward_layernorm) {
                let h = layer.post_attention_layernorm.forward(&h)?;
                x = (residual + h)?;
                let residual = x.clone();
                let h = layer.mlp.forward(&pre.forward(&x)?, input.use_adapters, capture)?;
                x = (residual + post.forward(&h)?)?;
            } else {
                x = (residual + h)?;
                let residual = x.clone();
                let h = layer.post_attention_layernorm.forward(&x)?;
                let h = layer.mlp.forward(&h, input.use_adapters, capture)?;
                x = (residual + h)?;
            }

            if let (Some(block), Some(inputs)) = (&layer.per_layer, &per_layer_inputs) {
                let residual = x.clone();
                let h = activate(&block.gate.forward(&x)?, self.config.hidden_act)?;
                let h = (h * inputs.narrow(2, index, 1)?.squeeze(2)?)?;
                let h = block.norm.forward(&block.projection.forward(&h)?)?;
                x = (residual + h)?;
            }

            if let Some(scalar) = &layer.layer_scalar {
                x = x.broadcast_mul(scalar)?;
            }
        }

        if let Some(cache) = cache {
            cache.seq_len += seq;
        }

        let x = self.norm.forward(&x)?;

        if request.hidden_states {
            captured.hidden_states.push(last(&x)?);
        }

        let logits = if input.all_logits {
            matmul_t(&x, &self.lm_head)?
        } else {
            matmul_t(&last(&x)?, &self.lm_head)?
        };
        Ok(match self.config.final_softcap {
            Some(cap) => softcap(&logits, cap)?,
            None => logits,
        })
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

    #[test]
    fn padding_mask_from_diagonal() -> Result<()> {
        let (_, mask) = prefill_inputs(&[vec![0, 0, 1, 1], vec![1, 1, 1, 1]], None, &Device::Cpu)?;
        let padding = Transformer::padding_mask(&mask)?.unwrap();
        assert_eq!(
            padding.to_vec2::<f32>()?,
            vec![vec![0.0, 0.0, 1.0, 1.0], vec![1.0, 1.0, 1.0, 1.0]]
        );
        Ok(())
    }

    #[test]
    fn proportional_rope_rotates_only_a_fraction() -> Result<()> {
        let spec = RopeSpec {
            theta: 10000.0,
            scaling: RopeScaling::Proportional { factor: 1.0 },
            partial_rotary_factor: 0.25,
            head_dim: 16,
        };
        let rope = Rope::new(&spec, 4096, &Device::Cpu)?;
        let freqs = rope.inv_freq.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(freqs.len(), 8);
        assert_eq!(freqs[0], 1.0);
        assert!(freqs[1] > 0.0 && freqs[2] == 0.0);
        Ok(())
    }
}
