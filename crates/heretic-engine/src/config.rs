// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Parsing of Transformers `config.json` files into an architecture description.
//!
//! One generic decoder implementation (`transformer.rs`) covers every supported architecture;
//! the differences between them are expressed through the flags and per-layer specifications
//! built here. Each family mirrors its `configuration_*.py` / `modeling_*.py` in Transformers.

use anyhow::{Result, bail};
use serde_json::Value;

/// Rotary position embedding scaling variants.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScaling {
    Default,
    Linear {
        factor: f64,
    },
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max_position_embeddings: f64,
    },
    Yarn {
        factor: f64,
        original_max_position_embeddings: f64,
        beta_fast: f64,
        beta_slow: f64,
        attention_factor: Option<f64>,
        mscale: Option<f64>,
        mscale_all_dim: Option<f64>,
        truncate: bool,
    },
    /// Gemma 4 "proportional" RoPE (`_compute_proportional_rope_parameters`): the frequencies span
    /// the whole head dimension, with zeros for the non-rotated part.
    Proportional {
        factor: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Silu,
    GeluTanh,
    Gelu,
}

impl Activation {
    fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "silu" | "swish" => Activation::Silu,
            "gelu_pytorch_tanh" | "gelu_new" | "gelu_fast" => Activation::GeluTanh,
            "gelu" => Activation::Gelu,
            other => bail!("Unsupported activation function \"{other}\" in config.json"),
        })
    }
}

/// Architecture families with their own numerics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Llama, Mistral, Qwen2, Qwen3.
    Standard,
    /// Gemma (1).
    Gemma,
    Gemma2,
    Gemma3,
    Gemma4,
    /// Qwen3.5 (hybrid Gated DeltaNet / gated attention).
    Qwen35,
}

/// How RMSNorm applies its weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormStyle {
    /// Llama: normalize in float32, cast back, multiply by the weight in the model dtype.
    Llama,
    /// Gemma 1-3 and Qwen3.5: multiply by `1 + weight` in float32, then cast.
    OffsetF32,
    /// Gemma 4: multiply by `weight` in float32, then cast.
    PlainF32,
}

/// The rotary embedding used by a group of layers.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeSpec {
    pub theta: f64,
    pub scaling: RopeScaling,
    /// Fraction of the head dimension that receives rotary embeddings.
    pub partial_rotary_factor: f64,
    pub head_dim: usize,
}

impl RopeSpec {
    /// Number of rotated dimensions per head (the whole head for proportional RoPE).
    pub fn rotary_dim(&self) -> usize {
        match self.scaling {
            RopeScaling::Proportional { .. } => self.head_dim,
            _ => ((self.head_dim as f64 * self.partial_rotary_factor) as usize) & !1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Softmax attention.
    Attention,
    /// Gated DeltaNet linear attention (Qwen3.5).
    LinearAttention,
}

/// Gemma 4 key/value sharing between layers (`Gemma4TextConfig.kv_sharing_modes`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvSharing {
    None,
    /// The last unshared layer of its type: its keys/values are reused by later layers.
    Store,
    /// Reuses the keys/values of the storing layer of the same type; has no k/v projections.
    Reuse,
}

/// Per-layer architecture.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerSpec {
    pub kind: LayerKind,
    pub head_dim: usize,
    pub num_key_value_heads: usize,
    /// Sliding window size (`None` = full attention).
    pub sliding_window: Option<usize>,
    /// Index into `TextConfig::ropes`.
    pub rope: usize,
    pub kv_sharing: KvSharing,
    /// Gemma 4 `attention_k_eq_v`: the key projection doubles as the value projection.
    pub k_eq_v: bool,
    pub intermediate_size: usize,
}

/// Gated DeltaNet dimensions (Qwen3.5 `linear_*` settings).
#[derive(Debug, Clone, PartialEq)]
pub struct LinearAttentionConfig {
    pub num_key_heads: usize,
    pub num_value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel: usize,
}

/// Description of a decoder-only text transformer.
#[derive(Debug, Clone)]
pub struct TextConfig {
    /// Transformers `model_type` of the text model (e.g. "qwen2").
    pub model_type: String,
    pub family: Family,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    /// RoPE of the first layer type (kept for callers that only need one).
    pub rope_theta: f64,
    pub rope_scaling: RopeScaling,
    pub partial_rotary_factor: f64,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub hidden_act: Activation,
    /// Bias on the q/k/v projections.
    pub qkv_bias: bool,
    /// Bias on the attention output projection.
    pub o_bias: bool,
    /// Bias on the MLP projections.
    pub mlp_bias: bool,
    /// Per-head RMSNorm on queries and keys (Qwen3, Gemma 3/4, Qwen3.5).
    pub qk_norm: bool,
    /// Sliding window size per layer (`None` = full attention).
    pub sliding_window: Vec<Option<usize>>,
    /// Dtype the checkpoint was saved in ("bfloat16", ...).
    pub torch_dtype: Option<String>,

    pub layers: Vec<LayerSpec>,
    pub ropes: Vec<RopeSpec>,
    pub norm_style: NormStyle,
    /// Gemma: embeddings are multiplied by `sqrt(hidden_size)` (rounded to the model dtype).
    pub embed_scale: bool,
    /// Gemma 2: `attn_logit_softcapping`.
    pub attn_softcap: Option<f64>,
    /// Gemma 2/3/4: `final_logit_softcapping`.
    pub final_softcap: Option<f64>,
    /// Attention score scaling (`None` = `head_dim ** -0.5`).
    pub attn_scaling: Option<f64>,
    /// Gemma 2+: post-attention and pre/post-feedforward norms around each block.
    pub sandwich_norms: bool,
    /// Gemma 4: scale-free RMSNorm on values.
    pub v_norm: bool,
    /// Qwen3.5: `attn_output_gate`, the query projection also produces a sigmoid gate.
    pub attn_output_gate: bool,
    pub linear_attention: Option<LinearAttentionConfig>,
    /// Gemma 4 per-layer embeddings: (`hidden_size_per_layer_input`, `vocab_size_per_layer_input`).
    pub per_layer_input: Option<(usize, usize)>,
}

fn get_f64(config: &Value, key: &str) -> Option<f64> {
    config.get(key).and_then(Value::as_f64)
}

fn get_usize(config: &Value, key: &str) -> Option<usize> {
    config.get(key).and_then(Value::as_u64).map(|v| v as usize)
}

fn get_bool(config: &Value, key: &str) -> Option<bool> {
    config.get(key).and_then(Value::as_bool)
}

fn require_usize(config: &Value, key: &str) -> Result<usize> {
    get_usize(config, key).ok_or_else(|| anyhow::anyhow!("config.json is missing \"{key}\""))
}

/// Returns the text configuration of a (possibly multimodal) model config,
/// like `PretrainedConfig.get_text_config()`.
pub fn text_config_json(config: &Value) -> &Value {
    for key in ["text_config", "llm_config", "language_config"] {
        if let Some(text) = config.get(key).filter(|v| v.is_object()) {
            return text;
        }
    }
    config
}

/// Whether the model has a vision tower (and therefore an image processor).
pub fn is_multimodal(config: &Value) -> bool {
    config.get("vision_config").is_some_and(|v| !v.is_null())
}

/// Builds a RoPE specification from a `rope_parameters`/`rope_scaling` object (or none).
fn rope_spec(parameters: Option<&Value>, config: &Value, theta_default: f64, head_dim: usize) -> Result<RopeSpec> {
    let theta = parameters
        .and_then(|p| get_f64(p, "rope_theta"))
        .or_else(|| get_f64(config, "rope_theta"))
        .unwrap_or(theta_default);

    let partial = parameters
        .and_then(|p| get_f64(p, "partial_rotary_factor"))
        .or_else(|| get_f64(config, "partial_rotary_factor"))
        .unwrap_or(1.0);

    let Some(parameters) = parameters else {
        return Ok(RopeSpec {
            theta,
            scaling: RopeScaling::Default,
            partial_rotary_factor: partial,
            head_dim,
        });
    };

    let rope_type = parameters
        .get("rope_type")
        .or_else(|| parameters.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("default");

    let factor = get_f64(parameters, "factor").unwrap_or(1.0);
    let original = get_f64(parameters, "original_max_position_embeddings")
        .or_else(|| get_f64(config, "original_max_position_embeddings"))
        .or_else(|| get_f64(config, "max_position_embeddings"))
        .unwrap_or(2048.0);

    let scaling = match rope_type {
        "default" => RopeScaling::Default,
        "linear" => RopeScaling::Linear { factor },
        "llama3" => RopeScaling::Llama3 {
            factor,
            low_freq_factor: get_f64(parameters, "low_freq_factor").unwrap_or(1.0),
            high_freq_factor: get_f64(parameters, "high_freq_factor").unwrap_or(4.0),
            original_max_position_embeddings: original,
        },
        "yarn" => RopeScaling::Yarn {
            factor,
            original_max_position_embeddings: original,
            beta_fast: get_f64(parameters, "beta_fast").unwrap_or(32.0),
            beta_slow: get_f64(parameters, "beta_slow").unwrap_or(1.0),
            attention_factor: get_f64(parameters, "attention_factor"),
            mscale: get_f64(parameters, "mscale"),
            mscale_all_dim: get_f64(parameters, "mscale_all_dim"),
            truncate: get_bool(parameters, "truncate").unwrap_or(true),
        },
        "proportional" => RopeScaling::Proportional { factor },
        // Dynamic NTK scaling only changes frequencies beyond the original context length,
        // which Heretic's short prompts never reach.
        "dynamic" => RopeScaling::Default,
        other => bail!("Unsupported RoPE type \"{other}\" in config.json"),
    };

    Ok(RopeSpec {
        theta,
        scaling,
        partial_rotary_factor: partial,
        head_dim,
    })
}

/// The `rope_parameters`/`rope_scaling` object of a config with a single RoPE.
fn single_rope_parameters<'a>(config: &'a Value, root: &'a Value) -> Option<&'a Value> {
    config
        .get("rope_parameters")
        .or_else(|| config.get("rope_scaling"))
        .or_else(|| root.get("rope_scaling"))
        .filter(|v| v.is_object())
}

/// The architectures supported by the generic transformer implementation.
pub const SUPPORTED_MODEL_TYPES: &[&str] = &[
    "llama",
    "mistral",
    "qwen2",
    "qwen3",
    "qwen3_5_text",
    "gemma",
    "gemma2",
    "gemma3_text",
    "gemma4_text",
];

/// Human-readable list of supported families, for error messages.
pub const SUPPORTED_FAMILIES: &str =
    "Llama 2/3, Mistral, Qwen2/2.5, Qwen3, Qwen3.5 (dense), Gemma, Gemma 2, Gemma 3, Gemma 4 (dense)";

/// Explains why a model type is not supported, with a hint where one is known.
fn unsupported(model_type: &str, architectures: &str) -> anyhow::Error {
    let hint = match model_type {
        t if t.contains("moe")
            || [
                "mixtral",
                "gpt_oss",
                "deepseek_v3",
                "deepseek_v4",
                "phimoe",
                "qwen3_next",
            ]
            .contains(&t) =>
        {
            "Mixture-of-experts models are not supported by this version yet. Use a dense model of the same family."
        }
        t if t.starts_with("gemma3n") => {
            "Gemma 3n uses AltUp and Laurel blocks, which are not supported. Use Gemma 3 or Gemma 4 instead."
        }
        "" => {
            "config.json has no \"model_type\". Make sure you selected the original Transformers model folder, \
             not a GGUF or ONNX export."
        }
        _ => "This architecture is not implemented yet.",
    };
    let shown = if model_type.is_empty() { "unknown" } else { model_type };
    let architectures = if architectures.is_empty() {
        String::new()
    } else {
        format!(" ({architectures})")
    };
    anyhow::anyhow!("Unsupported model type \"{shown}\"{architectures}. {hint} Supported: {SUPPORTED_FAMILIES}.")
}

/// Layer types (`sliding_attention`, `full_attention`, `linear_attention`) per layer.
fn layer_types(config: &Value, family: Family, layers: usize) -> Vec<String> {
    if let Some(types) = config.get("layer_types").and_then(Value::as_array) {
        let mut types: Vec<String> = types
            .iter()
            .map(|t| t.as_str().unwrap_or("full_attention").to_string())
            .chain(std::iter::repeat("full_attention".to_string()))
            .take(layers)
            .collect();
        if family == Family::Gemma4
            && let Some(last) = types.last_mut()
        {
            // Gemma4TextConfig forces the last layer to full attention.
            *last = "full_attention".into();
        }
        return types;
    }

    let pattern = |period: usize, sliding: &str| -> Vec<String> {
        (0..layers)
            .map(|i| {
                if (i + 1) % period != 0 {
                    sliding.to_string()
                } else {
                    "full_attention".to_string()
                }
            })
            .collect()
    };

    match family {
        // Gemma2Config: sliding attention on even layers.
        Family::Gemma2 => pattern(2, "sliding_attention"),
        Family::Gemma3 => pattern(
            get_usize(config, "sliding_window_pattern").unwrap_or(6),
            "sliding_attention",
        ),
        Family::Gemma4 => {
            let mut types = pattern(6, "sliding_attention");
            if let Some(last) = types.last_mut() {
                *last = "full_attention".into();
            }
            types
        }
        Family::Qwen35 => pattern(
            get_usize(config, "full_attention_interval").unwrap_or(4),
            "linear_attention",
        ),
        Family::Standard | Family::Gemma => vec!["full_attention".into(); layers],
    }
}

impl TextConfig {
    pub fn from_json(root: &Value) -> Result<TextConfig> {
        let config = text_config_json(root);

        let model_type = config
            .get("model_type")
            .and_then(Value::as_str)
            .or_else(|| root.get("model_type").and_then(Value::as_str))
            .unwrap_or_default()
            .to_string();

        let architectures = root
            .get("architectures")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(", "))
            .unwrap_or_default();

        let family = match model_type.as_str() {
            "llama" | "mistral" | "qwen2" | "qwen3" => Family::Standard,
            "qwen3_5_text" | "qwen3_5" => Family::Qwen35,
            "gemma" => Family::Gemma,
            "gemma2" => Family::Gemma2,
            "gemma3_text" | "gemma3" => Family::Gemma3,
            "gemma4_text" | "gemma4" => Family::Gemma4,
            _ => return Err(unsupported(&model_type, &architectures)),
        };

        if family == Family::Gemma4 && get_bool(config, "enable_moe_block").unwrap_or(false) {
            return Err(unsupported("gemma4_moe", &architectures));
        }

        let hidden_size = require_usize(config, "hidden_size")?;
        let num_attention_heads = require_usize(config, "num_attention_heads")?;
        let num_hidden_layers = require_usize(config, "num_hidden_layers")?;
        let num_key_value_heads = get_usize(config, "num_key_value_heads").unwrap_or(num_attention_heads);
        let head_dim = get_usize(config, "head_dim").unwrap_or(hidden_size / num_attention_heads);
        let intermediate_size = require_usize(config, "intermediate_size")?;

        let gemma = matches!(family, Family::Gemma | Family::Gemma2 | Family::Gemma3 | Family::Gemma4);

        let attention_bias = get_bool(config, "attention_bias").unwrap_or(false);
        let (qkv_bias, o_bias, qk_norm) = match family {
            // Qwen2 always has q/k/v biases and never an o_proj bias.
            Family::Standard if model_type == "qwen2" => (true, false, false),
            Family::Standard if model_type == "qwen3" => (attention_bias, attention_bias, true),
            Family::Standard | Family::Gemma | Family::Gemma2 => (attention_bias, attention_bias, false),
            Family::Gemma3 | Family::Gemma4 | Family::Qwen35 => (attention_bias, attention_bias, true),
        };

        let types = layer_types(config, family, num_hidden_layers);
        let window = get_usize(config, "sliding_window");

        // RoPE per layer type.
        let mut ropes: Vec<RopeSpec> = Vec::new();
        let mut rope_by_type: Vec<(String, usize)> = Vec::new();
        let per_type_rope = config
            .get("rope_parameters")
            .filter(|p| p.get("full_attention").is_some() || p.get("sliding_attention").is_some());

        let global_head_dim = if family == Family::Gemma4 {
            get_usize(config, "global_head_dim").unwrap_or(512)
        } else {
            head_dim
        };

        let mut unique_types: Vec<String> = types.clone();
        unique_types.sort();
        unique_types.dedup();
        for layer_type in &unique_types {
            let rope_head_dim = if layer_type == "full_attention" {
                global_head_dim
            } else {
                head_dim
            };
            let spec = if let Some(per_type) = per_type_rope {
                rope_spec(
                    per_type.get(layer_type).filter(|v| v.is_object()),
                    config,
                    10000.0,
                    rope_head_dim,
                )?
            } else if family == Family::Gemma3 && layer_type == "sliding_attention" {
                // Old Gemma 3 configs: local layers use the default RoPE with `rope_local_base_freq`.
                RopeSpec {
                    theta: get_f64(config, "rope_local_base_freq").unwrap_or(10000.0),
                    scaling: RopeScaling::Default,
                    partial_rotary_factor: 1.0,
                    head_dim,
                }
            } else {
                rope_spec(single_rope_parameters(config, root), config, 10000.0, rope_head_dim)?
            };
            rope_by_type.push((layer_type.clone(), ropes.len()));
            ropes.push(spec);
        }
        let rope_index = |t: &str| rope_by_type.iter().find(|(name, _)| name == t).map_or(0, |(_, i)| *i);

        // Gemma 4 key/value sharing (`Gemma4TextConfig.kv_sharing_modes`).
        let shared = if family == Family::Gemma4 {
            get_usize(config, "num_kv_shared_layers").unwrap_or(0)
        } else {
            0
        };
        let mut sharing = vec![KvSharing::None; num_hidden_layers];
        if shared > 0 && shared < num_hidden_layers {
            let first_shared = num_hidden_layers - shared;
            for mode in sharing.iter_mut().skip(first_shared) {
                *mode = KvSharing::Reuse;
            }
            for layer_type in &unique_types {
                match types[..first_shared].iter().rposition(|t| t == layer_type) {
                    Some(producer) => sharing[producer] = KvSharing::Store,
                    None => bail!(
                        "config.json shares the key/value cache of \"{layer_type}\" layers, but no earlier layer of \
                         that type produces it"
                    ),
                }
            }
        }

        let k_eq_v = family == Family::Gemma4 && get_bool(config, "attention_k_eq_v").unwrap_or(false);
        let global_kv_heads = if family == Family::Gemma4 && k_eq_v {
            get_usize(config, "num_global_key_value_heads")
        } else {
            None
        };
        let double_wide = family == Family::Gemma4 && get_bool(config, "use_double_wide_mlp").unwrap_or(false);

        let layers: Vec<LayerSpec> = (0..num_hidden_layers)
            .map(|i| {
                let layer_type = types[i].as_str();
                let full = layer_type == "full_attention";
                let reuse = sharing[i] == KvSharing::Reuse;
                LayerSpec {
                    kind: if layer_type == "linear_attention" {
                        LayerKind::LinearAttention
                    } else {
                        LayerKind::Attention
                    },
                    head_dim: if full { global_head_dim } else { head_dim },
                    num_key_value_heads: match (full, global_kv_heads) {
                        (true, Some(heads)) => heads,
                        _ => num_key_value_heads,
                    },
                    sliding_window: if layer_type == "sliding_attention" {
                        window
                    } else {
                        None
                    },
                    rope: rope_index(layer_type),
                    kv_sharing: sharing[i],
                    k_eq_v: k_eq_v && full,
                    intermediate_size: if double_wide && reuse {
                        intermediate_size * 2
                    } else {
                        intermediate_size
                    },
                }
            })
            .collect();

        // Old Qwen2/Qwen3 sliding window settings without explicit layer types.
        let mut sliding_window: Vec<Option<usize>> = layers.iter().map(|l| l.sliding_window).collect();
        if config.get("layer_types").is_none() {
            match model_type.as_str() {
                "mistral" => sliding_window = vec![window; num_hidden_layers],
                "qwen2" | "qwen3" if get_bool(config, "use_sliding_window").unwrap_or(false) => {
                    let max_window_layers = get_usize(config, "max_window_layers").unwrap_or(num_hidden_layers);
                    sliding_window = (0..num_hidden_layers)
                        .map(|i| if i >= max_window_layers { window } else { None })
                        .collect();
                }
                _ => {}
            }
        }
        let mut layers = layers;
        for (layer, window) in layers.iter_mut().zip(&sliding_window) {
            layer.sliding_window = *window;
        }

        let linear_attention = if family == Family::Qwen35 {
            Some(LinearAttentionConfig {
                num_key_heads: require_usize(config, "linear_num_key_heads")?,
                num_value_heads: require_usize(config, "linear_num_value_heads")?,
                key_head_dim: require_usize(config, "linear_key_head_dim")?,
                value_head_dim: require_usize(config, "linear_value_head_dim")?,
                conv_kernel: get_usize(config, "linear_conv_kernel_dim").unwrap_or(4),
            })
        } else {
            None
        };

        let per_layer_input = match get_usize(config, "hidden_size_per_layer_input") {
            Some(dim) if dim > 0 && family == Family::Gemma4 => Some((
                dim,
                get_usize(config, "vocab_size_per_layer_input").unwrap_or(require_usize(config, "vocab_size")?),
            )),
            _ => None,
        };

        let activation_name = if gemma {
            config
                .get("hidden_activation")
                .and_then(Value::as_str)
                .unwrap_or("gelu_pytorch_tanh")
        } else {
            config.get("hidden_act").and_then(Value::as_str).unwrap_or("silu")
        };

        let attn_scaling = match family {
            Family::Gemma2 | Family::Gemma3 => Some(
                get_f64(config, "query_pre_attn_scalar")
                    .unwrap_or(head_dim as f64)
                    .powf(-0.5),
            ),
            Family::Gemma4 => Some(1.0),
            _ => None,
        };

        let tie_word_embeddings = get_bool(config, "tie_word_embeddings")
            .or_else(|| get_bool(root, "tie_word_embeddings"))
            .unwrap_or(gemma);

        let torch_dtype = config
            .get("dtype")
            .or_else(|| config.get("torch_dtype"))
            .or_else(|| root.get("dtype"))
            .or_else(|| root.get("torch_dtype"))
            .and_then(Value::as_str)
            .map(String::from);

        let first_rope = ropes[layers.first().map_or(0, |l| l.rope)].clone();

        Ok(TextConfig {
            vocab_size: require_usize(config, "vocab_size")?,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps: get_f64(config, "rms_norm_eps").unwrap_or(1e-6),
            rope_theta: first_rope.theta,
            rope_scaling: first_rope.scaling.clone(),
            partial_rotary_factor: first_rope.partial_rotary_factor,
            max_position_embeddings: get_usize(config, "max_position_embeddings").unwrap_or(4096),
            tie_word_embeddings,
            hidden_act: Activation::parse(activation_name)?,
            qkv_bias,
            o_bias,
            mlp_bias: get_bool(config, "mlp_bias").unwrap_or(false),
            qk_norm,
            sliding_window,
            torch_dtype,
            layers,
            ropes,
            norm_style: match family {
                Family::Standard => NormStyle::Llama,
                Family::Gemma4 => NormStyle::PlainF32,
                _ => NormStyle::OffsetF32,
            },
            embed_scale: gemma,
            attn_softcap: if family == Family::Gemma2 {
                get_f64(config, "attn_logit_softcapping")
            } else {
                None
            },
            final_softcap: if gemma {
                get_f64(config, "final_logit_softcapping")
            } else {
                None
            },
            attn_scaling,
            sandwich_norms: matches!(family, Family::Gemma2 | Family::Gemma3 | Family::Gemma4),
            v_norm: family == Family::Gemma4,
            attn_output_gate: family == Family::Qwen35 && get_bool(config, "attn_output_gate").unwrap_or(true),
            linear_attention,
            per_layer_input,
            family,
            model_type,
        })
    }

    /// Number of query heads per key/value head.
    pub fn n_rep(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Number of rotary dimensions per head (of the first layer's RoPE).
    pub fn rotary_dim(&self) -> usize {
        ((self.head_dim as f64 * self.partial_rotary_factor) as usize) & !1
    }

    /// The smallest sliding window of any layer.
    pub fn smallest_window(&self) -> Option<usize> {
        self.sliding_window.iter().flatten().min().copied()
    }

    /// Whether some layers use linear attention (recurrent state instead of a key/value cache).
    pub fn is_hybrid(&self) -> bool {
        self.layers.iter().any(|l| l.kind == LayerKind::LinearAttention)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_qwen2() {
        let json = serde_json::json!({
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 100,
            "rope_theta": 1000000.0, "tie_word_embeddings": true, "torch_dtype": "bfloat16",
            "use_sliding_window": false, "sliding_window": 32, "hidden_act": "silu"
        });
        let config = TextConfig::from_json(&json).unwrap();
        assert_eq!(config.head_dim, 16);
        assert_eq!(config.n_rep(), 2);
        assert!(config.qkv_bias && !config.o_bias);
        assert_eq!(config.rope_theta, 1e6);
        assert_eq!(config.sliding_window, vec![None, None]);
        assert_eq!(config.torch_dtype.as_deref(), Some("bfloat16"));
        assert_eq!(config.family, Family::Standard);
        assert_eq!(config.norm_style, NormStyle::Llama);
    }

    #[test]
    fn parses_multimodal_text_config_and_rope_parameters() {
        let json = serde_json::json!({
            "model_type": "mistral3",
            "vision_config": {"hidden_size": 8},
            "text_config": {
                "model_type": "mistral",
                "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 3,
                "num_attention_heads": 4, "head_dim": 32, "vocab_size": 100,
                "rope_parameters": {"rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0},
                "sliding_window": null
            }
        });
        assert!(is_multimodal(&json));
        let config = TextConfig::from_json(&json).unwrap();
        assert_eq!(config.head_dim, 32);
        assert_eq!(config.rope_theta, 500000.0);
        assert!(matches!(config.rope_scaling, RopeScaling::Llama3 { factor, .. } if factor == 8.0));
        assert_eq!(config.sliding_window, vec![None, None, None]);
    }

    #[test]
    fn rejects_unknown_architecture_with_hint() {
        let json = serde_json::json!({"model_type": "mamba", "architectures": ["MambaForCausalLM"]});
        let error = TextConfig::from_json(&json).unwrap_err().to_string();
        assert!(error.contains("Unsupported model type \"mamba\""));
        assert!(error.contains("Supported:"));

        let json = serde_json::json!({"model_type": "qwen3_moe"});
        let error = TextConfig::from_json(&json).unwrap_err().to_string();
        assert!(error.contains("Mixture-of-experts"));
    }

    #[test]
    fn parses_qwen3_5_hybrid() {
        let json = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "vision_config": {"depth": 2},
            "text_config": {
                "model_type": "qwen3_5_text", "attn_output_gate": true, "head_dim": 256,
                "hidden_size": 1024, "intermediate_size": 3584, "num_hidden_layers": 4,
                "num_attention_heads": 8, "num_key_value_heads": 2, "vocab_size": 248320,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
                "linear_num_value_heads": 16, "linear_value_head_dim": 128, "tie_word_embeddings": true,
                "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "default",
                                    "rope_theta": 10000000, "partial_rotary_factor": 0.25}
            }
        });
        let config = TextConfig::from_json(&json).unwrap();
        assert_eq!(config.family, Family::Qwen35);
        assert!(config.is_hybrid() && config.attn_output_gate && config.qk_norm);
        assert_eq!(config.layers[0].kind, LayerKind::LinearAttention);
        assert_eq!(config.layers[3].kind, LayerKind::Attention);
        assert_eq!(config.ropes[config.layers[3].rope].rotary_dim(), 64);
        assert_eq!(config.norm_style, NormStyle::OffsetF32);
    }

    #[test]
    fn parses_gemma4_sharing_and_per_layer_config() {
        let json = serde_json::json!({
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text", "hidden_size": 8, "intermediate_size": 64, "num_hidden_layers": 4,
                "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 32, "global_head_dim": 64,
                "vocab_size": 100, "hidden_size_per_layer_input": 2, "vocab_size_per_layer_input": 100,
                "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
                "num_kv_shared_layers": 2, "use_double_wide_mlp": true, "sliding_window": 512,
                "final_logit_softcapping": 30.0, "hidden_activation": "gelu_pytorch_tanh",
                "rope_parameters": {
                    "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
                    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
                }
            }
        });
        let config = TextConfig::from_json(&json).unwrap();
        assert_eq!(config.family, Family::Gemma4);
        let modes: Vec<KvSharing> = config.layers.iter().map(|l| l.kv_sharing).collect();
        assert_eq!(
            modes,
            vec![KvSharing::Store, KvSharing::Store, KvSharing::Reuse, KvSharing::Reuse]
        );
        assert_eq!(config.layers[1].head_dim, 64);
        assert_eq!(config.layers[0].head_dim, 32);
        assert_eq!(config.layers[2].intermediate_size, 128);
        assert_eq!(config.layers[0].sliding_window, Some(512));
        let full = &config.ropes[config.layers[1].rope];
        assert!(matches!(full.scaling, RopeScaling::Proportional { .. }));
        assert_eq!(full.rotary_dim(), 64);
        assert_eq!(config.per_layer_input, Some((2, 100)));
        assert_eq!(config.attn_scaling, Some(1.0));
        assert_eq!(config.final_softcap, Some(30.0));
    }

    #[test]
    fn parses_old_gemma3_ropes() {
        let json = serde_json::json!({
            "model_type": "gemma3_text", "hidden_size": 32, "intermediate_size": 128, "num_hidden_layers": 2,
            "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 32, "vocab_size": 100,
            "query_pre_attn_scalar": 168, "rope_local_base_freq": 10000.0,
            "rope_scaling": {"factor": 8.0, "rope_type": "linear"}, "rope_theta": 1000000.0,
            "sliding_window": 1024, "sliding_window_pattern": 2
        });
        let config = TextConfig::from_json(&json).unwrap();
        assert_eq!(config.layers[0].sliding_window, Some(1024));
        assert_eq!(config.layers[1].sliding_window, None);
        let local = &config.ropes[config.layers[0].rope];
        let global = &config.ropes[config.layers[1].rope];
        assert_eq!(local.theta, 10000.0);
        assert_eq!(global.theta, 1e6);
        assert!(matches!(global.scaling, RopeScaling::Linear { factor } if factor == 8.0));
        assert!(config.tie_word_embeddings && config.embed_scale && config.sandwich_norms);
    }
}
