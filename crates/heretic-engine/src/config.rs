// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Parsing of Transformers `config.json` files into an architecture description.

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
            other => bail!("Unsupported activation function: {other}"),
        })
    }
}

/// Description of a decoder-only text transformer.
///
/// A single generic implementation covers many architectures; the differences
/// between them are expressed through these flags.
#[derive(Debug, Clone)]
pub struct TextConfig {
    /// Transformers `model_type` of the text model (e.g. "qwen2").
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: RopeScaling,
    /// Fraction of the head dimension that receives rotary embeddings.
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
    /// Per-head RMSNorm on queries and keys (Qwen3).
    pub qk_norm: bool,
    /// Sliding window size per layer (`None` = full attention).
    pub sliding_window: Vec<Option<usize>>,
    /// Dtype the checkpoint was saved in ("bfloat16", ...).
    pub torch_dtype: Option<String>,
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

fn parse_rope(config: &Value, root: &Value) -> Result<(f64, RopeScaling, f64)> {
    // Transformers 5 moved RoPE settings into `rope_parameters`.
    let parameters = config
        .get("rope_parameters")
        .or_else(|| config.get("rope_scaling"))
        .or_else(|| root.get("rope_scaling"))
        .filter(|v| v.is_object());

    let theta = parameters
        .and_then(|p| get_f64(p, "rope_theta"))
        .or_else(|| get_f64(config, "rope_theta"))
        .unwrap_or(10000.0);

    let partial = parameters
        .and_then(|p| get_f64(p, "partial_rotary_factor"))
        .or_else(|| get_f64(config, "partial_rotary_factor"))
        .unwrap_or(1.0);

    let Some(parameters) = parameters else {
        return Ok((theta, RopeScaling::Default, partial));
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
        // Dynamic NTK scaling only changes frequencies beyond the original context length,
        // which Heretic's short prompts never reach.
        "dynamic" => RopeScaling::Default,
        other => bail!("Unsupported RoPE type: {other}"),
    };

    Ok((theta, scaling, partial))
}

/// The architectures supported by the generic transformer implementation.
pub const SUPPORTED_MODEL_TYPES: &[&str] = &["llama", "mistral", "qwen2", "qwen3"];

impl TextConfig {
    pub fn from_json(root: &Value) -> Result<TextConfig> {
        let config = text_config_json(root);

        let model_type = config
            .get("model_type")
            .and_then(Value::as_str)
            .or_else(|| root.get("model_type").and_then(Value::as_str))
            .unwrap_or_default()
            .to_string();

        if !SUPPORTED_MODEL_TYPES.contains(&model_type.as_str()) {
            let architectures = root
                .get("architectures")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(", "))
                .unwrap_or_default();
            bail!(
                "Unsupported model type \"{model_type}\" ({architectures}). Supported: {}",
                SUPPORTED_MODEL_TYPES.join(", ")
            );
        }

        let hidden_size = require_usize(config, "hidden_size")?;
        let num_attention_heads = require_usize(config, "num_attention_heads")?;
        let num_hidden_layers = require_usize(config, "num_hidden_layers")?;
        let num_key_value_heads = get_usize(config, "num_key_value_heads").unwrap_or(num_attention_heads);
        let head_dim = get_usize(config, "head_dim").unwrap_or(hidden_size / num_attention_heads);

        let (rope_theta, rope_scaling, partial_rotary_factor) = parse_rope(config, root)?;

        let attention_bias = get_bool(config, "attention_bias").unwrap_or(false);
        let (qkv_bias, o_bias, qk_norm) = match model_type.as_str() {
            // Qwen2 always has q/k/v biases and never an o_proj bias.
            "qwen2" => (true, false, false),
            "qwen3" => (attention_bias, attention_bias, true),
            _ => (attention_bias, attention_bias, false),
        };

        let sliding_window = Self::parse_sliding_window(config, &model_type, num_hidden_layers);

        let tie_word_embeddings = get_bool(config, "tie_word_embeddings")
            .or_else(|| get_bool(root, "tie_word_embeddings"))
            .unwrap_or(false);

        let torch_dtype = config
            .get("dtype")
            .or_else(|| config.get("torch_dtype"))
            .or_else(|| root.get("dtype"))
            .or_else(|| root.get("torch_dtype"))
            .and_then(Value::as_str)
            .map(String::from);

        Ok(TextConfig {
            vocab_size: require_usize(config, "vocab_size")?,
            hidden_size,
            intermediate_size: require_usize(config, "intermediate_size")?,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps: get_f64(config, "rms_norm_eps").unwrap_or(1e-6),
            rope_theta,
            rope_scaling,
            partial_rotary_factor,
            max_position_embeddings: get_usize(config, "max_position_embeddings").unwrap_or(4096),
            tie_word_embeddings,
            hidden_act: Activation::parse(config.get("hidden_act").and_then(Value::as_str).unwrap_or("silu"))?,
            qkv_bias,
            o_bias,
            mlp_bias: get_bool(config, "mlp_bias").unwrap_or(false),
            qk_norm,
            sliding_window,
            torch_dtype,
            model_type,
        })
    }

    fn parse_sliding_window(config: &Value, model_type: &str, layers: usize) -> Vec<Option<usize>> {
        let window = get_usize(config, "sliding_window");

        // Explicit per-layer types (Transformers 4.53+).
        if let Some(types) = config.get("layer_types").and_then(Value::as_array) {
            return types
                .iter()
                .map(|t| match t.as_str() {
                    Some("sliding_attention") => window,
                    _ => None,
                })
                .chain(std::iter::repeat(None))
                .take(layers)
                .collect();
        }

        match model_type {
            "mistral" => vec![window; layers],
            "qwen2" | "qwen3" => {
                if get_bool(config, "use_sliding_window").unwrap_or(false) {
                    let max_window_layers = get_usize(config, "max_window_layers").unwrap_or(layers);
                    (0..layers)
                        .map(|i| if i >= max_window_layers { window } else { None })
                        .collect()
                } else {
                    vec![None; layers]
                }
            }
            _ => vec![None; layers],
        }
    }

    /// Number of query heads per key/value head.
    pub fn n_rep(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Number of rotary dimensions per head.
    pub fn rotary_dim(&self) -> usize {
        ((self.head_dim as f64 * self.partial_rotary_factor) as usize) & !1
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
    fn rejects_unknown_architecture() {
        let json = serde_json::json!({"model_type": "mamba", "architectures": ["MambaForCausalLM"]});
        let error = TextConfig::from_json(&json).unwrap_err().to_string();
        assert!(error.contains("Unsupported model type \"mamba\""));
    }
}
