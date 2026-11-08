// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Conversion of Hugging Face checkpoints to unquantized GGUF files.
//!
//! A port of the relevant parts of llama.cpp's `convert_hf_to_gguf.py` (`conversion/base.py`,
//! `llama.py`, `qwen.py`, `gemma.py`) and `gguf-py` (`constants.py`, `tensor_mapping.py`):
//! the same tensor names, transformations, tensor types and metadata keys, so that the
//! result loads in llama.cpp and can be quantized with `llama-quantize`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, anyhow, bail};
use heretic_core::report::{ReporterExt, SharedReporter};
use safetensors::Dtype;
use serde_json::{Map, Value};

use super::vocab::{self, ModelFiles, SpecialVocab};
use super::writer::{GGML_QUANT_VERSION, GgmlType, GgufTensorInfo, GgufValue, Metadata, write_gguf};

/// A tensor of the source checkpoint.
#[derive(Debug, Clone)]
pub struct SourceTensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

/// The weights to convert: a checkpoint on disk, or a model with merged LoRA adapters.
pub trait TensorSource {
    fn tensors(&self) -> &[SourceTensor];
    /// Raw little-endian data of tensor `index`, in its `dtype`.
    fn read(&self, index: usize) -> Result<Vec<u8>>;
}

/// The safetensors files of a model directory, memory-mapped.
pub struct SafetensorsDirectory {
    maps: Vec<memmap2::Mmap>,
    tensors: Vec<SourceTensor>,
    locations: Vec<(usize, String)>,
}

impl SafetensorsDirectory {
    pub fn open(dir: &Path) -> Result<SafetensorsDirectory> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("Failed to read {}", dir.display()))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            bail!("{} contains no safetensors files", dir.display());
        }
        let mut maps = Vec::new();
        let mut tensors = Vec::new();
        let mut locations = Vec::new();
        for (index, path) in files.iter().enumerate() {
            let file = std::fs::File::open(path)?;
            // SAFETY: The files are not modified while mapped.
            let map = unsafe { memmap2::Mmap::map(&file)? };
            let (_, metadata) = safetensors::SafeTensors::read_metadata(&map)
                .map_err(|e| anyhow!("Invalid safetensors file {}: {e}", path.display()))?;
            let mut entries: Vec<_> = metadata.tensors().into_iter().collect();
            entries.sort_by_key(|(_, info)| info.data_offsets.0);
            for (name, info) in entries {
                tensors.push(SourceTensor {
                    name: name.clone(),
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                });
                locations.push((index, name));
            }
            maps.push(map);
        }
        Ok(SafetensorsDirectory {
            maps,
            tensors,
            locations,
        })
    }
}

impl TensorSource for SafetensorsDirectory {
    fn tensors(&self) -> &[SourceTensor] {
        &self.tensors
    }

    fn read(&self, index: usize) -> Result<Vec<u8>> {
        let (file, name) = &self.locations[index];
        Ok(safetensors::SafeTensors::deserialize(&self.maps[*file])
            .map_err(|e| anyhow!("{e}"))?
            .tensor(name)
            .map_err(|e| anyhow!("{e}"))?
            .data()
            .to_vec())
    }
}

/// The float type of the unquantized tensors (`--outtype` upstream).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutType {
    F32,
    F16,
    BF16,
}

impl OutType {
    fn ggml(self) -> GgmlType {
        match self {
            OutType::F32 => GgmlType::F32,
            OutType::F16 => GgmlType::F16,
            OutType::BF16 => GgmlType::BF16,
        }
    }

    /// `LlamaFileType`: ALL_F32, MOSTLY_F16, MOSTLY_BF16.
    fn file_type(self) -> u32 {
        match self {
            OutType::F32 => 0,
            OutType::F16 => 1,
            OutType::BF16 => 32,
        }
    }

    /// Like `--outtype auto`: BF16 for bfloat16 checkpoints, F16 otherwise.
    pub fn guess(tensors: &[SourceTensor]) -> OutType {
        match tensors.iter().find(|t| t.shape.len() >= 2).map(|t| t.dtype) {
            Some(Dtype::BF16) => OutType::BF16,
            _ => OutType::F16,
        }
    }
}

/// The architectures Heretic can export (`MODEL_ARCH` upstream).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Llama,
    Qwen2,
    Qwen3,
    Qwen35,
    Gemma2,
    Gemma3,
}

/// Hugging Face architectures and the GGUF architecture they convert to.
pub const SUPPORTED_ARCHITECTURES: &[(&str, Arch)] = &[
    ("LlamaForCausalLM", Arch::Llama),
    ("LLaMAForCausalLM", Arch::Llama),
    ("MistralForCausalLM", Arch::Llama),
    ("Qwen2ForCausalLM", Arch::Qwen2),
    ("Qwen2Model", Arch::Qwen2),
    ("Qwen3ForCausalLM", Arch::Qwen3),
    ("Qwen3Model", Arch::Qwen3),
    ("Qwen3_5ForConditionalGeneration", Arch::Qwen35),
    ("Qwen3_5ForCausalLM", Arch::Qwen35),
    ("Qwen3_5TextModel", Arch::Qwen35),
    ("Gemma2ForCausalLM", Arch::Gemma2),
    ("Gemma3ForCausalLM", Arch::Gemma3),
    ("Gemma3ForConditionalGeneration", Arch::Gemma3),
];

impl Arch {
    /// `MODEL_ARCH_NAMES`.
    pub fn gguf_name(self) -> &'static str {
        match self {
            Arch::Llama => "llama",
            Arch::Qwen2 => "qwen2",
            Arch::Qwen3 => "qwen3",
            Arch::Qwen35 => "qwen35",
            Arch::Gemma2 => "gemma2",
            Arch::Gemma3 => "gemma3",
        }
    }

    /// The GGUF architecture for a `config.json`.
    pub fn detect(config: &Value) -> Result<Arch> {
        let names: Vec<&str> = config
            .get("architectures")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let Some(name) = names.first() else {
            bail!("config.json has no \"architectures\"; cannot choose a GGUF architecture");
        };
        SUPPORTED_ARCHITECTURES
            .iter()
            .find(|(hf, _)| hf == name)
            .map(|(_, arch)| *arch)
            .ok_or_else(|| {
                let mut supported: Vec<&str> = SUPPORTED_ARCHITECTURES.iter().map(|(n, _)| *n).collect();
                supported.dedup();
                anyhow!(
                    "GGUF export does not support {name} yet. Supported: {}. Save the model as safetensors instead \
                     and convert it with llama.cpp's own tools.",
                    supported.join(", ")
                )
            })
    }

    /// Block tensors, from `tensor_mapping.py` restricted to `MODEL_TENSORS[arch]`:
    /// (Hugging Face module below `layers.{bid}.`, GGUF name below `blk.{bid}.`).
    fn block_tensors(self) -> Vec<(&'static str, &'static str)> {
        let mut map = vec![
            ("input_layernorm", "attn_norm"),
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.v_proj", "attn_v"),
            ("self_attn.o_proj", "attn_output"),
            ("mlp.gate_proj", "ffn_gate"),
            ("mlp.up_proj", "ffn_up"),
            ("mlp.down_proj", "ffn_down"),
        ];
        match self {
            Arch::Llama | Arch::Qwen2 => map.push(("post_attention_layernorm", "ffn_norm")),
            Arch::Qwen3 => map.extend([
                ("post_attention_layernorm", "ffn_norm"),
                ("self_attn.q_norm", "attn_q_norm"),
                ("self_attn.k_norm", "attn_k_norm"),
            ]),
            Arch::Qwen35 => map.extend([
                // QWEN35 has no FFN_NORM, so post_attention_layernorm maps to ATTN_POST_NORM.
                ("post_attention_layernorm", "post_attention_norm"),
                ("self_attn.q_norm", "attn_q_norm"),
                ("self_attn.k_norm", "attn_k_norm"),
                ("linear_attn.in_proj_qkv", "attn_qkv"),
                ("linear_attn.in_proj_z", "attn_gate"),
                ("linear_attn.in_proj_a", "ssm_alpha"),
                ("linear_attn.in_proj_b", "ssm_beta"),
                ("linear_attn.conv1d", "ssm_conv1d"),
                ("linear_attn.dt_proj", "ssm_dt"),
                ("linear_attn.A_log", "ssm_a"),
                ("linear_attn.norm", "ssm_norm"),
                ("linear_attn.out_proj", "ssm_out"),
            ]),
            Arch::Gemma2 => map.extend([
                ("post_attention_layernorm", "post_attention_norm"),
                ("pre_feedforward_layernorm", "ffn_norm"),
                ("post_feedforward_layernorm", "post_ffw_norm"),
            ]),
            Arch::Gemma3 => map.extend([
                ("post_attention_layernorm", "post_attention_norm"),
                ("pre_feedforward_layernorm", "ffn_norm"),
                ("post_feedforward_layernorm", "post_ffw_norm"),
                ("self_attn.q_norm", "attn_q_norm"),
                ("self_attn.k_norm", "attn_k_norm"),
            ]),
        }
        map
    }
}

/// `{**hparams, **hparams["text_config"]}`, like `TextModel.__init__`.
pub fn text_hparams(config: &Value) -> Value {
    let mut merged: Map<String, Value> = config.as_object().cloned().unwrap_or_default();
    if let Some(Value::Object(text)) = config.get("text_config") {
        for (key, value) in text {
            merged.insert(key.clone(), value.clone());
        }
    }
    Value::Object(merged)
}

fn find<'a>(hparams: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|k| hparams.get(*k).filter(|v| !v.is_null()))
}

fn find_u64(hparams: &Value, keys: &[&str]) -> Option<u64> {
    find(hparams, keys).and_then(Value::as_u64)
}

fn find_f64(hparams: &Value, keys: &[&str]) -> Option<f64> {
    find(hparams, keys).and_then(Value::as_f64)
}

fn require_u64(hparams: &Value, key: &str) -> Result<u64> {
    hparams
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("config.json is missing \"{key}\""))
}

/// `self.rope_parameters` after `TextModel.__init__` mirrored the global settings into it.
fn rope_parameters(hparams: &Value) -> Map<String, Value> {
    let mut rope: Map<String, Value> = find(hparams, &["rope_parameters", "rope_scaling"])
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let rope_theta = find(
        hparams,
        &[
            "global_rope_theta",
            "rope_global_theta",
            "rope_theta_global",
            "rope_theta",
            "rotary_emb_base",
        ],
    )
    .cloned();
    let local_theta = find(
        hparams,
        &[
            "local_rope_theta",
            "rope_local_theta",
            "rope_theta_local",
            "swa_rope_theta",
            "rope_local_base_freq",
        ],
    )
    .cloned();
    let partial = find(hparams, &["partial_rotary_factor", "rope_pct", "rope_percent"]).cloned();
    let original = find(hparams, &["original_max_position_embeddings"]).cloned();

    if !rope.contains_key("full_attention") && !rope.contains_key("sliding_attention") {
        if let Some(local) = local_theta {
            rope.insert("sliding_attention".into(), serde_json::json!({"rope_theta": local}));
        }
        if !rope.contains_key("rope_theta")
            && let Some(theta) = rope_theta
        {
            rope.insert("rope_theta".into(), theta);
        }
        if !rope.contains_key("rope_type")
            && let Some(kind) = rope.get("type").cloned()
        {
            rope.insert("rope_type".into(), kind);
        }
        if !rope.contains_key("partial_rotary_factor")
            && let Some(partial) = partial
        {
            rope.insert("partial_rotary_factor".into(), partial);
        }
        if !rope.contains_key("original_max_position_embeddings")
            && let Some(original) = original
        {
            rope.insert("original_max_position_embeddings".into(), original);
        }
    }
    rope
}

struct Params<'a> {
    arch: Arch,
    hparams: &'a Value,
    rope: Map<String, Value>,
    out_type: OutType,
    block_count: u64,
}

impl Params<'_> {
    fn key(&self, suffix: &str) -> String {
        format!("{}.{suffix}", self.arch.gguf_name())
    }

    fn u32(&self, meta: &mut Metadata, suffix: &str, value: u64) {
        meta.set(self.key(suffix), GgufValue::U32(value as u32));
    }

    fn f32(&self, meta: &mut Metadata, suffix: &str, value: f64) {
        meta.set(self.key(suffix), GgufValue::F32(value as f32));
    }

    fn rope_full(&self) -> Map<String, Value> {
        self.rope
            .get("full_attention")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_else(|| self.rope.clone())
    }

    /// `TextModel.set_gguf_parameters`.
    fn text_model(&self, meta: &mut Metadata) -> Result<()> {
        let hp = self.hparams;
        self.u32(meta, "block_count", self.block_count);
        let context_keys = [
            "max_position_embeddings",
            "n_ctx",
            "n_positions",
            "max_length",
            "max_sequence_length",
            "model_max_length",
        ];
        if let Some(n) = find_u64(hp, &context_keys) {
            self.u32(meta, "context_length", n);
        }
        if let Some(n) = find_u64(hp, &["hidden_size", "n_embd", "dim"]) {
            self.u32(meta, "embedding_length", n);
        }
        let ff_keys = [
            "prefix_dense_intermediate_size",
            "dense_intermediate_size",
            "intermediate_size",
            "n_inner",
            "hidden_dim",
        ];
        if let Some(n) = find_u64(hp, &ff_keys) {
            self.u32(meta, "feed_forward_length", n);
        }
        if let Some(n) = find_u64(hp, &["num_attention_heads", "n_head", "n_heads"]) {
            self.u32(meta, "attention.head_count", n);
        }
        if let Some(n) = find_u64(hp, &["num_key_value_heads", "n_kv_heads"]) {
            self.u32(meta, "attention.head_count_kv", n);
        }
        if hp.get("is_causal") == Some(&Value::Bool(false))
            || hp.get("use_bidirectional_attention") == Some(&Value::Bool(true))
        {
            meta.set(self.key("attention.causal"), GgufValue::Bool(false));
        }

        let rope_params = self.rope_full();
        if let Some(rope_type) = rope_params.get("rope_type").and_then(Value::as_str) {
            let factor = rope_params.get("factor").and_then(Value::as_f64);
            match (rope_type, factor) {
                ("linear", Some(factor)) => {
                    meta.set(self.key("rope.scaling.type"), GgufValue::String("linear".into()));
                    self.f32(meta, "rope.scaling.factor", factor);
                }
                ("yarn", Some(factor)) => {
                    meta.set(self.key("rope.scaling.type"), GgufValue::String("yarn".into()));
                    self.f32(meta, "rope.scaling.factor", factor);
                    let original = rope_params
                        .get("original_max_position_embeddings")
                        .and_then(Value::as_u64)
                        .ok_or_else(|| anyhow!("YaRN RoPE scaling without original_max_position_embeddings"))?;
                    self.u32(meta, "rope.scaling.original_context_length", original);
                    if let Some(truncate) = rope_params.get("truncate").and_then(Value::as_bool) {
                        meta.set(self.key("rope.scaling.yarn_truncate"), GgufValue::Bool(truncate));
                    }
                    if let Some(v) = rope_params.get("extrapolation_factor").and_then(Value::as_f64) {
                        self.f32(meta, "rope.scaling.yarn_ext_factor", v);
                    }
                    if let Some(v) = rope_params
                        .get("attention_factor")
                        .or_else(|| rope_params.get("attn_factor"))
                        .and_then(Value::as_f64)
                    {
                        self.f32(meta, "rope.scaling.yarn_attn_factor", v);
                    }
                    if let Some(v) = rope_params.get("beta_fast").and_then(Value::as_f64) {
                        self.f32(meta, "rope.scaling.yarn_beta_fast", v);
                    }
                    if let Some(v) = rope_params.get("beta_slow").and_then(Value::as_f64) {
                        self.f32(meta, "rope.scaling.yarn_beta_slow", v);
                    }
                }
                ("su" | "longrope", _) => {
                    meta.set(self.key("rope.scaling.type"), GgufValue::String("longrope".into()));
                }
                // "dynamic" and "llama3" are handled elsewhere (llama3: rope_freqs tensor).
                _ => {}
            }
        }

        if let Some(sections) = self.rope.get("mrope_section").and_then(Value::as_array) {
            let mut sections: Vec<i32> = sections.iter().filter_map(Value::as_i64).map(|v| v as i32).collect();
            while sections.len() < 4 {
                sections.push(0);
            }
            sections.truncate(4);
            meta.set(self.key("rope.dimension_sections"), GgufValue::i32s(sections));
        }
        if let Some(theta) = rope_params.get("rope_theta").and_then(Value::as_f64) {
            self.f32(meta, "rope.freq_base", theta);
        }
        if let Some(theta) = self
            .rope
            .get("sliding_attention")
            .and_then(|s| s.get("rope_theta"))
            .and_then(Value::as_f64)
        {
            self.f32(meta, "rope.freq_base_swa", theta);
        }
        if let Some(eps) = find_f64(hp, &["rms_norm_eps", "norm_eps"]) {
            self.f32(meta, "attention.layer_norm_rms_epsilon", eps);
        }
        if let Some(eps) = find_f64(hp, &["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"]) {
            self.f32(meta, "attention.layer_norm_epsilon", eps);
        }
        if let Some(head_dim) = hp.get("head_dim").and_then(Value::as_u64) {
            self.u32(meta, "attention.key_length", head_dim);
            self.u32(meta, "attention.value_length", head_dim);
        }
        meta.set("general.file_type", GgufValue::U32(self.out_type.file_type()));
        Ok(())
    }

    fn head_dim(&self) -> Result<u64> {
        match self.hparams.get("head_dim").and_then(Value::as_u64) {
            Some(d) => Ok(d),
            None => Ok(require_u64(self.hparams, "hidden_size")? / require_u64(self.hparams, "num_attention_heads")?),
        }
    }

    fn set_parameters(&self, meta: &mut Metadata) -> Result<()> {
        let hp = self.hparams;
        match self.arch {
            Arch::Llama => {
                // LlamaModel.set_gguf_parameters
                self.text_model(meta)?;
                self.u32(meta, "vocab_size", require_u64(hp, "vocab_size")?);
                self.u32(meta, "rope.dimension_count", self.head_dim()?);
            }
            Arch::Qwen2 | Arch::Qwen3 => self.text_model(meta)?,
            Arch::Qwen35 => {
                // Qwen3_5TextModel: TextModel, Qwen3NextModel and _Qwen35MRopeMixin parameters
                // (without the MTP layers, like --no-mtp).
                self.text_model(meta)?;
                self.u32(meta, "ssm.conv_kernel", require_u64(hp, "linear_conv_kernel_dim")?);
                self.u32(meta, "ssm.state_size", require_u64(hp, "linear_key_head_dim")?);
                self.u32(meta, "ssm.group_count", require_u64(hp, "linear_num_key_heads")?);
                self.u32(meta, "ssm.time_step_rank", require_u64(hp, "linear_num_value_heads")?);
                self.u32(
                    meta,
                    "ssm.inner_size",
                    require_u64(hp, "linear_value_head_dim")? * require_u64(hp, "linear_num_value_heads")?,
                );
                if let Some(types) = hp.get("layer_types").and_then(Value::as_array) {
                    if types.len() as u64 != self.block_count {
                        bail!(
                            "layer_types has {} entries, expected num_hidden_layers ({})",
                            types.len(),
                            self.block_count
                        );
                    }
                    let recurrent = types.iter().map(|t| t.as_str() == Some("linear_attention"));
                    meta.set(self.key("attention.recurrent_layers"), GgufValue::bools(recurrent));
                }
                self.u32(
                    meta,
                    "full_attention_interval",
                    hp.get("full_attention_interval").and_then(Value::as_u64).unwrap_or(4),
                );
                let partial = self
                    .rope
                    .get("partial_rotary_factor")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.25);
                self.u32(meta, "rope.dimension_count", (self.head_dim()? as f64 * partial) as u64);
                if !self.rope.contains_key("mrope_section") {
                    meta.set(self.key("rope.dimension_sections"), GgufValue::i32s([11, 11, 10, 0]));
                }
            }
            Arch::Gemma2 => {
                // Gemma2Model.set_gguf_parameters (does not call TextModel's).
                self.u32(meta, "context_length", require_u64(hp, "max_position_embeddings")?);
                self.u32(meta, "embedding_length", require_u64(hp, "hidden_size")?);
                self.u32(meta, "block_count", self.block_count);
                self.u32(meta, "feed_forward_length", require_u64(hp, "intermediate_size")?);
                let heads = require_u64(hp, "num_attention_heads")?;
                self.u32(meta, "attention.head_count", heads);
                self.u32(
                    meta,
                    "attention.head_count_kv",
                    hp.get("num_key_value_heads").and_then(Value::as_u64).unwrap_or(heads),
                );
                let eps = hp
                    .get("rms_norm_eps")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| anyhow!("config.json is missing \"rms_norm_eps\""))?;
                self.f32(meta, "attention.layer_norm_rms_epsilon", eps);
                let head_dim = require_u64(hp, "head_dim")?;
                self.u32(meta, "attention.key_length", head_dim);
                self.u32(meta, "attention.value_length", head_dim);
                meta.set("general.file_type", GgufValue::U32(self.out_type.file_type()));
                for key in ["attn_logit_softcapping", "final_logit_softcapping"] {
                    let value = hp
                        .get(key)
                        .and_then(Value::as_f64)
                        .ok_or_else(|| anyhow!("config.json is missing \"{key}\""))?;
                    self.f32(meta, key, value);
                }
                self.u32(meta, "attention.sliding_window", require_u64(hp, "sliding_window")?);
            }
            Arch::Gemma3 => {
                // Gemma3Model.set_gguf_parameters, with its defaults for missing values.
                self.text_model(meta)?;
                let get = |key: &str| hp.get(key).filter(|v| !v.is_null());
                self.u32(
                    meta,
                    "context_length",
                    get("max_position_embeddings").and_then(Value::as_u64).unwrap_or(131072),
                );
                self.u32(
                    meta,
                    "attention.head_count",
                    get("num_attention_heads").and_then(Value::as_u64).unwrap_or(8),
                );
                self.f32(
                    meta,
                    "attention.layer_norm_rms_epsilon",
                    get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6),
                );
                let head_dim = get("head_dim").and_then(Value::as_u64).unwrap_or(256);
                self.u32(meta, "attention.key_length", head_dim);
                self.u32(meta, "attention.value_length", head_dim);
                let theta = self
                    .rope_full()
                    .get("rope_theta")
                    .and_then(Value::as_f64)
                    .unwrap_or(1_000_000.0);
                self.f32(meta, "rope.freq_base", theta);
                if get("attn_logit_softcapping").is_some() {
                    bail!("Gemma 3 checkpoints are not expected to have attn_logit_softcapping");
                }
                if let Some(cap) = get("final_logit_softcapping").and_then(Value::as_f64)
                    && cap != 0.0
                {
                    self.f32(meta, "final_logit_softcapping", cap);
                }
                if get("sliding_window_pattern").and_then(Value::as_u64) != Some(1) {
                    self.u32(meta, "attention.sliding_window", require_u64(hp, "sliding_window")?);
                }
                self.u32(
                    meta,
                    "attention.head_count_kv",
                    get("num_key_value_heads").and_then(Value::as_u64).unwrap_or(4),
                );
            }
        }
        Ok(())
    }
}

/// An elementwise or layout transformation applied to a tensor before writing it.
#[derive(Debug, Clone, PartialEq)]
enum Transform {
    /// `LlamaModel.permute`: undo the Q/K RoPE row permutation of Hugging Face checkpoints.
    Permute {
        heads: usize,
    },
    AddOne,
    NegExp,
    /// Drop the middle dimension of a `(channels, 1, kernel)` convolution.
    Squeeze,
    /// `_reorder_v_heads` on rows `offset..` (grouped by K head to tiled order).
    ReorderRows {
        offset: usize,
        num_k_heads: usize,
        num_v_per_k: usize,
        head_dim: usize,
    },
    /// `_reorder_v_heads` on the columns of a 2D tensor.
    ReorderColumns {
        num_k_heads: usize,
        num_v_per_k: usize,
        head_dim: usize,
    },
    /// Keep only the first rows (Gemma 3 removes out-of-vocabulary embedding rows).
    TruncateRows(usize),
}

enum Data {
    Source(usize),
    RopeFreqs(Vec<f32>),
}

struct Planned {
    info: GgufTensorInfo,
    data: Data,
    transforms: Vec<Transform>,
    source_shape: Vec<usize>,
}

/// Normalizes a checkpoint tensor name to the text model's relative name
/// (`layers.N....`, `embed_tokens.weight`, `norm.weight`, `lm_head.weight`), or `None`
/// for tensors that are not part of the text model (vision towers, projectors, MTP heads).
fn relative_name(name: &str) -> Option<String> {
    const SKIPPED: &[&str] = &[
        "visual.",
        "vision_",
        "vision.",
        "audio",
        "multi_modal_projector",
        "mm_projector",
        "projector.",
        "mtp.",
        "talker.",
        "image_newline",
        "patch_embed",
    ];
    let mut rest = name;
    for prefix in [
        "model.language_model.",
        "language_model.model.",
        "language_model.",
        "model.",
    ] {
        if let Some(stripped) = rest.strip_prefix(prefix) {
            rest = stripped;
            break;
        }
    }
    if let Some(stripped) = rest.strip_prefix("text_model.") {
        rest = stripped;
    }
    if SKIPPED
        .iter()
        .any(|s| rest.starts_with(s) || name.contains(&format!(".{s}")))
    {
        return None;
    }
    if rest.ends_with(".rotary_emb.inv_freq") || rest.ends_with(".attention.masked_bias") {
        return None;
    }
    Some(rest.to_string())
}

fn block_index(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("layers.")?;
    let (index, module) = rest.split_once('.')?;
    Some((index.parse().ok()?, module))
}

/// Maps a relative tensor name to its GGUF name (`TensorNameMap.get_name` with the
/// ".weight"/".bias" suffixes).
pub fn gguf_tensor_name(arch: Arch, relative: &str) -> Option<String> {
    let global = match relative {
        "embed_tokens.weight" => Some("token_embd.weight"),
        "norm.weight" => Some("output_norm.weight"),
        "lm_head.weight" if arch != Arch::Gemma2 => Some("output.weight"),
        _ => None,
    };
    if let Some(name) = global {
        return Some(name.to_string());
    }
    let (bid, module) = block_index(relative)?;
    let (module, suffix) = if let Some(m) = module.strip_suffix(".weight") {
        (m, ".weight")
    } else if let Some(m) = module.strip_suffix(".bias") {
        (m, ".bias")
    } else {
        (module, "")
    };
    arch.block_tensors()
        .iter()
        .find(|(hf, _)| *hf == module)
        .map(|(_, gguf)| format!("blk.{bid}.{gguf}{suffix}"))
}

/// `ModelBase._tensor_requires_f32` (and the 1D/norm rule of `prepare_tensors`).
fn requires_f32(gguf_name: &str, n_dims: usize) -> bool {
    n_dims <= 1
        || gguf_name.ends_with("_norm.weight")
        || gguf_name.contains(".ssm_conv1d.")
        || !gguf_name.ends_with(".weight")
}

fn reorder_index(index: usize, num_k_heads: usize, num_v_per_k: usize, head_dim: usize) -> usize {
    // Output position (j, i, d) in tiled order takes the grouped input (i, j, d).
    let d = index % head_dim;
    let i = (index / head_dim) % num_k_heads;
    let j = index / head_dim / num_k_heads;
    (i * num_v_per_k + j) * head_dim + d
}

fn apply(transform: &Transform, data: Vec<f32>, shape: &mut Vec<usize>) -> Result<Vec<f32>> {
    let cols: usize = shape.iter().skip(1).product();
    Ok(match transform {
        Transform::AddOne => data.into_iter().map(|v| v + 1.0).collect(),
        Transform::NegExp => data.into_iter().map(|v| -v.exp()).collect(),
        Transform::Squeeze => {
            shape.retain(|&d| d != 1);
            data
        }
        Transform::TruncateRows(rows) => {
            let rows = (*rows).min(shape[0]);
            shape[0] = rows;
            data[..rows * cols].to_vec()
        }
        Transform::Permute { heads } => {
            let rows = shape[0];
            if *heads == 0 {
                bail!("config.json has no attention head count");
            }
            let half = rows / heads / 2;
            if half == 0 || rows != heads * 2 * half {
                bail!("Cannot permute {rows} rows for {heads} heads");
            }
            let mut out = vec![0f32; data.len()];
            for h in 0..*heads {
                for j in 0..half {
                    for t in 0..2 {
                        let src = h * 2 * half + t * half + j;
                        let dst = h * 2 * half + j * 2 + t;
                        out[dst * cols..(dst + 1) * cols].copy_from_slice(&data[src * cols..(src + 1) * cols]);
                    }
                }
            }
            out
        }
        Transform::ReorderRows {
            offset,
            num_k_heads,
            num_v_per_k,
            head_dim,
        } => {
            let rows = shape[0] - offset;
            if rows != num_k_heads * num_v_per_k * head_dim {
                bail!("Cannot reorder {rows} rows into {num_k_heads}x{num_v_per_k}x{head_dim}");
            }
            let mut out = data.clone();
            for row in 0..rows {
                let src = offset + reorder_index(row, *num_k_heads, *num_v_per_k, *head_dim);
                let dst = offset + row;
                out[dst * cols..(dst + 1) * cols].copy_from_slice(&data[src * cols..(src + 1) * cols]);
            }
            out
        }
        Transform::ReorderColumns {
            num_k_heads,
            num_v_per_k,
            head_dim,
        } => {
            if shape.len() != 2 || cols != num_k_heads * num_v_per_k * head_dim {
                bail!("Cannot reorder the columns of a {shape:?} tensor");
            }
            let mut out = vec![0f32; data.len()];
            for row in 0..shape[0] {
                for col in 0..cols {
                    out[row * cols + col] =
                        data[row * cols + reorder_index(col, *num_k_heads, *num_v_per_k, *head_dim)];
                }
            }
            out
        }
    })
}

fn decode_f32(raw: &[u8], dtype: Dtype) -> Result<Vec<f32>> {
    Ok(match dtype {
        Dtype::F32 => raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
        Dtype::F16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::f16::from_le_bytes(*c).to_f32())
            .collect(),
        Dtype::BF16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::bf16::from_le_bytes(*c).to_f32())
            .collect(),
        Dtype::F64 => raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c) as f32)
            .collect(),
        other => bail!("Unsupported tensor dtype {other:?} for GGUF export"),
    })
}

fn encode(data: &[f32], ggml: GgmlType) -> Result<Vec<u8>> {
    Ok(match ggml {
        GgmlType::F32 => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        // Round to nearest even, like numpy's astype(float16) and gguf.quants for BF16.
        GgmlType::F16 => data
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect(),
        GgmlType::BF16 => data
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect(),
        GgmlType::Other(id) => bail!("Cannot encode tensors as GGML type {id}"),
    })
}

fn same_type(dtype: Dtype, ggml: GgmlType) -> bool {
    matches!(
        (dtype, ggml),
        (Dtype::F32, GgmlType::F32) | (Dtype::F16, GgmlType::F16) | (Dtype::BF16, GgmlType::BF16)
    )
}

/// What a conversion produced.
#[derive(Debug, Clone)]
pub struct ConvertedGguf {
    pub path: PathBuf,
    pub architecture: &'static str,
    pub tensor_count: usize,
    pub out_type: OutType,
}

/// Number of real vocabulary entries for Gemma 3 (rows beyond it are dropped from the embedding).
fn gemma3_vocab_rows(files: &dyn ModelFiles, hparams: &Value) -> Result<Option<usize>> {
    if let Some(vocab) = vocab::sentencepiece_vocab(files, hparams)? {
        return Ok(Some(vocab.tokens.len()));
    }
    let Some(path) = files.path("tokenizer.json") else {
        return Ok(None);
    };
    let tokenizer: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let base = tokenizer
        .pointer("/model/vocab")
        .and_then(Value::as_object)
        .map_or(0, Map::len);
    let added = tokenizer
        .get("added_tokens")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    Ok(Some(base + added))
}

/// Writes the tokenizer metadata (`set_vocab` of the architecture's model class).
fn set_vocab(arch: Arch, meta: &mut Metadata, files: &dyn ModelFiles, hparams: &Value) -> Result<()> {
    match arch {
        Arch::Llama => {
            // LlamaModel.set_vocab: SentencePiece, then LlamaHfVocab, then GPT-2 BPE (Llama 3).
            if let Some(config) = files.path("tokenizer_config.json") {
                let config: Value = serde_json::from_str(&std::fs::read_to_string(config)?)?;
                if let Some(add) = config.get("add_prefix_space").and_then(Value::as_bool) {
                    meta.set("tokenizer.ggml.add_space_prefix", GgufValue::Bool(add));
                }
            }
            if let Some(vocab) = vocab::sentencepiece_vocab(files, hparams)? {
                vocab::add_vocab(meta, &vocab);
                SpecialVocab::load(files, false, Some(vocab.tokens.len()))?.add_to(meta);
            } else if let Some(vocab) = vocab::llama_hf_vocab(files, hparams)? {
                vocab::add_vocab(meta, &vocab);
                SpecialVocab::load(files, false, Some(vocab.tokens.len()))?.add_to(meta);
            } else {
                let vocab = vocab::gpt2_vocab(files, hparams)?;
                vocab::add_vocab(meta, &vocab);
                SpecialVocab::load(files, true, None)?.add_to(meta);
            }
            let vocab_size = hparams.get("vocab_size").and_then(Value::as_u64);
            if vocab_size == Some(32016) {
                // CodeLlama infill tokens.
                let mut special = SpecialVocab::default();
                for (typ, id) in [("prefix", 32007), ("suffix", 32008), ("middle", 32009), ("eot", 32010)] {
                    special.set_special_token_id(typ, id);
                }
                special.add_to(meta);
            }
            if vocab_size == Some(49152) {
                meta.set("tokenizer.ggml.add_bos_token", GgufValue::Bool(false));
            }
        }
        Arch::Qwen2 | Arch::Qwen3 | Arch::Qwen35 => {
            // Qwen2Model.set_vocab: SentencePiece if present, otherwise GPT-2 BPE.
            if let Some(vocab) = vocab::sentencepiece_vocab(files, hparams)? {
                vocab::add_vocab(meta, &vocab);
                SpecialVocab::load(files, false, Some(vocab.tokens.len()))?.add_to(meta);
            } else {
                let vocab = vocab::gpt2_vocab(files, hparams)?;
                vocab::add_vocab(meta, &vocab);
                SpecialVocab::load(files, true, None)?.add_to(meta);
            }
        }
        Arch::Gemma2 => {
            let vocab = vocab::sentencepiece_vocab(files, hparams)?
                .ok_or_else(|| anyhow!("Gemma 2 GGUF export needs tokenizer.model"))?;
            vocab::add_vocab(meta, &vocab);
            SpecialVocab::load(files, false, Some(vocab.tokens.len()))?.add_to(meta);
            meta.set("tokenizer.ggml.add_space_prefix", GgufValue::Bool(false));
        }
        Arch::Gemma3 => {
            if let Some(vocab) = vocab::sentencepiece_vocab(files, hparams)? {
                vocab::add_vocab(meta, &vocab);
                SpecialVocab::load(files, false, Some(vocab.tokens.len()))?.add_to(meta);
                meta.set("tokenizer.ggml.add_space_prefix", GgufValue::Bool(false));
            } else {
                let vocab = vocab::gpt2_vocab(files, hparams)?;
                vocab::add_vocab(meta, &vocab);
                SpecialVocab::load(files, true, None)?.add_to(meta);
            }
        }
    }
    Ok(())
}

/// `LlamaModel.generate_extra_tensors`: per-frequency factors for Llama 3 RoPE scaling.
fn llama3_rope_factors(rope: &Map<String, Value>, head_dim: u64) -> Option<Vec<f32>> {
    if !rope
        .get("rope_type")
        .and_then(Value::as_str)?
        .eq_ignore_ascii_case("llama3")
    {
        return None;
    }
    let base = rope.get("rope_theta").and_then(Value::as_f64).unwrap_or(10000.0) as f32;
    let factor = rope.get("factor").and_then(Value::as_f64).unwrap_or(8.0) as f32;
    let low = rope.get("low_freq_factor").and_then(Value::as_f64).unwrap_or(1.0) as f32;
    let high = rope.get("high_freq_factor").and_then(Value::as_f64).unwrap_or(4.0) as f32;
    let old_context = rope
        .get("original_max_position_embeddings")
        .and_then(Value::as_f64)
        .unwrap_or(8192.0) as f32;
    let low_wavelen = old_context / low;
    let high_wavelen = old_context / high;
    let dim = head_dim as usize;
    Some(
        (0..dim)
            .step_by(2)
            .map(|i| {
                let freq = 1.0 / base.powf(i as f32 / dim as f32);
                let wavelen = 2.0 * std::f32::consts::PI / freq;
                if wavelen < high_wavelen {
                    1.0
                } else if wavelen > low_wavelen {
                    factor
                } else {
                    let smooth = (old_context / wavelen - low) / (high - low);
                    1.0 / ((1.0 - smooth) / factor + smooth)
                }
            })
            .collect(),
    )
}

/// Converts a checkpoint to an unquantized GGUF file.
pub fn convert_to_gguf(
    config: &Value,
    files: &dyn ModelFiles,
    source: &dyn TensorSource,
    name: &str,
    out_type: OutType,
    output: &Path,
    reporter: &SharedReporter,
) -> Result<ConvertedGguf> {
    let arch = Arch::detect(config)?;
    let hparams = text_hparams(config);
    for key in ["num_local_experts", "num_experts", "n_routed_experts"] {
        if hparams.get(key).and_then(Value::as_u64).is_some_and(|n| n > 0) {
            bail!("GGUF export of mixture-of-experts models is not supported yet");
        }
    }
    let block_count = find_u64(&hparams, &["n_layers", "num_hidden_layers", "n_layer", "num_layers"])
        .ok_or_else(|| anyhow!("config.json is missing \"num_hidden_layers\""))?;
    let rope = rope_parameters(&hparams);
    let params = Params {
        arch,
        hparams: &hparams,
        rope,
        out_type,
        block_count,
    };

    reporter.info(format!(
        "* GGUF architecture: **{}**, tensor type **{:?}**",
        arch.gguf_name(),
        out_type
    ));

    // Metadata, in the order the upstream writer adds it.
    let mut meta = Metadata::default();
    meta.set("general.architecture", GgufValue::String(arch.gguf_name().into()));
    meta.set("general.type", GgufValue::String("model".into()));
    meta.set("general.name", GgufValue::String(name.into()));
    params.set_parameters(&mut meta)?;
    meta.set("general.quantization_version", GgufValue::U32(GGML_QUANT_VERSION));
    meta.set("general.alignment", GgufValue::U32(32));
    set_vocab(arch, &mut meta, files, &hparams)?;

    // Tensor plan.
    let n_head = find_u64(&hparams, &["n_heads", "num_attention_heads"]).unwrap_or(0) as usize;
    let n_kv_head = find_u64(&hparams, &["n_kv_heads", "num_key_value_heads"])
        .map(|v| v as usize)
        .unwrap_or(n_head);
    let gemma3_rows = if arch == Arch::Gemma3 {
        gemma3_vocab_rows(files, &hparams)?
    } else {
        None
    };
    let linear = if arch == Arch::Qwen35 {
        let get = |k: &str| require_u64(&hparams, k).map(|v| v as usize);
        Some((
            get("linear_num_key_heads")?,
            get("linear_num_value_heads")?,
            get("linear_key_head_dim")?,
            get("linear_value_head_dim")?,
        ))
    } else {
        None
    };

    let mut planned: Vec<Planned> = Vec::new();
    if arch == Arch::Llama
        && let Some(factors) = llama3_rope_factors(&params.rope_full(), params.head_dim()?)
    {
        planned.push(Planned {
            info: GgufTensorInfo {
                name: "rope_freqs.weight".into(),
                ggml_type: GgmlType::F32,
                shape: vec![factors.len() as u64],
            },
            source_shape: vec![factors.len()],
            data: Data::RopeFreqs(factors),
            transforms: Vec::new(),
        });
    }

    let mut skipped = 0usize;
    let mut seen: HashMap<String, String> = HashMap::new();
    for (index, tensor) in source.tensors().iter().enumerate() {
        let Some(mut relative) = relative_name(&tensor.name) else {
            skipped += 1;
            continue;
        };
        if matches!(arch, Arch::Gemma2) && relative == "lm_head.weight" {
            // Gemma's filter_tensors: llama.cpp ties the output to the embeddings.
            skipped += 1;
            continue;
        }
        let mut transforms = Vec::new();

        match arch {
            Arch::Llama => {
                if relative.ends_with("q_proj.weight") || relative.ends_with("q_proj.bias") {
                    transforms.push(Transform::Permute { heads: n_head });
                }
                if relative.ends_with("k_proj.weight") || relative.ends_with("k_proj.bias") {
                    transforms.push(Transform::Permute { heads: n_kv_head });
                }
            }
            Arch::Qwen35 => {
                let (num_k, num_v, k_dim, v_dim) = linear.unwrap();
                // _LinearAttentionVReorderBase.modify_tensors
                if num_k != num_v && relative.contains("linear_attn.") {
                    let per_k = num_v / num_k;
                    if relative.contains(".in_proj_qkv.") {
                        transforms.push(Transform::ReorderRows {
                            offset: 2 * k_dim * num_k,
                            num_k_heads: num_k,
                            num_v_per_k: per_k,
                            head_dim: v_dim,
                        });
                    } else if relative.contains(".in_proj_z.") {
                        transforms.push(Transform::ReorderRows {
                            offset: 0,
                            num_k_heads: num_k,
                            num_v_per_k: per_k,
                            head_dim: v_dim,
                        });
                    } else if relative.contains(".in_proj_b.")
                        || relative.contains(".in_proj_a.")
                        || relative.contains(".A_log")
                        || relative.contains(".dt_bias")
                        || relative.contains(".dt_proj")
                    {
                        transforms.push(Transform::ReorderRows {
                            offset: 0,
                            num_k_heads: num_k,
                            num_v_per_k: per_k,
                            head_dim: 1,
                        });
                    } else if relative.contains(".conv1d") {
                        transforms.push(Transform::Squeeze);
                        transforms.push(Transform::ReorderRows {
                            offset: 2 * k_dim * num_k,
                            num_k_heads: num_k,
                            num_v_per_k: per_k,
                            head_dim: v_dim,
                        });
                    } else if relative.contains(".out_proj.") {
                        transforms.push(Transform::ReorderColumns {
                            num_k_heads: num_k,
                            num_v_per_k: per_k,
                            head_dim: v_dim,
                        });
                    }
                }
                // Qwen3NextModel.modify_tensors
                if relative.ends_with(".A_log") {
                    transforms.push(Transform::NegExp);
                } else if let Some(base) = relative.strip_suffix(".dt_bias") {
                    relative = format!("{base}.dt_proj.bias");
                } else if relative.contains("conv1d") {
                    if !transforms.contains(&Transform::Squeeze) {
                        transforms.push(Transform::Squeeze);
                    }
                } else if relative.ends_with("norm.weight") && !relative.ends_with("linear_attn.norm.weight") {
                    transforms.push(Transform::AddOne);
                }
            }
            Arch::Gemma2 => {
                if relative.ends_with("norm.weight") {
                    transforms.push(Transform::AddOne);
                }
            }
            Arch::Gemma3 => {
                if relative.contains("embed_tokens.weight")
                    && let Some(rows) = gemma3_rows
                {
                    transforms.push(Transform::TruncateRows(rows));
                }
                if relative.ends_with("norm.weight") {
                    transforms.push(Transform::AddOne);
                }
            }
            Arch::Qwen2 | Arch::Qwen3 => {}
        }

        let gguf_name = gguf_tensor_name(arch, &relative).ok_or_else(|| {
            anyhow!(
                "Can not map tensor {} to a {} GGUF tensor",
                tensor.name,
                arch.gguf_name()
            )
        })?;
        if let Some(previous) = seen.insert(gguf_name.clone(), tensor.name.clone()) {
            bail!("Tensors {previous} and {} both map to {gguf_name}", tensor.name);
        }

        // Output shape after the shape-changing transforms.
        let mut shape = tensor.shape.clone();
        for transform in &transforms {
            match transform {
                Transform::Squeeze => shape.retain(|&d| d != 1),
                Transform::TruncateRows(rows) => shape[0] = (*rows).min(shape[0]),
                _ => {}
            }
        }
        let ggml_type = if requires_f32(&gguf_name, shape.len()) {
            GgmlType::F32
        } else {
            out_type.ggml()
        };
        planned.push(Planned {
            info: GgufTensorInfo {
                name: gguf_name,
                ggml_type,
                shape: shape.iter().map(|&d| d as u64).collect(),
            },
            data: Data::Source(index),
            transforms,
            source_shape: tensor.shape.clone(),
        });
    }

    if planned.iter().all(|p| p.info.name != "token_embd.weight") {
        bail!("The checkpoint has no token embeddings (embed_tokens.weight)");
    }
    if skipped > 0 {
        reporter.debug(format!("Skipped {skipped} tensors that are not part of the text model"));
    }

    let infos: Vec<GgufTensorInfo> = planned.iter().map(|p| p.info.clone()).collect();
    let total = planned.len();
    let tensors = source.tensors();
    write_gguf(output, &meta, &infos, |i| {
        let plan = &planned[i];
        reporter.progress("Writing GGUF", i as u64, Some(total as u64));
        match &plan.data {
            Data::RopeFreqs(values) => encode(values, plan.info.ggml_type),
            Data::Source(index) => {
                let source_tensor = &tensors[*index];
                let raw = source.read(*index)?;
                if plan.transforms.is_empty() && same_type(source_tensor.dtype, plan.info.ggml_type) {
                    return Ok(raw);
                }
                let mut data = decode_f32(&raw, source_tensor.dtype)?;
                drop(raw);
                let mut shape = plan.source_shape.clone();
                for transform in &plan.transforms {
                    data = apply(transform, data, &mut shape)
                        .with_context(|| format!("Failed to convert {}", source_tensor.name))?;
                }
                encode(&data, plan.info.ggml_type)
            }
        }
    })?;
    reporter.progress("Writing GGUF", total as u64, Some(total as u64));

    Ok(ConvertedGguf {
        path: output.to_path_buf(),
        architecture: arch.gguf_name(),
        tensor_count: total,
        out_type,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_llama_and_qwen2_tensor_names() {
        assert_eq!(
            gguf_tensor_name(Arch::Llama, "layers.3.self_attn.q_proj.weight").as_deref(),
            Some("blk.3.attn_q.weight")
        );
        assert_eq!(
            gguf_tensor_name(Arch::Llama, "layers.0.post_attention_layernorm.weight").as_deref(),
            Some("blk.0.ffn_norm.weight")
        );
        assert_eq!(gguf_tensor_name(Arch::Llama, "layers.0.self_attn.q_norm.weight"), None);
        assert_eq!(
            gguf_tensor_name(Arch::Qwen2, "layers.12.self_attn.k_proj.bias").as_deref(),
            Some("blk.12.attn_k.bias")
        );
        assert_eq!(
            gguf_tensor_name(Arch::Qwen2, "embed_tokens.weight").as_deref(),
            Some("token_embd.weight")
        );
        assert_eq!(
            gguf_tensor_name(Arch::Qwen2, "norm.weight").as_deref(),
            Some("output_norm.weight")
        );
        assert_eq!(
            gguf_tensor_name(Arch::Qwen2, "lm_head.weight").as_deref(),
            Some("output.weight")
        );
        assert_eq!(
            gguf_tensor_name(Arch::Qwen35, "layers.0.linear_attn.A_log").as_deref(),
            Some("blk.0.ssm_a")
        );
        assert_eq!(
            gguf_tensor_name(Arch::Qwen35, "layers.0.linear_attn.dt_proj.bias").as_deref(),
            Some("blk.0.ssm_dt.bias")
        );
        assert_eq!(
            gguf_tensor_name(Arch::Gemma3, "layers.1.pre_feedforward_layernorm.weight").as_deref(),
            Some("blk.1.ffn_norm.weight")
        );
    }

    #[test]
    fn normalizes_checkpoint_names() {
        assert_eq!(
            relative_name("model.language_model.layers.0.mlp.down_proj.weight").as_deref(),
            Some("layers.0.mlp.down_proj.weight")
        );
        assert_eq!(relative_name("lm_head.weight").as_deref(), Some("lm_head.weight"));
        assert_eq!(relative_name("model.visual.blocks.0.attn.qkv.weight"), None);
        assert_eq!(relative_name("mtp.layers.0.mlp.up_proj.weight"), None);
        assert_eq!(relative_name("model.vision_tower.encoder.weight"), None);
        assert_eq!(relative_name("multi_modal_projector.linear.weight"), None);
    }

    #[test]
    fn tensor_types_follow_upstream_rules() {
        assert!(requires_f32("blk.0.attn_norm.weight", 1));
        assert!(requires_f32("blk.0.ssm_conv1d.weight", 2));
        assert!(requires_f32("blk.0.ssm_a", 1));
        assert!(requires_f32("blk.0.attn_q.bias", 1));
        assert!(!requires_f32("blk.0.attn_q.weight", 2));
        assert!(!requires_f32("token_embd.weight", 2));
    }

    #[test]
    fn permute_matches_pytorch_reference() {
        // permute(w, n_head=2): reshape(2, 2, 1, cols).swapaxes(1, 2) on 4 rows.
        let mut shape = vec![4, 1];
        let out = apply(&Transform::Permute { heads: 2 }, vec![0.0, 1.0, 2.0, 3.0], &mut shape).unwrap();
        assert_eq!(out, vec![0.0, 1.0, 2.0, 3.0]);
        // 8 rows, 2 heads: half = 2; head rows [a0 a1 b0 b1] -> [a0 b0 a1 b1].
        let mut shape = vec![8];
        let out = apply(
            &Transform::Permute { heads: 2 },
            (0..8).map(|v| v as f32).collect(),
            &mut shape,
        )
        .unwrap();
        assert_eq!(out, vec![0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]);
    }

    #[test]
    fn v_head_reorder_matches_reference() {
        // 2 K heads x 2 V heads per K, head_dim 1: grouped [G0v0 G0v1 G1v0 G1v1] -> tiled [G0v0 G1v0 G0v1 G1v1].
        let mut shape = vec![4];
        let reorder = Transform::ReorderRows {
            offset: 0,
            num_k_heads: 2,
            num_v_per_k: 2,
            head_dim: 1,
        };
        let out = apply(&reorder, vec![0.0, 1.0, 2.0, 3.0], &mut shape).unwrap();
        assert_eq!(out, vec![0.0, 2.0, 1.0, 3.0]);
        let mut shape = vec![1, 4];
        let reorder = Transform::ReorderColumns {
            num_k_heads: 2,
            num_v_per_k: 2,
            head_dim: 1,
        };
        assert_eq!(
            apply(&reorder, vec![0.0, 1.0, 2.0, 3.0], &mut shape).unwrap(),
            vec![0.0, 2.0, 1.0, 3.0]
        );
    }

    #[test]
    fn llama3_rope_factors_match_reference() {
        let rope: Map<String, Value> = serde_json::from_value(serde_json::json!({
            "rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0,
            "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192
        }))
        .unwrap();
        let factors = llama3_rope_factors(&rope, 128).unwrap();
        assert_eq!(factors.len(), 64);
        assert_eq!(factors[0], 1.0);
        assert_eq!(factors[63], 8.0);
    }

    #[test]
    fn detects_architectures() {
        let config = serde_json::json!({"architectures": ["MistralForCausalLM"]});
        assert_eq!(Arch::detect(&config).unwrap(), Arch::Llama);
        let config = serde_json::json!({"architectures": ["Qwen3_5ForConditionalGeneration"]});
        assert_eq!(Arch::detect(&config).unwrap(), Arch::Qwen35);
        let error = Arch::detect(&serde_json::json!({"architectures": ["MambaForCausalLM"]}))
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not support MambaForCausalLM"), "{error}");
    }
}
