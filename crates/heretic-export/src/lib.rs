// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Exporting modified models: merged safetensors checkpoints (streamed tensor by
//! tensor, so memory use doesn't grow with model size) and PEFT LoRA adapters.

pub mod gguf;
pub mod hub_upload;
pub mod model_card;
pub mod safetensors_writer;

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context as _, Result, anyhow, bail};
use candle_core::{DType, Device, Tensor};
use heretic_core::report::{ReporterExt, SharedReporter};
use heretic_engine::{LoraLinear, Model};
use safetensors::Dtype;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::safetensors_writer::{PlannedTensor, write_safetensors};

/// Files that are never copied to an exported model.
fn is_excluded_file(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".safetensors")
        || lower.ends_with(".bin")
        || lower.ends_with(".pt")
        || lower.ends_with(".pth")
        || lower.ends_with(".gguf")
        || lower.ends_with(".msgpack")
        || lower.ends_with(".h5")
        || lower.ends_with(".onnx")
        || lower == "model.safetensors.index.json"
        || lower == "pytorch_model.bin.index.json"
        || lower == "readme.md"
        || lower.starts_with('.')
        || lower.contains("/.")
        || lower.starts_with("onnx/")
        || lower.starts_with("reproduce/")
        || lower.contains('/') && !lower.starts_with("tokenizer")
}

pub fn candle_to_safetensors_dtype(dtype: DType) -> Result<Dtype> {
    Ok(match dtype {
        DType::F32 => Dtype::F32,
        DType::F16 => Dtype::F16,
        DType::BF16 => Dtype::BF16,
        DType::F64 => Dtype::F64,
        other => bail!("Unsupported export dtype {other:?}"),
    })
}

fn safetensors_to_candle_dtype(dtype: Dtype) -> Option<DType> {
    Some(match dtype {
        Dtype::F32 => DType::F32,
        Dtype::F16 => DType::F16,
        Dtype::BF16 => DType::BF16,
        Dtype::F64 => DType::F64,
        _ => return None,
    })
}

fn is_float(dtype: Dtype) -> bool {
    matches!(dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32 | Dtype::F64)
}

/// Raw little-endian bytes of a tensor.
pub fn tensor_bytes(tensor: &Tensor) -> Result<Vec<u8>> {
    let tensor = tensor.to_device(&Device::Cpu)?.flatten_all()?;
    Ok(match tensor.dtype() {
        DType::F32 => tensor.to_vec1::<f32>()?.iter().flat_map(|v| v.to_le_bytes()).collect(),
        DType::F64 => tensor.to_vec1::<f64>()?.iter().flat_map(|v| v.to_le_bytes()).collect(),
        DType::BF16 => tensor
            .to_vec1::<half::bf16>()?
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect(),
        DType::F16 => tensor
            .to_vec1::<half::f16>()?
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect(),
        other => bail!("Unsupported dtype {other:?}"),
    })
}

/// SHA-256 of a file, as a lowercase hex string.
pub fn file_sha256(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let count = std::io::Read::read(&mut file, &mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

struct CheckpointTensor {
    name: String,
    file: usize,
    dtype: Dtype,
    shape: Vec<usize>,
    offset: usize,
}

/// The tensors of a model's checkpoint with the LoRA adapters merged into the modified
/// weights, read one at a time from the memory-mapped source files.
///
/// Like PEFT's `merge_and_unload`, modified weights become `W + (B @ A)` computed in the
/// model's dtype, and all floating point tensors are produced in the dtype the model was
/// loaded with.
pub struct MergedTensors<'a> {
    model: &'a Model,
    maps: Vec<memmap2::Mmap>,
    sources: Vec<CheckpointTensor>,
    modules: HashMap<String, Arc<LoraLinear>>,
    tensors: Vec<gguf::SourceTensor>,
}

impl<'a> MergedTensors<'a> {
    pub fn open(model: &'a Model) -> Result<MergedTensors<'a>> {
        let target_dtype = candle_to_safetensors_dtype(model.dtype)?;

        let modules: HashMap<String, Arc<LoraLinear>> = model
            .all_modules()
            .into_iter()
            .map(|module| (format!("{}.weight", module.name), module))
            .collect();

        // Map the source files and collect their tensors in file order.
        let mut maps = Vec::new();
        for path in &model.weight_files {
            let file = std::fs::File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
            // SAFETY: The source files are not modified while mapped.
            let map = unsafe { memmap2::Mmap::map(&file)? };
            maps.push(map);
        }

        let mut sources = Vec::new();
        for (index, map) in maps.iter().enumerate() {
            let (_, metadata) = safetensors::SafeTensors::read_metadata(map)
                .map_err(|e| anyhow!("Invalid safetensors file {}: {e}", model.weight_files[index].display()))?;
            let mut entries: Vec<CheckpointTensor> = metadata
                .tensors()
                .into_iter()
                .map(|(name, info)| CheckpointTensor {
                    name: name.clone(),
                    file: index,
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    offset: info.data_offsets.0,
                })
                .collect();
            entries.sort_by_key(|e| e.offset);
            sources.extend(entries);
        }

        let tensors = sources
            .iter()
            .map(|source| gguf::SourceTensor {
                name: source.name.clone(),
                dtype: if is_float(source.dtype) {
                    target_dtype
                } else {
                    source.dtype
                },
                shape: source.shape.clone(),
            })
            .collect();

        Ok(MergedTensors {
            model,
            maps,
            sources,
            modules,
            tensors,
        })
    }

    /// Number of modules whose weights change.
    pub fn modified_count(&self) -> usize {
        self.modules.values().filter(|m| m.has_adapter()).count()
    }
}

impl gguf::TensorSource for MergedTensors<'_> {
    fn tensors(&self) -> &[gguf::SourceTensor] {
        &self.tensors
    }

    fn read(&self, index: usize) -> Result<Vec<u8>> {
        let source = &self.sources[index];
        let raw = safetensors::SafeTensors::deserialize(&self.maps[source.file])
            .map_err(|e| anyhow!("{e}"))?
            .tensor(&source.name)
            .map_err(|e| anyhow!("{e}"))?
            .data()
            .to_vec();

        if let Some(module) = self.modules.get(&source.name)
            && let Some(adapter) = module.adapter()
        {
            // PEFT: delta = B @ A, cast to the weight dtype, added in that dtype.
            let delta = adapter.b.matmul(&adapter.a)?.to_dtype(self.model.dtype)?;
            let merged = (module.weight() + delta)?;
            return tensor_bytes(&merged);
        }

        let out_dtype = self.tensors[index].dtype;
        if out_dtype == source.dtype {
            return Ok(raw);
        }
        let source_dtype = safetensors_to_candle_dtype(source.dtype).ok_or_else(|| anyhow!("Unsupported dtype"))?;
        let tensor = Tensor::from_raw_buffer(&raw, source_dtype, &source.shape, &Device::Cpu)?;
        tensor_bytes(&tensor.to_dtype(self.model.dtype)?)
    }
}

/// Result of saving a model.
#[derive(Debug, Clone)]
pub struct SavedModel {
    pub directory: PathBuf,
    /// Weight files written (relative names).
    pub weight_files: Vec<String>,
    /// All files written (relative names).
    pub files: Vec<String>,
}

fn copy_auxiliary_files(model: &Model, dir: &Path, saved_dtype: Option<DType>, files: &mut Vec<String>) -> Result<()> {
    for name in &model.repo.files {
        if is_excluded_file(name) {
            continue;
        }
        let source = model.repo.get(name, None)?;
        let target = dir.join(name);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }

        if name == "config.json"
            && let Some(dtype) = saved_dtype
        {
            // Record the dtype the weights were saved in, like save_pretrained does.
            let mut config: Value = serde_json::from_str(&std::fs::read_to_string(&source)?)?;
            let dtype_name = heretic_engine::device::dtype_name(dtype);
            set_dtype_recursive(&mut config, dtype_name);
            if let Value::Object(map) = &mut config {
                map.remove("quantization_config");
            }
            std::fs::write(&target, serde_json::to_string_pretty(&config)? + "\n")?;
        } else {
            std::fs::copy(&source, &target).with_context(|| format!("Failed to copy {name}"))?;
        }
        files.push(name.clone());
    }
    Ok(())
}

fn set_dtype_recursive(config: &mut Value, dtype: &str) {
    if let Value::Object(map) = config {
        for key in ["torch_dtype", "dtype"] {
            if map.contains_key(key) {
                map.insert(key.to_string(), json!(dtype));
            }
        }
        if !map.contains_key("torch_dtype") && !map.contains_key("dtype") && map.contains_key("architectures") {
            map.insert("torch_dtype".into(), json!(dtype));
        }
        for key in ["text_config", "llm_config", "language_config"] {
            if let Some(sub) = map.get_mut(key) {
                set_dtype_recursive(sub, dtype);
            }
        }
    }
}

/// Saves the model with the LoRA adapters merged into the weights.
///
/// Like PEFT's `merge_and_unload` followed by `save_pretrained`, modified weights become
/// `W + (B @ A)` computed in the model's dtype, and all floating point tensors are saved
/// in the dtype the model was loaded with.
pub fn save_merged(model: &Model, dir: &Path, max_shard_size: u64, reporter: &SharedReporter) -> Result<SavedModel> {
    std::fs::create_dir_all(dir).with_context(|| format!("Failed to create {}", dir.display()))?;

    let merged = MergedTensors::open(model)?;
    let sources = gguf::TensorSource::tensors(&merged);
    let byte_size =
        |source: &gguf::SourceTensor| -> usize { source.shape.iter().product::<usize>() * source.dtype.bitsize() / 8 };

    // Split into shards greedily in checkpoint order, like huggingface_hub does.
    let mut shards: Vec<Vec<usize>> = vec![Vec::new()];
    let mut current_size = 0u64;
    for (index, source) in sources.iter().enumerate() {
        let size = byte_size(source) as u64;
        if !shards.last().unwrap().is_empty() && current_size + size > max_shard_size {
            shards.push(Vec::new());
            current_size = 0;
        }
        shards.last_mut().unwrap().push(index);
        current_size += size;
    }

    let shard_count = shards.len();
    let shard_name = |i: usize| {
        if shard_count == 1 {
            "model.safetensors".to_string()
        } else {
            format!("model-{:05}-of-{:05}.safetensors", i + 1, shard_count)
        }
    };

    let mut weight_map = BTreeMap::new();
    let mut total_size = 0u64;
    let mut weight_files = Vec::new();
    let merged_count = merged.modified_count();
    reporter.info(format!(
        "* Writing **{}** tensors in **{shard_count}** file(s) (**{merged_count}** modified)...",
        sources.len()
    ));

    for (shard_index, shard) in shards.iter().enumerate() {
        let file_name = shard_name(shard_index);
        let planned: Vec<PlannedTensor> = shard
            .iter()
            .map(|&i| {
                let source = &sources[i];
                PlannedTensor {
                    name: source.name.clone(),
                    dtype: source.dtype,
                    shape: source.shape.clone(),
                }
            })
            .collect();

        let produce = |position: usize| -> Result<Vec<u8>> { gguf::TensorSource::read(&merged, shard[position]) };

        let path = dir.join(&file_name);
        reporter.progress(
            &format!("Writing {file_name}"),
            shard_index as u64,
            Some(shard_count as u64),
        );
        write_safetensors(&path, &planned, produce)?;

        for tensor in &planned {
            weight_map.insert(tensor.name.clone(), file_name.clone());
            total_size += (tensor.shape.iter().product::<usize>() * tensor.dtype.bitsize() / 8) as u64;
        }
        weight_files.push(file_name);
    }
    reporter.progress("Writing weights", shard_count as u64, Some(shard_count as u64));

    let mut files = weight_files.clone();
    if shard_count > 1 {
        let index = json!({
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        });
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_string_pretty(&index)? + "\n",
        )?;
        files.push("model.safetensors.index.json".into());
    }

    copy_auxiliary_files(model, dir, Some(model.dtype), &mut files)?;
    files.sort();

    Ok(SavedModel {
        directory: dir.to_path_buf(),
        weight_files,
        files,
    })
}

/// Name of a module in the PEFT adapter format.
fn peft_module_name(module_name: &str) -> String {
    format!("base_model.model.{module_name}")
}

/// Saves only the LoRA adapters, in PEFT format (loadable with `PeftModel.from_pretrained`).
pub fn save_adapter(model: &Model, dir: &Path) -> Result<SavedModel> {
    std::fs::create_dir_all(dir).with_context(|| format!("Failed to create {}", dir.display()))?;
    let rank = model
        .lora_rank()
        .ok_or_else(|| anyhow!("The model has no LoRA adapters"))?;

    let modules = model.all_modules();
    let mut planned = Vec::new();
    let mut data: Vec<Vec<u8>> = Vec::new();
    let mut target_modules = Vec::new();

    for module in &modules {
        let (a, b) = match module.adapter() {
            Some(adapter) => (adapter.a, adapter.b),
            // Untouched modules get zero adapters (identity), like PEFT saves them.
            None => (
                Tensor::zeros((rank, module.in_features()), DType::F32, &Device::Cpu)?,
                Tensor::zeros((module.out_features(), rank), DType::F32, &Device::Cpu)?,
            ),
        };
        let prefix = peft_module_name(&module.name);
        for (suffix, tensor) in [("lora_A.weight", a), ("lora_B.weight", b)] {
            let tensor = tensor.to_dtype(DType::F32)?;
            planned.push(PlannedTensor {
                name: format!("{prefix}.{suffix}"),
                dtype: Dtype::F32,
                shape: tensor.dims().to_vec(),
            });
            data.push(tensor_bytes(&tensor)?);
        }
        target_modules.push(module.name.clone());
    }
    target_modules.sort();

    write_safetensors(&dir.join("adapter_model.safetensors"), &planned, |i| {
        Ok(std::mem::take(&mut data[i]))
    })?;

    let config = json!({
        "alpha_pattern": {},
        "auto_mapping": null,
        "base_model_name_or_path": model.model_id,
        "bias": "none",
        "fan_in_fan_out": false,
        "inference_mode": true,
        "init_lora_weights": true,
        "layers_pattern": null,
        "layers_to_transform": null,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "modules_to_save": null,
        "peft_type": "LORA",
        "r": rank,
        "rank_pattern": {},
        "revision": model.repo.commit,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "use_dora": false,
        "use_rslora": false
    });
    std::fs::write(
        dir.join("adapter_config.json"),
        serde_json::to_string_pretty(&config)? + "\n",
    )?;

    Ok(SavedModel {
        directory: dir.to_path_buf(),
        weight_files: vec!["adapter_model.safetensors".into()],
        files: vec!["adapter_config.json".into(), "adapter_model.safetensors".into()],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn excluded_files() {
        assert!(is_excluded_file("model-00001-of-00002.safetensors"));
        assert!(is_excluded_file("README.md"));
        assert!(is_excluded_file("onnx/model.onnx"));
        assert!(!is_excluded_file("config.json"));
        assert!(!is_excluded_file("tokenizer.json"));
        assert!(!is_excluded_file("chat_template.jinja"));
    }

    #[test]
    fn dtype_is_recorded_in_config() {
        let mut config = json!({"architectures": ["X"], "torch_dtype": "float32", "text_config": {"dtype": "float32"}});
        set_dtype_recursive(&mut config, "bfloat16");
        assert_eq!(config["torch_dtype"], "bfloat16");
        assert_eq!(config["text_config"]["dtype"], "bfloat16");
    }
}
