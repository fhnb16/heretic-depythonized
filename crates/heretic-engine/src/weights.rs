// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Memory-mapped access to the safetensors files of a model.

use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::{Context as _, Result, anyhow, bail};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Tensor};
use heretic_core::hub::ResolvedRepo;

/// Lists the safetensors files of a repository (sharded or not).
pub fn safetensors_files(repo: &ResolvedRepo) -> Result<Vec<String>> {
    if repo.has_file("model.safetensors.index.json") {
        let index_path = repo.get("model.safetensors.index.json", None)?;
        let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)
            .with_context(|| format!("Invalid {}", index_path.display()))?;
        let files: BTreeSet<String> = index
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| anyhow!("model.safetensors.index.json has no weight_map"))?
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        return Ok(files.into_iter().collect());
    }

    if repo.has_file("model.safetensors") {
        return Ok(vec!["model.safetensors".into()]);
    }

    let files: Vec<String> = repo
        .files
        .iter()
        .filter(|f| f.ends_with(".safetensors") && !f.contains('/'))
        .cloned()
        .collect();
    if files.is_empty() {
        bail!(
            "{} contains no safetensors weights (PyTorch .bin checkpoints are not supported)",
            repo.id
        );
    }
    Ok(files)
}

/// The weights of a model, with the location of the text model inside the checkpoint.
pub struct WeightStore {
    tensors: MmapedSafetensors,
    names: BTreeSet<String>,
    /// Prefix of the text model's decoder (e.g. "model." or "model.language_model.").
    pub text_prefix: String,
    pub files: Vec<PathBuf>,
}

impl WeightStore {
    pub fn open(paths: Vec<PathBuf>) -> Result<WeightStore> {
        // SAFETY: The files are not modified while mapped (they live in the model cache).
        let tensors = unsafe { MmapedSafetensors::multi(&paths) }.context("Failed to map safetensors files")?;
        let names: BTreeSet<String> = tensors.tensors().into_iter().map(|(name, _)| name).collect();
        let text_prefix = detect_text_prefix(&names)?;
        Ok(WeightStore {
            tensors,
            names,
            text_prefix,
            files: paths,
        })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.names.contains(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.names.iter()
    }

    /// Loads a tensor with an exact checkpoint name.
    pub fn load(&self, name: &str, dtype: DType, device: &Device) -> Result<Tensor> {
        let tensor = self
            .tensors
            .load(name, &Device::Cpu)
            .with_context(|| format!("Failed to load tensor {name}"))?;
        let tensor = if tensor.dtype() == dtype {
            tensor
        } else {
            tensor.to_dtype(dtype)?
        };
        Ok(tensor.to_device(device)?)
    }

    /// Loads a tensor of the text model, relative to the text prefix (e.g. "layers.0.mlp.down_proj.weight").
    pub fn load_text(&self, relative: &str, dtype: DType, device: &Device) -> Result<(String, Tensor)> {
        let name = format!("{}{relative}", self.text_prefix);
        let tensor = self.load(&name, dtype, device)?;
        Ok((name, tensor))
    }

    pub fn load_text_optional(&self, relative: &str, dtype: DType, device: &Device) -> Result<Option<Tensor>> {
        let name = format!("{}{relative}", self.text_prefix);
        if self.contains(&name) {
            Ok(Some(self.load(&name, dtype, device)?))
        } else {
            Ok(None)
        }
    }

    pub fn has_text(&self, relative: &str) -> bool {
        self.contains(&format!("{}{relative}", self.text_prefix))
    }

    /// Finds the language modeling head, which lives outside the decoder.
    pub fn lm_head_name(&self) -> Option<String> {
        let candidates = [
            "lm_head.weight".to_string(),
            "language_model.lm_head.weight".to_string(),
            "model.language_model.lm_head.weight".to_string(),
            format!("{}lm_head.weight", self.text_prefix),
        ];
        candidates.into_iter().find(|name| self.contains(name))
    }
}

/// Detects the prefix under which the decoder layers and embeddings are stored.
fn detect_text_prefix(names: &BTreeSet<String>) -> Result<String> {
    let mut prefixes: Vec<String> = names
        .iter()
        .filter_map(|name| name.strip_suffix("embed_tokens.weight").map(String::from))
        .collect();
    // Prefer the shortest prefix that also contains decoder layers
    // (vision towers can have their own embeddings).
    prefixes.sort_by_key(String::len);
    for prefix in &prefixes {
        if names.iter().any(|n| n.starts_with(&format!("{prefix}layers.0."))) {
            return Ok(prefix.clone());
        }
    }
    bail!("Could not locate the text model in the checkpoint (no embed_tokens.weight next to decoder layers)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_prefix_detection() {
        let names: BTreeSet<String> = [
            "model.embed_tokens.weight",
            "model.layers.0.mlp.down_proj.weight",
            "lm_head.weight",
        ]
        .map(String::from)
        .into();
        assert_eq!(detect_text_prefix(&names).unwrap(), "model.");

        let names: BTreeSet<String> = [
            "model.vision_tower.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
        ]
        .map(String::from)
        .into();
        assert_eq!(detect_text_prefix(&names).unwrap(), "model.language_model.");
    }
}
