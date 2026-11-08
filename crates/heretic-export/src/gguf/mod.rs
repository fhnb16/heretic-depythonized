// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! GGUF export: a pure Rust Hugging Face → GGUF converter (unquantized F16/BF16 tensors) and
//! quantization with `llama-quantize` from a llama.cpp build the user downloaded.

pub mod convert;
pub mod llama_cpp;
mod pre_tokenizer;
pub mod vocab;
pub mod writer;

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, anyhow};
use heretic_core::report::{ReporterExt, SharedReporter};
use heretic_engine::Model;
use serde_json::Value;

pub use convert::{Arch, OutType, SUPPORTED_ARCHITECTURES, SafetensorsDirectory, SourceTensor, TensorSource};
pub use llama_cpp::{GgufType, LLAMA_CPP_RELEASES, LlamaCpp};
pub use vocab::{DirectoryFiles, ModelFiles};
pub use writer::{GgufHeader, GgufValue, read_gguf_header, read_gguf_tensor};

/// Result of a GGUF export.
#[derive(Debug, Clone)]
pub struct GgufExport {
    pub path: PathBuf,
    pub kind: GgufType,
    pub bytes: u64,
    pub architecture: &'static str,
}

/// File name characters that are safe everywhere.
fn sanitize(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '-'
            }
        })
        .collect();
    let cleaned = cleaned.trim_matches(['-', '.']).to_string();
    if cleaned.is_empty() { "model".into() } else { cleaned }
}

/// The output file: `output` itself if it ends in `.gguf`, otherwise `<output>/<name>-<TYPE>.gguf`.
pub fn output_path(output: &Path, name: &str, kind: GgufType) -> PathBuf {
    if output.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
        output.to_path_buf()
    } else {
        output.join(format!("{}-{}.gguf", sanitize(name), kind.name()))
    }
}

/// Removes a file when dropped (the intermediate unquantized GGUF).
struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[allow(clippy::too_many_arguments)]
fn run_export(
    config: &Value,
    files: &dyn ModelFiles,
    source: &dyn TensorSource,
    name: &str,
    output: &Path,
    kind: GgufType,
    llama_cpp_dir: Option<&Path>,
    reporter: &SharedReporter,
) -> Result<GgufExport> {
    // Fail before the long conversion if llama.cpp is missing or unsupported.
    convert::Arch::detect(config)?;
    let llama_cpp = if kind.needs_quantize() {
        let llama_cpp = LlamaCpp::locate(llama_cpp_dir.unwrap_or(Path::new("")))?;
        llama_cpp.check()?;
        reporter.info(format!(
            "* llama.cpp: **{}**{}",
            llama_cpp.directory.display(),
            llama_cpp
                .version
                .as_ref()
                .map(|v| format!(" (version {v})"))
                .unwrap_or_default()
        ));
        Some(llama_cpp)
    } else {
        None
    };

    let path = output_path(output, name, kind);
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).with_context(|| format!("Failed to create {}", parent.display()))?;
    }

    let out_type = match kind {
        GgufType::F16 => OutType::F16,
        GgufType::BF16 => OutType::BF16,
        _ => OutType::guess(source.tensors()),
    };

    match llama_cpp {
        None => {
            convert::convert_to_gguf(config, files, source, name, out_type, &path, reporter)?;
        }
        Some(llama_cpp) => {
            let suffix = format!("{out_type:?}.tmp.gguf").to_lowercase();
            let intermediate = TempFile(path.with_extension(suffix));
            reporter.info("* Writing the unquantized GGUF file...");
            convert::convert_to_gguf(config, files, source, name, out_type, &intermediate.0, reporter)?;
            llama_cpp.quantize(&intermediate.0, &path, kind, reporter)?;
        }
    }

    read_gguf_header(&path).context("The written GGUF file cannot be read back")?;
    let bytes = std::fs::metadata(&path)?.len();
    Ok(GgufExport {
        architecture: convert::Arch::detect(config)?.gguf_name(),
        path,
        kind,
        bytes,
    })
}

/// Exports a model with its LoRA adapters merged as a GGUF file of the given type.
///
/// `output` is a `.gguf` file or a folder (the file is then named `<name>-<TYPE>.gguf`).
/// Quantized types need `llama_cpp_dir`, a llama.cpp folder containing `llama-quantize`.
pub fn export_gguf(
    model: &Model,
    name: &str,
    output: &Path,
    kind: GgufType,
    llama_cpp_dir: Option<&Path>,
    reporter: &SharedReporter,
) -> Result<GgufExport> {
    let config_path = model.repo.get("config.json", None)?;
    let config: Value = serde_json::from_str(&std::fs::read_to_string(&config_path)?)
        .with_context(|| format!("Invalid JSON in {}", config_path.display()))?;
    let merged = crate::MergedTensors::open(model)?;
    reporter.info(format!(
        "Exporting **{}** as GGUF ({} modified modules)...",
        kind.name(),
        merged.modified_count()
    ));
    run_export(
        &config,
        &model.repo,
        &merged,
        name,
        output,
        kind,
        llama_cpp_dir,
        reporter,
    )
}

/// Converts a Hugging Face model folder (safetensors) to a GGUF file of the given type.
pub fn convert_directory(
    dir: &Path,
    output: &Path,
    kind: GgufType,
    llama_cpp_dir: Option<&Path>,
    reporter: &SharedReporter,
) -> Result<GgufExport> {
    let config: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("config.json"))
            .with_context(|| format!("{} has no config.json", dir.display()))?,
    )?;
    let source = SafetensorsDirectory::open(dir)?;
    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .ok_or_else(|| anyhow!("Invalid model folder {}", dir.display()))?;
    let files = DirectoryFiles(dir.to_path_buf());
    run_export(&config, &files, &source, &name, output, kind, llama_cpp_dir, reporter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_paths() {
        assert_eq!(
            output_path(Path::new("out/model.gguf"), "x", GgufType::Q4_K_M),
            PathBuf::from("out/model.gguf")
        );
        assert_eq!(
            output_path(Path::new("out"), "Qwen/Qwen3 4B", GgufType::Q4_K_M),
            PathBuf::from("out").join("Qwen-Qwen3-4B-Q4_K_M.gguf")
        );
    }
}
