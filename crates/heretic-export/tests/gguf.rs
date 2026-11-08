// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! GGUF export of real checkpoints, read back with the GGUF reader.
//! Run with `cargo test -p heretic-export --release --test gguf -- --ignored --nocapture`.
//!
//! Set `HERETIC_TEST_LLAMA_CPP` to a llama.cpp folder to also test quantization.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{DType, Tensor};
use heretic_core::Settings;
use heretic_core::config::DeviceMap;
use heretic_core::hub::{RepoType, ResolvedRepo};
use heretic_core::report::{NullReporter, SharedReporter};
use heretic_engine::{MLP_DOWN_PROJ, Model};
use heretic_export::gguf::{self, GgufType, GgufValue, TensorSource};

fn reporter() -> SharedReporter {
    Arc::new(NullReporter)
}

fn decode_f16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| half::f16::from_le_bytes(*c).to_f32())
        .collect()
}

fn string(header: &gguf::GgufHeader, key: &str) -> String {
    header
        .metadata
        .get(key)
        .and_then(GgufValue::as_str)
        .unwrap_or_else(|| panic!("missing {key}"))
        .to_string()
}

fn snapshot(id: &str, commit: Option<&str>) -> PathBuf {
    let repo = ResolvedRepo::open(RepoType::Model, id, commit).unwrap();
    for file in repo.files.clone() {
        repo.get(&file, None).unwrap();
    }
    repo.get("config.json", None).unwrap().parent().unwrap().to_path_buf()
}

#[test]
#[ignore = "downloads a model"]
fn merged_adapters_are_written_to_gguf() {
    let settings = Settings {
        model: "tiny-random/qwen2.5".into(),
        model_commit: Some("7a6a3128ee4137a248d6d1582824592b87a81647".into()),
        dtypes: vec!["float32".into()],
        device_map: DeviceMap::Device("cpu".into()),
        ..Settings::default()
    };
    let model = Model::load(&settings, reporter()).unwrap();
    model.apply_lora(2);
    let module = model.layer_modules(1)[MLP_DOWN_PROJ][0].clone();
    let a = Tensor::randn(0f32, 0.5, (2, module.in_features()), &model.device).unwrap();
    let b = Tensor::randn(0f32, 0.5, (module.out_features(), 2), &model.device).unwrap();
    module.set_adapter(a, b).unwrap();

    let dir = tempfile::tempdir_in(env!("CARGO_TARGET_TMPDIR")).unwrap();
    let export = gguf::export_gguf(&model, "tiny-qwen", dir.path(), GgufType::F16, None, &reporter()).unwrap();
    assert_eq!(export.path, dir.path().join("tiny-qwen-F16.gguf"));
    assert_eq!(export.architecture, "qwen2");

    let header = gguf::read_gguf_header(&export.path).unwrap();
    assert_eq!(header.architecture(), Some("qwen2"));
    assert_eq!(string(&header, "tokenizer.ggml.model"), "gpt2");
    println!("tokenizer.ggml.pre = {}", string(&header, "tokenizer.ggml.pre"));
    assert_eq!(
        header.metadata.get("qwen2.block_count").and_then(GgufValue::as_u64),
        Some(model.num_layers() as u64)
    );
    assert!(header.metadata.contains("tokenizer.chat_template"));

    // The modified weight in the GGUF file is the merged weight, rounded to F16.
    let info = header.tensor("blk.1.ffn_down.weight").unwrap();
    assert_eq!(
        info.shape,
        vec![module.out_features() as u64, module.in_features() as u64]
    );
    let written = decode_f16(&gguf::read_gguf_tensor(&export.path, &header, "blk.1.ffn_down.weight").unwrap());
    let adapter = module.adapter().unwrap();
    let expected = (module.weight() + adapter.b.matmul(&adapter.a).unwrap())
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let max_diff = written
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("max |gguf - merged| = {max_diff}");
    assert!(max_diff < 2e-2, "{max_diff}");

    // Norms are stored as F32.
    assert_eq!(
        header.tensor("blk.0.attn_norm.weight").unwrap().ggml_type,
        gguf::writer::GgmlType::F32
    );

    // Every text tensor of the checkpoint is in the file.
    let merged = heretic_export::MergedTensors::open(&model).unwrap();
    assert_eq!(header.tensors.len(), merged.tensors().len());
}

fn check_directory(out: &Path, id: &str, arch: &str, pre: &str) -> PathBuf {
    let dir = snapshot(id, None);
    let export = gguf::convert_directory(&dir, out, GgufType::BF16, None, &reporter()).unwrap();
    let header = gguf::read_gguf_header(&export.path).unwrap();
    assert_eq!(header.architecture(), Some(arch));
    assert_eq!(string(&header, "tokenizer.ggml.pre"), pre, "{id}");
    println!(
        "{id}: {} tensors, {} metadata keys, {:.1} MB",
        header.tensors.len(),
        header.metadata.len(),
        export.bytes as f64 / 1e6
    );
    export.path
}

#[test]
#[ignore = "downloads models"]
fn converts_qwen_checkpoints() {
    let out = tempfile::tempdir_in(env!("CARGO_TARGET_TMPDIR")).unwrap();
    let qwen25 = check_directory(out.path(), "Qwen/Qwen2.5-0.5B-Instruct", "qwen2", "qwen2");
    let header = gguf::read_gguf_header(&qwen25).unwrap();
    assert!(header.tensor("blk.0.attn_q.bias").is_some());

    check_directory(out.path(), "Qwen/Qwen3-0.6B", "qwen3", "qwen2");

    let qwen35 = check_directory(
        out.path(),
        "RikiyaT/Qwen3.5-0.8B-weight-space-programming",
        "qwen35",
        "qwen35",
    );
    let header = gguf::read_gguf_header(&qwen35).unwrap();
    // Linear attention layers and the gated full attention layers.
    assert!(header.tensor("blk.0.ssm_a").is_some());
    assert_eq!(header.tensor("blk.0.ssm_conv1d.weight").unwrap().shape.len(), 2);
    assert!(header.tensor("blk.3.attn_q.weight").is_some());
    assert!(
        header
            .tensors
            .iter()
            .all(|t| !t.name.contains("visual") && !t.name.contains("mtp"))
    );
    let recurrent = header
        .metadata
        .get("qwen35.attention.recurrent_layers")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(recurrent.len(), 24);

    if let Ok(llama_cpp) = std::env::var("HERETIC_TEST_LLAMA_CPP") {
        let out = out.path();
        let dir = snapshot("Qwen/Qwen2.5-0.5B-Instruct", None);
        let export = gguf::convert_directory(
            &dir,
            &out.join("q4.gguf"),
            GgufType::Q4_K_M,
            Some(Path::new(&llama_cpp)),
            &reporter(),
        )
        .unwrap();
        println!("Q4_K_M: {:.1} MB", export.bytes as f64 / 1e6);
        assert!(export.bytes < std::fs::metadata(&qwen25).unwrap().len() / 2);
        // The intermediate file is removed.
        assert!(!out.join("q4.bf16.tmp.gguf").exists());
    }
}
