// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Saving a modified model and loading it again must reproduce the modified behavior.
//! Run with `cargo test -p heretic-export --release --test roundtrip -- --ignored --nocapture`.

use std::sync::Arc;

use candle_core::{DType, Tensor};
use heretic_core::Settings;
use heretic_core::config::DeviceMap;
use heretic_core::prompt::Prompt;
use heretic_core::report::{NullReporter, SharedReporter};
use heretic_engine::{MLP_DOWN_PROJ, Model};

fn settings(model: &str) -> Settings {
    Settings {
        model: model.into(),
        dtypes: vec!["float32".into()],
        device_map: DeviceMap::Device("cpu".into()),
        ..Settings::default()
    }
}

fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
    (a - b).unwrap().abs().unwrap().max_all().unwrap().to_scalar().unwrap()
}

#[test]
#[ignore = "downloads a model"]
fn merged_model_reproduces_adapter_output() {
    let reporter: SharedReporter = Arc::new(NullReporter);
    let model = Model::load(&settings("Qwen/Qwen2.5-0.5B-Instruct"), reporter.clone()).unwrap();
    let prompts = vec![Prompt::new("You are a helpful assistant.", "Tell me a joke.")];

    let original = model.get_logits(&prompts).unwrap();

    model.apply_lora(2);
    for layer in [5, 10, 15] {
        let module = &model.layer_modules(layer)[MLP_DOWN_PROJ][0];
        let a = (Tensor::randn(0f32, 0.02, (2, module.in_features()), &model.device).unwrap())
            .to_dtype(DType::F32)
            .unwrap();
        let b = Tensor::randn(0f32, 0.02, (module.out_features(), 2), &model.device).unwrap();
        module.set_adapter(a, b).unwrap();
    }
    let modified = model.get_logits(&prompts).unwrap();
    assert!(max_diff(&original, &modified) > 1e-2);

    // The system temp directory may be on a small drive; use the target directory instead.
    let dir = tempfile::tempdir_in(env!("CARGO_TARGET_TMPDIR")).unwrap();
    let merged_dir = dir.path().join("merged");
    // A small shard size forces a sharded checkpoint with an index.
    let saved = heretic_export::save_merged(&model, &merged_dir, 300_000_000, &reporter).unwrap();
    println!("{:#?}", saved.files);
    assert!(saved.weight_files.len() > 1);
    assert!(merged_dir.join("model.safetensors.index.json").is_file());
    assert!(merged_dir.join("tokenizer.json").is_file());
    assert!(!merged_dir.join("README.md").exists());

    let reloaded = Model::load(&settings(merged_dir.to_str().unwrap()), reporter.clone()).unwrap();
    let reloaded_logits = reloaded.get_logits(&prompts).unwrap();
    let diff = max_diff(&modified, &reloaded_logits);
    println!("max |modified - reloaded| = {diff}");
    assert!(diff < 1e-3, "{diff}");

    let adapter_dir = dir.path().join("adapter");
    heretic_export::save_adapter(&model, &adapter_dir).unwrap();
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(adapter_dir.join("adapter_config.json")).unwrap()).unwrap();
    assert_eq!(config["r"], 2);
    assert_eq!(
        config["target_modules"].as_array().unwrap().len(),
        2 * model.num_layers()
    );
    let bytes = std::fs::read(adapter_dir.join("adapter_model.safetensors")).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let a = tensors
        .tensor("base_model.model.model.layers.5.mlp.down_proj.lora_A.weight")
        .unwrap();
    assert_eq!(a.shape(), &[2, model.layer_modules(5)[MLP_DOWN_PROJ][0].in_features()]);
}
