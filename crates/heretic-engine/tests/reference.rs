// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Numerical parity against the independent implementations in candle-transformers.
//! Run with `cargo test -p heretic-engine --release --test reference -- --ignored --nocapture`.

use std::sync::Arc;

use candle_core::{D, DType, Device, Tensor};
use heretic_core::Settings;
use heretic_core::prompt::Prompt;
use heretic_core::report::NullReporter;
use heretic_engine::Model;

fn load(model: &str) -> Model {
    let settings = Settings {
        model: model.into(),
        dtypes: vec!["float32".into()],
        device_map: heretic_core::config::DeviceMap::Device("cpu".into()),
        ..Settings::default()
    };
    Model::load(&settings, Arc::new(NullReporter)).unwrap()
}

fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
    (a - b).unwrap().abs().unwrap().max_all().unwrap().to_scalar().unwrap()
}

fn top_tokens(model: &Model, logits: &Tensor, k: usize) -> Vec<(String, f32)> {
    let values: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    indices
        .into_iter()
        .take(k)
        .map(|i| (model.tokenizer.decode(&[i as u32], false).unwrap(), values[i]))
        .collect()
}

/// Computes our logits and the reference logits for the same prompt.
fn compare(model_id: &str, prefix: &str, user: &str, reference: impl FnOnce(&Model, &Tensor) -> Tensor) {
    let mut model = load(model_id);
    model.options.response_prefix = prefix.into();

    let prompt = Prompt::new("You are a helpful assistant.", user);
    let text = model.render_prompts(std::slice::from_ref(&prompt)).unwrap().remove(0);
    let ids = model.tokenizer.encode(&text, true).unwrap();

    let ours = model.get_logits(std::slice::from_ref(&prompt)).unwrap();
    println!("ours: {:?}", top_tokens(&model, &ours, 5));

    let input = Tensor::new(ids.as_slice(), &Device::Cpu).unwrap().unsqueeze(0).unwrap();
    let theirs = reference(&model, &input);
    println!("reference: {:?}", top_tokens(&model, &theirs, 5));

    let diff = max_diff(&ours, &theirs);
    let scale: f32 = theirs
        .abs()
        .unwrap()
        .max_keepdim(D::Minus1)
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar()
        .unwrap();
    println!("max |diff| = {diff} (max |logit| = {scale})");
    assert!(diff < 1e-3 * scale.max(1.0), "{diff}");
}

fn var_builder(model: &Model) -> candle_nn::VarBuilder<'static> {
    unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&model.weight_files, DType::F32, &Device::Cpu).unwrap() }
}

#[test]
#[ignore = "downloads a model"]
fn qwen2_matches_candle_transformers() {
    compare(
        "Qwen/Qwen2.5-0.5B-Instruct",
        "",
        "What is 2+2? Answer with a number only.",
        |model, input| {
            let config: candle_transformers::models::qwen2::Config =
                serde_json::from_value(model.config_json.clone()).unwrap();
            let mut reference =
                candle_transformers::models::qwen2::ModelForCausalLM::new(&config, var_builder(model)).unwrap();
            reference.forward(input, 0).unwrap().squeeze(1).unwrap()
        },
    );
}

#[test]
#[ignore = "downloads a model"]
fn qwen3_matches_candle_transformers() {
    compare(
        "Qwen/Qwen3-0.6B",
        "<think>\n\n</think>\n\n",
        "Name the largest planet in the solar system. One word.",
        |model, input| {
            let config: candle_transformers::models::qwen3::Config =
                serde_json::from_value(model.config_json.clone()).unwrap();
            let mut reference =
                candle_transformers::models::qwen3::ModelForCausalLM::new(&config, var_builder(model)).unwrap();
            let logits = reference.forward(input, 0).unwrap();
            logits.reshape((1, logits.elem_count())).unwrap()
        },
    );
}
