// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Smoke tests for every supported architecture family on tiny random checkpoints:
//! loading, left-padding invariance of the logits (masks, linear attention padding),
//! residuals/module I/O and cached generation.
//!
//! Run with `cargo test --release -p heretic-engine --test architectures -- --ignored --nocapture`.
//! Set `HERETIC_ARCH_MODELS=org/model[@commit],...` to test other (e.g. real) models; their
//! greedy answers are printed so they can be checked for coherence.

use std::sync::Arc;

use heretic_core::Settings;
use heretic_core::prompt::Prompt;
use heretic_core::report::StderrReporter;
use heretic_engine::Model;

const TINY_MODELS: &[(&str, &str)] = &[
    ("tiny-random/qwen2.5", "7a6a3128ee4137a248d6d1582824592b87a81647"),
    ("tiny-random/qwen3.5", "07ad2f323e908adc8d023136610ded7656a40ff0"),
    ("tiny-random/gemma-2", "ebd61568c424fcc6a4e42a69b2aba7c64447ffc9"),
    ("tiny-random/gemma-3", "69a78d1ad0ad66620c43579acd1327553713e22a"),
    ("tiny-random/gemma-4e", "3a207ada2c2cd95e9671942e84cf47ea58f0f6af"),
];

fn load(model: &str, commit: Option<&str>) -> Model {
    let dtype = std::env::var("HERETIC_TEST_DTYPE").unwrap_or_else(|_| "float32".into());
    let mut toml = format!("model = \"{model}\"\nbatch_size = 4\nmax_response_length = 24\ndtypes = [\"{dtype}\"]\n");
    if let Some(commit) = commit {
        toml.push_str(&format!("model_commit = \"{commit}\"\n"));
    }
    if let Ok(device) = std::env::var("HERETIC_TEST_DEVICE") {
        toml.push_str(&format!("device_map = \"{device}\"\n"));
    }
    let table: heretic_core::toml::Table = heretic_core::toml::from_str(&toml).unwrap();
    let settings = Settings::from_table(table).unwrap();
    Model::load(&settings, Arc::new(StderrReporter)).unwrap_or_else(|e| panic!("{model}: {e:#}"))
}

fn check(model_id: &str, commit: Option<&str>) {
    eprintln!("=== {model_id}");
    let model = load(model_id, commit);
    let short = Prompt::new("", "What is the capital of France?");
    let long = Prompt::new(
        "You are a helpful assistant.",
        "Explain in two sentences why the sky is blue during the day and red at sunset.",
    );

    // Left padding must not change the logits of the shorter prompt.
    let alone = model.get_logits(std::slice::from_ref(&short)).unwrap();
    let batched = model.get_logits(&[long.clone(), short.clone()]).unwrap();
    let padded = batched.narrow(0, 1, 1).unwrap();
    let scale = alone
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .max(1e-6);
    let diff = (alone - padded)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    eprintln!("padding invariance: max |delta logits| = {diff:.2e} (scale {scale:.2e})");
    // Batched half-precision matrix products round differently; float32 must agree closely.
    let tolerance = if model.dtype == heretic_engine::DType::F32 {
        2e-3
    } else {
        3e-2
    };
    assert!(
        diff / scale < tolerance,
        "{model_id}: left padding changes the logits ({diff} / {scale})"
    );

    let residuals = model.get_residuals(&[short.clone(), long.clone()], 1.0).unwrap();
    assert_eq!(residuals.dims()[1], model.num_layers() + 1);
    let io = model.get_module_io(std::slice::from_ref(&short)).unwrap();
    assert_eq!(io.len(), model.num_layers());
    assert!(
        io.iter()
            .all(|layer| layer.contains_key("attn.o_proj") && layer.contains_key("mlp.down_proj"))
    );

    let responses = model.get_responses(&[short, long], true).unwrap();
    for response in &responses {
        eprintln!("response: {response:?}");
    }
}

#[test]
#[ignore = "downloads models"]
fn architectures() {
    match std::env::var("HERETIC_ARCH_MODELS") {
        Ok(list) => {
            for entry in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                let (model, commit) = match entry.split_once('@') {
                    Some((model, commit)) => (model, Some(commit)),
                    None => (entry, None),
                };
                check(model, commit);
            }
        }
        Err(_) => {
            for (model, commit) in TINY_MODELS {
                check(model, Some(commit));
            }
        }
    }
}
