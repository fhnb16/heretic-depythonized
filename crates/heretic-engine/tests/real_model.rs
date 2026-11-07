// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! End-to-end checks with real (small) instruction-tuned models.
//! Run with `cargo test -p heretic-engine --release --test real_model -- --ignored --nocapture`.

use std::sync::Arc;

use heretic_core::Settings;
use heretic_core::prompt::Prompt;
use heretic_core::report::StderrReporter;
use heretic_engine::{ATTN_O_PROJ, MLP_DOWN_PROJ, Model};

fn load(model: &str) -> Model {
    let settings = Settings {
        model: model.into(),
        dtypes: vec!["float32".into()],
        device_map: heretic_core::config::DeviceMap::Device("cpu".into()),
        max_response_length: 24,
        ..Settings::default()
    };
    Model::load(&settings, Arc::new(StderrReporter)).unwrap()
}

fn check_model(model_id: &str, response_prefix: &str) {
    let mut model = load(model_id);
    model.options.batch_size = 4;
    assert_eq!(model.lora_rank(), None);
    model.options.response_prefix = response_prefix.into();

    let system = "You are a helpful assistant.";
    let questions = [
        "What is the capital of France? Answer in one word.",
        "What color is the sky on a clear day? Answer in one word.",
        "Name the largest planet in the solar system. One word.",
    ];
    let prompts: Vec<Prompt> = questions.iter().map(|q| Prompt::new(system, *q)).collect();

    let batched = model.get_responses_batched(&prompts, true).unwrap();
    println!("{model_id}: {batched:#?}");
    assert!(batched[0].contains("Paris"), "{}", batched[0]);
    assert!(batched[1].to_lowercase().contains("blue"), "{}", batched[1]);
    // Too hard for the smallest models (Qwen3-0.6B answers "Mars", which the reference
    // implementation confirms), so only check that there is an answer.
    assert!(!batched[2].trim().is_empty());

    // Left padding must not change the result.
    for (prompt, expected) in prompts.iter().zip(&batched) {
        let single = model.get_responses(std::slice::from_ref(prompt), true).unwrap();
        assert_eq!(&single[0], expected);
    }

    let layers = model.num_layers();
    let hidden = model.config().hidden_size;

    let residuals = model.get_residuals_batched(&prompts, 1.0).unwrap();
    assert_eq!(residuals.dims(), &[3, layers + 1, hidden]);

    let mean = model.get_residuals_mean(&prompts, 1.0).unwrap();
    assert_eq!(mean.dims(), &[layers + 1, hidden]);

    let logits = model.get_logits_batched(&prompts).unwrap();
    assert_eq!(logits.dims()[0], 3);

    let io = model.get_module_io_batched(&prompts).unwrap();
    assert_eq!(io.len(), layers);
    let (input, output) = &io[0][ATTN_O_PROJ][&0];
    assert_eq!(input.dims()[0], 3);
    assert_eq!(output.dims(), &[3, hidden]);
    assert!(io[layers - 1].contains_key(MLP_DOWN_PROJ));

    // The captured module output must equal the module applied to the captured input.
    let module = &model.layer_modules(0)[ATTN_O_PROJ][0];
    let recomputed = module.forward(input).unwrap();
    let difference: f32 = (recomputed - output)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!(difference < 1e-4, "{difference}");

    // Adapters change the output, and resetting them restores it.
    model.apply_lora(1);
    let before = model.get_logits(&prompts[..1]).unwrap();
    let module = &model.layer_modules(layers / 2)[MLP_DOWN_PROJ][0];
    let a = heretic_engine::Tensor::ones((1, module.in_features()), heretic_engine::DType::F32, &model.device).unwrap();
    let b = (heretic_engine::Tensor::ones((module.out_features(), 1), heretic_engine::DType::F32, &model.device)
        .unwrap()
        * 0.01)
        .unwrap();
    module.set_adapter(a, b).unwrap();
    let changed = model.get_logits(&prompts[..1]).unwrap();
    model.reset_adapters();
    let after = model.get_logits(&prompts[..1]).unwrap();
    let diff = |x: &heretic_engine::Tensor, y: &heretic_engine::Tensor| -> f32 {
        (x - y).unwrap().abs().unwrap().max_all().unwrap().to_scalar().unwrap()
    };
    assert!(diff(&before, &changed) > 1e-3);
    assert_eq!(diff(&before, &after), 0.0);
}

#[test]
#[ignore = "downloads a model"]
fn qwen2_5_instruct() {
    check_model("Qwen/Qwen2.5-0.5B-Instruct", "");
}

#[test]
#[ignore = "downloads a model"]
fn qwen3() {
    // A closed chain-of-thought block skips the thinking phase.
    check_model("Qwen/Qwen3-0.6B", "<think>\n\n</think>\n\n");
}
