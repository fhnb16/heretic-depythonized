// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Tests that need access to the Hugging Face Hub. Run with `cargo test -- --ignored`.

use heretic_core::Prompt;
use heretic_core::dataset::{DatasetSpecification, SingleDatasetSpecification, load_prompts};
use heretic_core::hub::{RepoType, ResolvedRepo};
use heretic_core::tokenizer::ModelTokenizer;

#[test]
#[ignore = "requires network access"]
fn loads_pinned_hub_dataset() {
    let mut spec = DatasetSpecification::Single(SingleDatasetSpecification {
        dataset: "mlabonne/harmless_alpaca".into(),
        commit: Some("02c6a92cfcf11bb0c387334f8146d149d65b587f".into()),
        split: Some("test[:5]".into()),
        column: Some("text".into()),
        ..Default::default()
    });
    let prompts = load_prompts(&mut spec, "You are a helpful assistant.").unwrap();
    assert_eq!(prompts.len(), 5);
    assert!(prompts.iter().all(|p| !p.user.is_empty()));
    println!("{:#?}", &prompts[..2]);
}

#[test]
#[ignore = "requires network access"]
fn unpinned_dataset_gets_pinned() {
    let mut spec = DatasetSpecification::Single(SingleDatasetSpecification {
        dataset: "mlabonne/harmful_behaviors".into(),
        split: Some("train[:3]".into()),
        column: Some("text".into()),
        ..Default::default()
    });
    let prompts = load_prompts(&mut spec, "sys").unwrap();
    assert_eq!(prompts.len(), 3);
    let commit = spec.specifications()[0].commit.clone().unwrap();
    assert_eq!(commit.len(), 40);
}

#[test]
#[ignore = "requires network access"]
fn tiny_qwen_tokenizer_and_template() {
    let repo = ResolvedRepo::open(
        RepoType::Model,
        "tiny-random/qwen2.5",
        Some("7a6a3128ee4137a248d6d1582824592b87a81647"),
    )
    .unwrap();
    let tokenizer = ModelTokenizer::from_repo(&repo).unwrap();
    assert!(tokenizer.supports_system_role());

    let rendered = tokenizer
        .render_prompt(&Prompt::new("You are a helpful assistant.", "What is 1+1?"))
        .unwrap();
    println!("{rendered}");
    assert_eq!(
        rendered,
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n"
    );

    let batch = tokenizer.encode_batch(&[rendered.clone(), "Hi".into()]).unwrap();
    assert_eq!(batch.input_ids[0].len(), batch.input_ids[1].len());
    assert_eq!(batch.input_ids[1][0], tokenizer.pad_token_id);
    let decoded = tokenizer.decode(&batch.input_ids[0], false).unwrap();
    assert_eq!(decoded, rendered);
}
