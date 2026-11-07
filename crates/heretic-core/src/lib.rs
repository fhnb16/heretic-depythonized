// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Core building blocks of Heretic that don't depend on a tensor library:
//! settings, datasets, the Hugging Face Hub client, tokenization and chat templates.

pub mod chat;
pub mod config;
pub mod dataset;
pub mod hub;
pub mod prompt;
pub mod report;
pub mod tokenizer;
pub mod util;

pub use config::Settings;
pub use prompt::{Message, Prompt};
pub use toml;
