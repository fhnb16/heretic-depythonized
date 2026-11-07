// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Model runtime of Heretic, built on candle: loading safetensors checkpoints,
//! generation, and access to residuals, module inputs/outputs and LoRA adapters.

pub mod config;
pub mod device;
pub mod generate;
pub mod lora;
pub mod model;
pub mod transformer;
pub mod weights;

pub use candle_core::{DType, Device, Tensor};
pub use lora::LoraLinear;
pub use model::{Cancelled, InferenceOptions, Model, is_cancelled};
pub use transformer::{ATTN_O_PROJ, MLP_DOWN_PROJ, ModuleIo};
