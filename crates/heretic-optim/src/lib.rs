// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Optimization machinery: a port of Optuna's multivariate (multi-objective) TPE sampler
//! with persistent studies, plus the numerical optimizers used by modifiers.

pub mod distributions;
pub mod hypervolume;
pub mod lbfgs;
pub mod lowrank;
pub mod special;
pub mod study;
pub mod tpe;

pub use distributions::Distribution;
pub use study::{Direction, FrozenTrial, Study, Trial, TrialState};
pub use tpe::TpeConfig;
