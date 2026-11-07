// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

pub mod abliteration;
pub mod ara;

use anyhow::Result;
use candle_core::{D, Tensor};
use heretic_core::dataset::{DatasetSpecification, SingleDatasetSpecification};

/// `F.normalize(x, p=2, dim)`: divides by the L2 norm, clamped to at least 1e-12.
pub(crate) fn normalize(x: &Tensor, dim: usize) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(dim)?.sqrt()?.clamp(1e-12f32, f32::MAX)?;
    Ok(x.broadcast_div(&norm)?)
}

/// L2 norms of the rows of a matrix, shape (rows, 1).
pub(crate) fn row_norms(x: &Tensor) -> Result<Tensor> {
    Ok(x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?)
}

pub(crate) fn default_good_prompts() -> DatasetSpecification {
    DatasetSpecification::Single(SingleDatasetSpecification {
        dataset: "mlabonne/harmless_alpaca".into(),
        split: Some("train[:400]".into()),
        column: Some("text".into()),
        ..Default::default()
    })
}

pub(crate) fn default_bad_prompts() -> DatasetSpecification {
    DatasetSpecification::Single(SingleDatasetSpecification {
        dataset: "mlabonne/harmful_behaviors".into(),
        split: Some("train[:400]".into()),
        column: Some("text".into()),
        ..Default::default()
    })
}
