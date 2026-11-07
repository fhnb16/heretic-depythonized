// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! KL divergence of the first-token distributions from the original model's.
//! Measures how much the model's behavior has drifted (lower = less damage).

use anyhow::Result;
use candle_core::{D, DType, Tensor};
use heretic_core::dataset::{DatasetSpecification, SingleDatasetSpecification};
use heretic_core::prompt::Prompt;
use heretic_core::report::ReporterExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Context, Score, Scorer};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct Settings {
    /// Dataset of prompts used to measure KL divergence from original model.
    pub prompts: DatasetSpecification,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            prompts: DatasetSpecification::Single(SingleDatasetSpecification {
                dataset: "mlabonne/harmless_alpaca".into(),
                split: Some("test[:100]".into()),
                column: Some("text".into()),
                ..Default::default()
            }),
        }
    }
}

pub struct KlDivergence {
    pub settings: Settings,
    prompts: Vec<Prompt>,
    baseline_logprobs: Option<Tensor>,
}

/// `F.kl_div(logprobs, target_logprobs, reduction="batchmean", log_target=True)`.
pub fn kl_divergence_batchmean(logprobs: &Tensor, target_logprobs: &Tensor) -> Result<f64> {
    let batch = logprobs.dim(0)?;
    let pointwise = (target_logprobs.exp()? * (target_logprobs - logprobs)?)?;
    let sum: f64 = pointwise.to_dtype(DType::F64)?.sum_all()?.to_scalar()?;
    Ok(sum / batch as f64)
}

impl KlDivergence {
    pub fn new(settings: Settings) -> Self {
        Self {
            settings,
            prompts: Vec::new(),
            baseline_logprobs: None,
        }
    }
}

impl Scorer for KlDivergence {
    fn class_name(&self) -> &'static str {
        "KLDivergence"
    }

    fn score_name(&self) -> String {
        "KL divergence".into()
    }

    fn init(&mut self, ctx: &Context) -> Result<()> {
        ctx.reporter.heading(format!(
            "Loading KL divergence evaluation prompts from **{}**...",
            self.settings.prompts.describe()
        ));
        self.prompts = ctx.load_prompts(&mut self.settings.prompts)?;
        ctx.reporter
            .info(format!("* **{}** prompts loaded", self.prompts.len()));
        ctx.reporter
            .info("* Obtaining baseline first-token probability distributions...");
        let logits = ctx.get_logits(&self.prompts)?;
        self.baseline_logprobs = Some(candle_nn::ops::log_softmax(&logits, D::Minus1)?);
        Ok(())
    }

    fn get_score(&mut self, ctx: &Context) -> Result<Score> {
        let logits = ctx.get_logits(&self.prompts)?;
        let logprobs = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
        let baseline = self
            .baseline_logprobs
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("KLDivergence scorer was not initialized"))?;
        let value = kl_divergence_batchmean(&logprobs, baseline)?;
        Ok(Score {
            value,
            display: format!("**{value:.4}**"),
            md_display: format!("{value:.4}"),
        })
    }

    fn get_baseline_score(&mut self, _ctx: &Context) -> Result<Score> {
        Ok(Score {
            value: 0.0,
            display: "**0** (by definition)".into(),
            md_display: "0 *(by definition)*".into(),
        })
    }

    fn dataset_specifications(&self) -> Vec<DatasetSpecification> {
        vec![self.settings.prompts.clone()]
    }

    fn settings_json(&self) -> Value {
        serde_json::to_value(&self.settings).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn kl_matches_definition() -> Result<()> {
        let device = Device::Cpu;
        let p = Tensor::new(&[[0.5f32, 0.5], [0.9, 0.1]], &device)?.log()?;
        let q = Tensor::new(&[[0.25f32, 0.75], [0.9, 0.1]], &device)?.log()?;
        // KL(target=p || q) per row: 0.5 ln(0.5/0.25) + 0.5 ln(0.5/0.75); second row 0.
        let expected = (0.5 * (2.0f64).ln() + 0.5 * (0.5f64 / 0.75).ln()) / 2.0;
        let value = kl_divergence_batchmean(&q, &p)?;
        assert!((value - expected).abs() < 1e-6, "{value} vs {expected}");
        assert!(kl_divergence_batchmean(&p, &p)?.abs() < 1e-7);
        Ok(())
    }
}
