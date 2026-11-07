// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Measures the mean length of responses.

use anyhow::Result;
use heretic_core::dataset::{DatasetSpecification, SingleDatasetSpecification};
use heretic_core::prompt::Prompt;
use heretic_core::report::ReporterExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Context, Score, Scorer, report_response};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct Settings {
    /// Dataset of prompts to evaluate the response length on.
    pub prompts: DatasetSpecification,
    /// Whether to strip leading and trailing whitespace from responses before measuring their length.
    pub strip_responses: bool,
    /// Length in characters below which a response is classified as a 'short response'.
    pub short_response_threshold: usize,
    /// Length value artificially assigned to 'short responses' for the purpose of scoring.
    /// This can be used to penalize very short (and thus likely invalid) responses.
    pub short_response_score_length: f64,
    /// Whether to print prompt/response pairs.
    pub print_responses: bool,
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
            strip_responses: true,
            // Chosen because "No" is the shortest common valid response.
            short_response_threshold: 2,
            short_response_score_length: 500.0,
            print_responses: false,
        }
    }
}

pub struct ResponseLength {
    pub settings: Settings,
    prompts: Vec<Prompt>,
}

impl ResponseLength {
    pub fn new(settings: Settings) -> Self {
        Self {
            settings,
            prompts: Vec::new(),
        }
    }

    /// Length in characters (Unicode code points, like Python's `len`).
    pub fn response_length(&self, response: &str) -> f64 {
        let text = if self.settings.strip_responses {
            response.trim()
        } else {
            response
        };
        let length = text.chars().count();
        if length < self.settings.short_response_threshold {
            self.settings.short_response_score_length
        } else {
            length as f64
        }
    }
}

impl Scorer for ResponseLength {
    fn class_name(&self) -> &'static str {
        "ResponseLength"
    }

    fn score_name(&self) -> String {
        "Mean response length".into()
    }

    fn init(&mut self, ctx: &Context) -> Result<()> {
        ctx.reporter.heading(format!(
            "Loading response length evaluation prompts from **{}**...",
            self.settings.prompts.describe()
        ));
        self.prompts = ctx.load_prompts(&mut self.settings.prompts)?;
        ctx.reporter
            .info(format!("* **{}** prompts loaded", self.prompts.len()));
        Ok(())
    }

    fn get_score(&mut self, ctx: &Context) -> Result<Score> {
        let responses = ctx.get_responses(&self.prompts)?;
        let mut total = 0.0;
        for (prompt, response) in self.prompts.iter().zip(&responses) {
            total += self.response_length(response);
            if self.settings.print_responses {
                report_response(ctx.reporter, prompt, response, "");
            }
        }
        let mean = if self.prompts.is_empty() {
            0.0
        } else {
            total / self.prompts.len() as f64
        };
        Ok(Score {
            value: mean,
            display: format!("**{mean:.1}**"),
            md_display: format!("{mean:.1}"),
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

    #[test]
    fn lengths() {
        let scorer = ResponseLength::new(Settings::default());
        assert_eq!(scorer.response_length("  Hallo  "), 5.0);
        assert_eq!(scorer.response_length("Ä"), 500.0);
        assert_eq!(scorer.response_length("No"), 2.0);
        assert_eq!(scorer.response_length("Привет"), 6.0);
    }
}
