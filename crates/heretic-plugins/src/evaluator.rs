// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Evaluation of the model with the configured scorers.

use std::collections::BTreeSet;

use anyhow::{Result, bail};
use heretic_core::config::{Optimization, ScorerConfig};
use heretic_core::dataset::DatasetSpecification;
use heretic_core::report::ReporterExt;
use heretic_optim::Direction;
use serde_json::{Value, json};

use crate::{Context, Score, Scorer, canonical_plugin_name, create_scorer, is_builtin_plugin};

pub struct ScorerEntry {
    pub scorer: Box<dyn Scorer>,
    /// Display name (score name, plus instance name if any).
    pub name: String,
    pub config: ScorerConfig,
}

/// Manages evaluation of the model using the configured scorers: loads them,
/// establishes baseline scores, and runs them during optimization.
pub struct Evaluator {
    pub entries: Vec<ScorerEntry>,
    pub baseline_scores: Vec<(String, Score)>,
}

impl Evaluator {
    /// Loads and initializes all scorers, then computes the baseline scores.
    pub fn new(ctx: &Context) -> Result<Self> {
        let configs = &ctx.settings.scorers;
        if configs.is_empty() {
            bail!("No scorers configured. Set \"scorers\" in config.toml");
        }

        ctx.reporter.heading("Loading and initializing scorers...");

        let mut keys = BTreeSet::new();
        let mut entries = Vec::new();
        for config in configs {
            let scorer = create_scorer(config, ctx.settings)?;
            let class_name = canonical_plugin_name(&config.plugin)?;
            let instance = config.instance_name.as_deref();
            ctx.reporter.info(format!(
                "* Loaded: **{class_name}{}**",
                instance.map(|i| format!(" - {i}")).unwrap_or_default()
            ));

            let key = match instance {
                Some(instance) => format!("{class_name}_{instance}"),
                None => class_name.clone(),
            };
            if !keys.insert(key.clone()) {
                bail!("Duplicate scorer instance name: {key}. Give each instance a unique `instance_name`.");
            }

            let name = match instance {
                Some(instance) => format!("{} - {instance}", scorer.score_name()),
                None => scorer.score_name(),
            };
            entries.push(ScorerEntry {
                scorer,
                name,
                config: config.clone(),
            });
        }

        for entry in entries.iter_mut() {
            entry.scorer.init(ctx)?;
        }

        let mut evaluator = Self {
            entries,
            baseline_scores: Vec::new(),
        };

        ctx.reporter.heading("Getting baseline scores...");
        let mut baseline = Vec::new();
        for entry in evaluator.entries.iter_mut() {
            let score = entry.scorer.get_baseline_score(ctx)?;
            ctx.reporter
                .info(format!("* Baseline **{}:** {}", entry.name, score.display));
            baseline.push((entry.name.clone(), score));
        }
        evaluator.baseline_scores = baseline;

        Ok(evaluator)
    }

    /// Runs all scorers.
    pub fn get_scores(&mut self, ctx: &Context) -> Result<Vec<(String, Score)>> {
        let mut scores = Vec::with_capacity(self.entries.len());
        for entry in self.entries.iter_mut() {
            scores.push((entry.name.clone(), entry.scorer.get_score(ctx)?));
        }
        Ok(scores)
    }

    pub fn dataset_specifications(&self) -> Vec<DatasetSpecification> {
        self.entries
            .iter()
            .flat_map(|e| e.scorer.dataset_specifications())
            .collect()
    }

    pub fn all_scorers_reproducible(&self) -> bool {
        self.entries.iter().all(|e| e.scorer.reproducible())
    }

    pub fn all_scorers_builtin(&self) -> bool {
        self.entries.iter().all(|e| is_builtin_plugin(&e.config.plugin))
    }

    /// Scorer entries that participate in optimization, in canonical order. Every
    /// objective-derived list (names, directions, values) is built from this so that
    /// they stay positionally aligned.
    fn objective_entries(&self) -> impl Iterator<Item = &ScorerEntry> {
        self.entries
            .iter()
            .filter(|e| e.config.optimization != Optimization::None)
    }

    pub fn objective_names(&self) -> Vec<String> {
        self.objective_entries().map(|e| e.name.clone()).collect()
    }

    pub fn objective_directions(&self) -> Vec<Direction> {
        self.objective_entries()
            .map(|e| match e.config.optimization {
                Optimization::Maximize => Direction::Maximize,
                _ => Direction::Minimize,
            })
            .collect()
    }

    /// Objective values, aligned with `objective_names` and `objective_directions`.
    pub fn objective_values(&self, scores: &[(String, Score)]) -> Vec<f64> {
        self.objective_entries()
            .map(|entry| {
                scores
                    .iter()
                    .find(|(name, _)| *name == entry.name)
                    .map(|(_, score)| score.value)
                    .unwrap_or(f64::NAN)
            })
            .collect()
    }

    /// Pairs each trial score with its baseline into one serializable record.
    pub fn paired_score_records(&self, scores: &[(String, Score)]) -> Value {
        let records: Vec<Value> = scores
            .iter()
            .zip(&self.baseline_scores)
            .map(|((name, score), (_, baseline))| {
                json!({
                    "name": name,
                    "score": score,
                    "baseline": baseline,
                })
            })
            .collect();
        Value::Array(records)
    }
}
