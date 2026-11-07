// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Modifiers (which change the model) and scorers (which evaluate it).
//!
//! Heretic 2.x (Python) loaded these as plugins from Python files. Here they are
//! compiled in and selected by name; the fully qualified names from 2.x
//! (e.g. `heretic.scorers.keyword_rate.KeywordRate`) are accepted as aliases.

pub mod evaluator;
pub mod modifiers;
pub mod scorers;

use std::cell::RefCell;
use std::collections::HashMap;

use anyhow::{Result, bail};
use candle_core::Tensor;
use heretic_core::Settings;
use heretic_core::config::{ModifierConfig, ScorerConfig};
use heretic_core::dataset::{DatasetSpecification, load_prompts};
use heretic_core::prompt::Prompt;
use heretic_core::report::SharedReporter;
use heretic_engine::Model;
use heretic_optim::Trial;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Result of evaluating a scorer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Score {
    /// Scalar value used for optimization (if enabled).
    pub value: f64,
    /// Formatted value shown to the user (may contain `**bold**` spans).
    pub display: String,
    /// Formatted value for the model card (Markdown).
    pub md_display: String,
}

/// Runtime context passed to plugins: access to the model and cached responses.
pub struct Context<'a> {
    pub settings: &'a Settings,
    pub model: &'a Model,
    pub reporter: &'a SharedReporter,
    responses: RefCell<HashMap<Vec<Prompt>, Vec<String>>>,
}

impl<'a> Context<'a> {
    pub fn new(settings: &'a Settings, model: &'a Model, reporter: &'a SharedReporter) -> Self {
        Self {
            settings,
            model,
            reporter,
            responses: RefCell::new(HashMap::new()),
        }
    }

    /// Model responses for the prompts (special tokens removed), cached within this context
    /// so that multiple scorers evaluating the same prompts don't regenerate them.
    pub fn get_responses(&self, prompts: &[Prompt]) -> Result<Vec<String>> {
        if let Some(cached) = self.responses.borrow().get(prompts) {
            return Ok(cached.clone());
        }
        let responses = self.model.get_responses_batched(prompts, true)?;
        self.responses.borrow_mut().insert(prompts.to_vec(), responses.clone());
        Ok(responses)
    }

    pub fn get_logits(&self, prompts: &[Prompt]) -> Result<Tensor> {
        self.model.get_logits_batched(prompts)
    }

    pub fn get_residuals(&self, prompts: &[Prompt]) -> Result<Tensor> {
        self.model.get_residuals_batched(prompts, 1.0)
    }

    /// Loads prompts, pinning Hub datasets to their latest commit if unpinned.
    pub fn load_prompts(&self, specification: &mut DatasetSpecification) -> Result<Vec<Prompt>> {
        load_prompts(specification, &self.settings.system_prompt)
    }
}

pub trait Scorer: Send {
    /// The class name, used for configuration tables (`[scorer.<ClassName>]`).
    fn class_name(&self) -> &'static str;

    /// Name of the score, shown to the user and on model cards.
    fn score_name(&self) -> String;

    /// Whether runs using this scorer can be reproduced bit-for-bit.
    fn reproducible(&self) -> bool {
        true
    }

    fn init(&mut self, _ctx: &Context) -> Result<()> {
        Ok(())
    }

    fn get_score(&mut self, ctx: &Context) -> Result<Score>;

    /// Score of the unmodified model. Defaults to [`get_score`](Self::get_score).
    fn get_baseline_score(&mut self, ctx: &Context) -> Result<Score> {
        self.get_score(ctx)
    }

    /// Datasets used by this scorer (with pinned commits after `init`).
    fn dataset_specifications(&self) -> Vec<DatasetSpecification> {
        Vec::new()
    }

    /// The scorer's settings, for reproducibility manifests.
    fn settings_json(&self) -> Value;
}

pub trait Modifier: Send {
    /// The class name, used for configuration tables (`[modifier.<ClassName>]`).
    fn class_name(&self) -> &'static str;

    /// Name of the modification technique, shown to the user and on model cards.
    fn modifier_name(&self) -> String;

    fn reproducible(&self) -> bool {
        true
    }

    /// One-time setup (e.g. computing refusal directions). Must enable LoRA adapters.
    fn init(&mut self, ctx: &Context) -> Result<()>;

    /// Samples parameters for a trial and returns them as JSON.
    fn suggest_parameters(&self, ctx: &Context, trial: &mut Trial) -> Result<Value>;

    /// Modifies the model (through its adapters) according to the parameters.
    fn modify_model(&self, ctx: &Context, parameters: &Value) -> Result<()>;

    /// Undoes the changes made by `modify_model`.
    fn reset_model(&self, ctx: &Context) -> Result<()> {
        ctx.model.reset_adapters();
        Ok(())
    }

    /// Parameters formatted for presentation, in a stable order.
    fn present_parameters(&self, parameters: &Value) -> Result<Vec<(String, String)>>;

    fn dataset_specifications(&self) -> Vec<DatasetSpecification> {
        Vec::new()
    }

    fn settings_json(&self) -> Value;
}

/// Resolves a plugin reference to the class name of a built-in plugin.
pub fn canonical_plugin_name(name: &str) -> Result<String> {
    if name.ends_with(".py") || name.contains(".py:") {
        bail!(
            "Plugin \"{name}\" refers to a Python file. Python plugins are not supported in this version of Heretic; \
             use one of the built-in plugins."
        );
    }
    Ok(name.rsplit('.').next().unwrap_or(name).to_string())
}

pub const BUILTIN_SCORERS: &[&str] = &["KeywordRate", "KLDivergence", "ResponseLength"];
pub const BUILTIN_MODIFIERS: &[&str] = &["ARA", "Abliteration"];

/// Whether a plugin reference resolves to a built-in plugin (always true for valid references).
pub fn is_builtin_plugin(name: &str) -> bool {
    canonical_plugin_name(name)
        .is_ok_and(|n| BUILTIN_SCORERS.contains(&n.as_str()) || BUILTIN_MODIFIERS.contains(&n.as_str()))
}

/// Creates a scorer from its configuration.
pub fn create_scorer(config: &ScorerConfig, settings: &Settings) -> Result<Box<dyn Scorer>> {
    let name = canonical_plugin_name(&config.plugin)?;
    let instance = config.instance_name.as_deref();
    Ok(match name.as_str() {
        "KeywordRate" => Box::new(scorers::keyword_rate::KeywordRate::new(
            settings.plugin_settings("scorer", &name, instance)?,
        )?),
        "KLDivergence" => Box::new(scorers::kl_divergence::KlDivergence::new(
            settings.plugin_settings("scorer", &name, instance)?,
        )),
        "ResponseLength" => Box::new(scorers::response_length::ResponseLength::new(
            settings.plugin_settings("scorer", &name, instance)?,
        )),
        "BERTClassifier" | "BenchmarkScore" => {
            bail!("The {name} scorer is not available in this version of Heretic yet")
        }
        _ => bail!(
            "Unknown scorer \"{}\" (available: {})",
            config.plugin,
            BUILTIN_SCORERS.join(", ")
        ),
    })
}

/// Creates a modifier from its configuration.
pub fn create_modifier(config: &ModifierConfig, settings: &Settings) -> Result<Box<dyn Modifier>> {
    let name = canonical_plugin_name(&config.plugin)?;
    let instance = config.instance_name.as_deref();
    let seed = settings.seed.unwrap_or(0);
    Ok(match name.as_str() {
        "ARA" => Box::new(modifiers::ara::Ara::new(
            settings.plugin_settings("modifier", &name, instance)?,
            seed,
        )),
        "Abliteration" => Box::new(modifiers::abliteration::Abliteration::new(
            settings.plugin_settings("modifier", &name, instance)?,
            seed,
        )),
        _ => bail!(
            "Unknown modifier \"{}\" (available: {})",
            config.plugin,
            BUILTIN_MODIFIERS.join(", ")
        ),
    })
}

/// Formats a prompt/response pair for `print_responses` output.
pub(crate) fn report_response(reporter: &SharedReporter, prompt: &Prompt, response: &str, suffix: &str) {
    use heretic_core::report::ReporterExt;
    reporter.detail("");
    reporter.detail(format!("**System prompt:** {}", prompt.system));
    reporter.detail(format!("**Prompt:** {}", prompt.user));
    let shown = if response.trim().is_empty() {
        "[empty]"
    } else {
        response
    };
    reporter.detail(format!("**Response:** {shown}{suffix}"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plugin_names() {
        assert_eq!(
            canonical_plugin_name("heretic.scorers.keyword_rate.KeywordRate").unwrap(),
            "KeywordRate"
        );
        assert_eq!(canonical_plugin_name("ARA").unwrap(), "ARA");
        assert!(canonical_plugin_name("plugins/my.py:MyScorer").is_err());
        assert!(is_builtin_plugin("heretic.modifiers.ara.ARA"));
        assert!(!is_builtin_plugin("my.module.Custom"));
    }

    #[test]
    fn default_plugins_can_be_created() {
        let settings = Settings::default();
        for config in &settings.scorers {
            create_scorer(config, &settings).unwrap();
        }
        for config in &settings.modifiers {
            create_modifier(config, &settings).unwrap();
        }
    }
}
