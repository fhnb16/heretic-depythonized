// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Combining several goal presets into one run.
//!
//! A run has a single modifier, so the presets are merged like this:
//!
//! * **Modifier:** the "good" and "bad" prompt sets of all presets are concatenated, so the
//!   direction that is removed is the one shared by all unwanted behaviors.
//! * **Scorers:** every goal keeps its own measurement. A scorer type used by several presets
//!   (e.g. KeywordRate for refusals and for slop) becomes one instance per preset, named after
//!   the preset; the multi-objective search optimizes all of them together.
//! * **KL divergence:** one scorer over the union of all "harmless" prompt sets, so damage is
//!   measured on everything the presets care about.
//! * A preset's global system prompt is attached to its own prompt sets, so it keeps applying
//!   to them only.

use anyhow::{Result, bail};
use heretic_core::Settings;
use heretic_core::config::{ModifierConfig, ScorerConfig};
use heretic_core::dataset::{DatasetSpecification, SingleDatasetSpecification};
use heretic_core::toml;
use serde::Serialize;

use crate::{canonical_plugin_name, modifiers, scorers};

/// A preset: its ID (used to name scorer instances) and its full settings.
pub struct Preset {
    pub id: String,
    pub settings: Settings,
}

fn to_toml<T: Serialize>(value: &T) -> Result<toml::Value> {
    Ok(toml::Value::try_from(value)?)
}

/// Copies a preset's global system prompt into prompt sets that don't set their own.
fn with_system_prompt(spec: &DatasetSpecification, settings: &Settings) -> Vec<SingleDatasetSpecification> {
    spec.specifications()
        .into_iter()
        .map(|single| {
            let mut single = single.clone();
            if single.system_prompt.is_none() {
                single.system_prompt = Some(settings.system_prompt.clone());
            }
            single
        })
        .collect()
}

fn push_unique(target: &mut Vec<SingleDatasetSpecification>, items: Vec<SingleDatasetSpecification>) {
    for item in items {
        if !target.contains(&item) {
            target.push(item);
        }
    }
}

fn as_specification(mut specs: Vec<SingleDatasetSpecification>) -> DatasetSpecification {
    if specs.len() == 1 {
        DatasetSpecification::Single(specs.remove(0))
    } else {
        DatasetSpecification::Multiple(specs)
    }
}

/// Instance names may not contain dots or whitespace.
fn instance_name(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Inserts `[<top>.<key>]` into the settings' plugin tables.
fn set_section(extra: &mut toml::Table, top: &str, key: &str, value: toml::Value) {
    let namespace = extra
        .entry(top.to_string())
        .or_insert_with(|| toml::Value::Table(toml::Table::new()));
    if let toml::Value::Table(table) = namespace {
        table.insert(key.to_string(), value);
    }
}

/// The resolved settings of a scorer as a TOML table (all defaults filled in).
fn scorer_section(settings: &Settings, class: &str, config: &ScorerConfig) -> Result<toml::Value> {
    let instance = config.instance_name.as_deref();
    Ok(match class {
        "KeywordRate" => {
            let mut resolved: scorers::keyword_rate::Settings = settings.plugin_settings("scorer", class, instance)?;
            resolved.prompts = as_specification(with_system_prompt(&resolved.prompts, settings));
            to_toml(&resolved)?
        }
        "ResponseLength" => {
            let mut resolved: scorers::response_length::Settings =
                settings.plugin_settings("scorer", class, instance)?;
            resolved.prompts = as_specification(with_system_prompt(&resolved.prompts, settings));
            to_toml(&resolved)?
        }
        // Scorers without typed settings here: copy their raw tables.
        _ => {
            let mut table = settings.plugin_namespace(&format!("scorer.{class}"))?;
            if let Some(instance) = instance {
                for (key, value) in settings.plugin_namespace(&format!("scorer.{class}_{instance}"))? {
                    table.insert(key, value);
                }
            }
            toml::Value::Table(table)
        }
    })
}

/// The good and bad prompt sets of a preset's modifier, with defaults filled in.
fn modifier_prompts(settings: &Settings) -> Result<(Vec<SingleDatasetSpecification>, Vec<SingleDatasetSpecification>)> {
    let config = settings
        .modifiers
        .first()
        .ok_or_else(|| anyhow::anyhow!("a preset has no modifier"))?;
    let class = canonical_plugin_name(&config.plugin)?;
    let instance = config.instance_name.as_deref();
    let (good, bad) = match class.as_str() {
        "ARA" => {
            let s: modifiers::ara::Settings = settings.plugin_settings("modifier", &class, instance)?;
            (s.good_prompts, s.bad_prompts)
        }
        "Abliteration" => {
            let s: modifiers::abliteration::Settings = settings.plugin_settings("modifier", &class, instance)?;
            (s.good_prompts, s.bad_prompts)
        }
        other => bail!("Unknown modifier \"{other}\""),
    };
    Ok((with_system_prompt(&good, settings), with_system_prompt(&bad, settings)))
}

/// Merges several presets into the settings of one run (see the module documentation).
pub fn combine_presets(presets: &[Preset]) -> Result<Settings> {
    let Some(first) = presets.first() else {
        bail!("No presets to combine");
    };
    if presets.len() == 1 {
        return Ok(first.settings.clone());
    }

    let mut combined = first.settings.clone();
    combined.extra.remove("scorer");
    combined.extra.remove("modifier");

    // Scorers.
    let class_users = |class: &str| -> usize {
        presets
            .iter()
            .filter(|p| {
                p.settings
                    .scorers
                    .iter()
                    .any(|s| canonical_plugin_name(&s.plugin).is_ok_and(|c| c == class))
            })
            .count()
    };

    let mut scorer_configs: Vec<ScorerConfig> = Vec::new();
    let mut kl_prompts: Vec<SingleDatasetSpecification> = Vec::new();
    let mut kl_settings: Option<scorers::kl_divergence::Settings> = None;
    let mut kl_config: Option<ScorerConfig> = None;

    for preset in presets {
        for config in &preset.settings.scorers {
            let class = canonical_plugin_name(&config.plugin)?;
            if class == "KLDivergence" {
                let resolved: scorers::kl_divergence::Settings =
                    preset
                        .settings
                        .plugin_settings("scorer", &class, config.instance_name.as_deref())?;
                push_unique(&mut kl_prompts, with_system_prompt(&resolved.prompts, &preset.settings));
                kl_settings.get_or_insert(resolved);
                kl_config.get_or_insert_with(|| ScorerConfig {
                    plugin: class.clone(),
                    optimization: config.optimization,
                    instance_name: None,
                });
                continue;
            }

            let section = scorer_section(&preset.settings, &class, config)?;
            let shared = class_users(&class) > 1;
            let instance = if shared {
                Some(match &config.instance_name {
                    Some(own) => format!("{}-{}", instance_name(&preset.id), own),
                    None => instance_name(&preset.id),
                })
            } else {
                config.instance_name.clone()
            };
            let key = match &instance {
                Some(instance) => format!("{class}_{instance}"),
                None => class.clone(),
            };
            set_section(&mut combined.extra, "scorer", &key, section);
            scorer_configs.push(ScorerConfig {
                plugin: class,
                optimization: config.optimization,
                instance_name: instance,
            });
        }
    }

    if let (Some(mut settings), Some(config)) = (kl_settings, kl_config) {
        settings.prompts = as_specification(kl_prompts);
        set_section(&mut combined.extra, "scorer", "KLDivergence", to_toml(&settings)?);
        scorer_configs.push(config);
    }
    combined.scorers = scorer_configs;

    // Modifier: the method of the first preset, prompt sets of all presets. Both methods get the
    // combined prompt sets, so switching the method later keeps them.
    let mut good = Vec::new();
    let mut bad = Vec::new();
    for preset in presets {
        let (g, b) = modifier_prompts(&preset.settings)?;
        push_unique(&mut good, g);
        push_unique(&mut bad, b);
    }
    let mut ara: modifiers::ara::Settings = first.settings.plugin_settings("modifier", "ARA", None)?;
    ara.good_prompts = as_specification(good.clone());
    ara.bad_prompts = as_specification(bad.clone());
    set_section(&mut combined.extra, "modifier", "ARA", to_toml(&ara)?);
    let mut abliteration: modifiers::abliteration::Settings =
        first.settings.plugin_settings("modifier", "Abliteration", None)?;
    abliteration.good_prompts = as_specification(good);
    abliteration.bad_prompts = as_specification(bad);
    set_section(&mut combined.extra, "modifier", "Abliteration", to_toml(&abliteration)?);
    let method = canonical_plugin_name(&first.settings.modifiers[0].plugin)?;
    combined.modifiers = vec![ModifierConfig {
        plugin: method,
        instance_name: None,
    }];

    // Everything else.
    combined.max_response_length = presets
        .iter()
        .map(|p| p.settings.max_response_length)
        .max()
        .unwrap_or(combined.max_response_length);
    let mut types: Vec<String> = Vec::new();
    let mut tags: Vec<String> = Vec::new();
    for preset in presets {
        if !types.contains(&preset.settings.modified_model_type) {
            types.push(preset.settings.modified_model_type.clone());
        }
        for tag in &preset.settings.upload_tags {
            if !tags.contains(tag) {
                tags.push(tag.clone());
            }
        }
    }
    combined.modified_model_type = types.join(", ");
    combined.upload_tags = tags;

    Ok(combined)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preset(id: &str, toml_text: &str) -> Preset {
        let table: toml::Table = toml::from_str(toml_text).unwrap();
        Preset {
            id: id.into(),
            settings: Settings::from_table(table).unwrap(),
        }
    }

    #[test]
    fn combines_refusals_and_slop() {
        let default = preset("default", "");
        let noslop = preset(
            "noslop",
            r#"
            max_response_length = 300
            modified_model_type = "slop-reduced"
            upload_tags = ["no-slop"]
            system_prompt = "You are a professional writer."
            [scorer.KeywordRate]
            score_name = "Responses with slop"
            keyword_markers = ["tapestry"]
            [scorer.KeywordRate.prompts]
            dataset = "llm-aes/writing-prompts"
            split = "train[1000:1100]"
            column = "prompt"
            [scorer.KLDivergence.prompts]
            dataset = "llm-aes/writing-prompts"
            split = "train[1000:1100]"
            column = "prompt"
            [modifier.ARA.good_prompts]
            dataset = "llm-aes/writing-prompts"
            split = "train[:500]"
            column = "prompt"
            [modifier.ARA.bad_prompts]
            dataset = "llm-aes/writing-prompts"
            split = "train[:500]"
            column = "prompt"
            prefix = "Use purple prose."
            "#,
        );

        let combined = combine_presets(&[default, noslop]).unwrap();

        // Two keyword scorers (one per goal) and one KL scorer.
        let names: Vec<(String, Option<String>)> = combined
            .scorers
            .iter()
            .map(|s| (s.plugin.clone(), s.instance_name.clone()))
            .collect();
        assert_eq!(
            names,
            vec![
                ("KeywordRate".into(), Some("default".into())),
                ("KeywordRate".into(), Some("noslop".into())),
                ("KLDivergence".into(), None),
            ]
        );

        let refusals: scorers::keyword_rate::Settings = combined
            .plugin_settings("scorer", "KeywordRate", Some("default"))
            .unwrap();
        assert_eq!(refusals.score_name, "Refusals");
        let slop: scorers::keyword_rate::Settings = combined
            .plugin_settings("scorer", "KeywordRate", Some("noslop"))
            .unwrap();
        assert_eq!(slop.score_name, "Responses with slop");
        assert_eq!(slop.keyword_markers, vec!["tapestry".to_string()]);
        // The writer system prompt stays with the slop prompts only.
        assert_eq!(
            slop.prompts.specifications()[0].system_prompt.as_deref(),
            Some("You are a professional writer.")
        );

        let kl: scorers::kl_divergence::Settings = combined.plugin_settings("scorer", "KLDivergence", None).unwrap();
        assert_eq!(kl.prompts.specifications().len(), 2);

        let ara: modifiers::ara::Settings = combined.plugin_settings("modifier", "ARA", None).unwrap();
        assert_eq!(ara.good_prompts.specifications().len(), 2);
        assert_eq!(ara.bad_prompts.specifications().len(), 2);
        let abliteration: modifiers::abliteration::Settings =
            combined.plugin_settings("modifier", "Abliteration", None).unwrap();
        assert_eq!(abliteration.bad_prompts.specifications().len(), 2);

        assert_eq!(combined.max_response_length, 300);
        assert!(combined.modified_model_type.contains("slop-reduced"));
        assert!(combined.upload_tags.contains(&"no-slop".to_string()));

        // The result is a valid configuration that survives a TOML round trip.
        let text = toml::to_string(&combined).unwrap();
        let reparsed = Settings::from_table(toml::from_str(&text).unwrap()).unwrap();
        assert_eq!(reparsed.scorers.len(), 3);
    }

    #[test]
    fn single_preset_is_unchanged() {
        let default = preset("default", "");
        let scorers = default.settings.scorers.clone();
        let combined = combine_presets(&[default]).unwrap();
        assert_eq!(combined.scorers, scorers);
    }
}
