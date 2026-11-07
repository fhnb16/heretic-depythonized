// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Model cards (README.md) for exported models.

use serde::Deserialize;
use yaml_rust2::{Yaml, YamlEmitter, YamlLoader};

/// A score with its baseline, as stored in trial records.
#[derive(Debug, Clone, Deserialize)]
pub struct CardScore {
    pub name: String,
    pub score: CardValue,
    pub baseline: CardValue,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CardValue {
    pub md_display: String,
}

pub struct CardInfo<'a> {
    /// Model ID, if it is a Hugging Face repository (local paths are hidden for privacy).
    pub hub_model_id: Option<&'a str>,
    pub modified_model_type: &'a str,
    pub modifier_name: &'a str,
    pub parameters: &'a [(String, String)],
    pub scores: &'a [CardScore],
    pub reproducible: bool,
}

/// The introduction prepended to the original model card.
pub fn readme_intro(info: &CardInfo) -> String {
    let model_link = match info.hub_model_id {
        Some(id) => format!("[{id}](https://huggingface.co/{id})"),
        None => "a model".to_string(),
    };

    let reproducibility = if info.reproducible {
        "\n> [!TIP]\n> **This model is reproducible!**\n>\n> See the [README](reproduce/README.md) in the `reproduce` directory for more information.\n"
    } else {
        ""
    };

    let parameter_rows: Vec<String> = info
        .parameters
        .iter()
        .map(|(name, value)| format!("| **{name}** | {value} |"))
        .collect();

    let score_rows: Vec<String> = info
        .scores
        .iter()
        .map(|s| {
            format!(
                "| **{}** | {} | {} |",
                s.name, s.score.md_display, s.baseline.md_display
            )
        })
        .collect();

    format!(
        "# This is a {kind} version of {model_link}, made using [Heretic](https://heretic-project.org) v{version}\n\
         {reproducibility}\n\
         ## {modifier} parameters\n\n\
         | Parameter | Value |\n\
         | :-------- | :---: |\n\
         {parameters}\n\n\
         ## Performance\n\n\
         | Metric | This model | Original model ({model_link}) |\n\
         | :----- | :--------: | :---------------------------: |\n\
         {scores}\n\n\
         -----\n\n",
        kind = info.modified_model_type,
        version = env!("CARGO_PKG_VERSION"),
        modifier = info.modifier_name,
        parameters = parameter_rows.join("\n"),
        scores = score_rows.join("\n"),
    )
}

/// Splits a README into its YAML front matter and the remaining text.
fn split_front_matter(readme: &str) -> (Option<&str>, &str) {
    if let Some(rest) = readme.strip_prefix("---\n").or_else(|| readme.strip_prefix("---\r\n"))
        && let Some(end) = rest.find("\n---")
    {
        let yaml = &rest[..end];
        let after = &rest[end + 4..];
        let after = after.strip_prefix('\r').unwrap_or(after);
        let after = after.strip_prefix('\n').unwrap_or(after);
        return (Some(yaml), after);
    }
    (None, readme)
}

/// Builds the model card: the original card with Heretic's introduction prepended and
/// its metadata updated (tags, base model).
pub fn build_model_card(original: &str, intro: &str, tags: &[String], base_model: Option<&str>) -> String {
    let (yaml, text) = split_front_matter(original);
    let text = text.trim_start_matches(['\r', '\n']);

    let mut data = yaml
        .and_then(|y| YamlLoader::load_from_str(y).ok())
        .and_then(|docs| docs.into_iter().next())
        .filter(|d| matches!(d, Yaml::Hash(_)))
        .unwrap_or_else(|| Yaml::Hash(Default::default()));

    if let Yaml::Hash(map) = &mut data {
        let tags_key = Yaml::String("tags".into());
        let mut existing: Vec<Yaml> = match map.get(&tags_key) {
            Some(Yaml::Array(items)) => items.clone(),
            Some(Yaml::String(tag)) => vec![Yaml::String(tag.clone())],
            _ => Vec::new(),
        };
        for tag in tags {
            let value = Yaml::String(tag.clone());
            if !existing.contains(&value) {
                existing.push(value);
            }
        }
        map.insert(tags_key, Yaml::Array(existing));

        if let Some(base_model) = base_model {
            map.insert(Yaml::String("base_model".into()), Yaml::String(base_model.into()));
        }
    }

    let mut yaml_text = String::new();
    {
        let mut emitter = YamlEmitter::new(&mut yaml_text);
        let _ = emitter.dump(&data);
    }
    // The emitter writes a leading "---" document marker.
    let yaml_text = yaml_text.trim_start_matches("---").trim_start_matches('\n');

    format!("---\n{yaml_text}\n---\n\n{intro}{text}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intro_contains_tables() {
        let parameters = vec![("neighbor_count".to_string(), "5".to_string())];
        let scores = vec![CardScore {
            name: "Refusals".into(),
            score: CardValue {
                md_display: "3/100".into(),
            },
            baseline: CardValue {
                md_display: "97/100".into(),
            },
        }];
        let intro = readme_intro(&CardInfo {
            hub_model_id: Some("Qwen/Qwen3-4B"),
            modified_model_type: "decensored",
            modifier_name: "Arbitrary-Rank Ablation (ARA)",
            parameters: &parameters,
            scores: &scores,
            reproducible: false,
        });
        assert!(
            intro
                .starts_with("# This is a decensored version of [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)")
        );
        assert!(intro.contains("| **neighbor_count** | 5 |"));
        assert!(intro.contains("| **Refusals** | 3/100 | 97/100 |"));
        assert!(!intro.contains("reproducible!"));
    }

    #[test]
    fn card_metadata_is_updated() {
        let original = "---\nlicense: apache-2.0\ntags:\n- chat\n---\n\n# Original\nText\n";
        let card = build_model_card(
            original,
            "# Intro\n\n",
            &["heretic".into(), "chat".into(), "uncensored".into()],
            Some("org/model"),
        );
        assert!(card.starts_with("---\n"));
        assert!(card.contains("license: apache-2.0"));
        assert!(card.contains("- chat\n  - heretic") || card.contains("- chat\n- heretic") || card.contains("heretic"));
        assert_eq!(card.matches("chat").count(), 1);
        assert!(card.contains("base_model: org/model"));
        assert!(card.contains("# Intro\n\n# Original\nText"));
    }

    #[test]
    fn card_without_front_matter() {
        let card = build_model_card("# Just text\n", "# Intro\n\n", &["heretic".into()], None);
        assert!(card.contains("tags:"));
        assert!(card.ends_with("# Intro\n\n# Just text\n"));
    }
}
