// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Dataset specifications and prompt loading.
//!
//! A "dataset" can be a Hugging Face dataset ID, a path to a dataset directory on
//! disk, or a path to a plain text file with one prompt per line. Supported data
//! file formats are Parquet, JSON Lines, JSON and CSV.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Context as _, Result, anyhow, bail};
use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::hub::{self, RepoType, ResolvedRepo};
use crate::prompt::Prompt;

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
pub struct SingleDatasetSpecification {
    /// Hugging Face dataset ID, or path to dataset on disk.
    pub dataset: String,

    /// Hugging Face commit hash of the dataset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,

    /// Dataset config/subset name. Each config can have its own split.
    /// Used to load a specific config of a dataset that has multiple configurations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<String>,

    /// Portion of the dataset to use. Required for datasets, optional for plain text files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<String>,

    /// Column in the dataset that contains the prompts.
    /// Required for datasets, ignored for plain text files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub column: Option<String>,

    /// Text to prepend to each prompt.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub prefix: String,

    /// Text to append to each prompt.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub suffix: String,

    /// System prompt to use with the prompts (overrides global system prompt if set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
}

/// One dataset, or a list of datasets whose prompts are concatenated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum DatasetSpecification {
    Single(SingleDatasetSpecification),
    Multiple(Vec<SingleDatasetSpecification>),
}

impl DatasetSpecification {
    pub fn specifications(&self) -> Vec<&SingleDatasetSpecification> {
        match self {
            DatasetSpecification::Single(spec) => vec![spec],
            DatasetSpecification::Multiple(specs) => specs.iter().collect(),
        }
    }

    pub fn specifications_mut(&mut self) -> Vec<&mut SingleDatasetSpecification> {
        match self {
            DatasetSpecification::Single(spec) => vec![spec],
            DatasetSpecification::Multiple(specs) => specs.iter_mut().collect(),
        }
    }

    /// Human-readable description, e.g. "mlabonne/harmful_behaviors (test[:100])".
    pub fn describe(&self) -> String {
        self.specifications()
            .iter()
            .map(|spec| spec.describe())
            .collect::<Vec<_>>()
            .join(" + ")
    }

    /// Whether every dataset is pinned to a Hub commit (required for reproducibility).
    pub fn is_reproducible(&self) -> bool {
        self.specifications()
            .iter()
            .all(|spec| hub::is_hf_path(&spec.dataset) && spec.commit.is_some())
    }
}

impl SingleDatasetSpecification {
    pub fn describe(&self) -> String {
        let mut text = self.dataset.clone();
        if let Some(config) = &self.config {
            text.push_str(&format!(" [{config}]"));
        }
        if let Some(split) = &self.split {
            text.push_str(&format!(" ({split})"));
        }
        text
    }
}

/// A split instruction such as `train[10:20]` or `test[:50%]`, resolved to absolute indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitSlice {
    pub split: String,
    pub from: usize,
    pub to: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rounding {
    Closest,
    Pct1DropRemainder,
}

/// Parses a split specification and resolves it to absolute indices, following the
/// semantics of `datasets.ReadInstruction` (including Python's banker's rounding for
/// percentages). Only the first instruction of a `+`-joined specification is used.
pub fn parse_split(spec: &str, num_examples: usize) -> Result<SplitSlice> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(
            r"^(?P<split>\w+(?:\.\w+)*)(?:\[(?P<from>-?\d+%?)?:(?P<to>-?\d+%?)?\])?(?:\((?P<rounding>[^)]*)\))?$",
        )
        .unwrap()
    });

    let first = spec.split('+').next().unwrap_or_default().trim();
    let captures = re
        .captures(first)
        .ok_or_else(|| anyhow!("Unrecognized split format: {spec:?}"))?;

    let split = captures["split"].to_string();

    let rounding = match captures.name("rounding").map(|m| m.as_str()) {
        None | Some("closest") => Rounding::Closest,
        Some("pct1_dropremainder") => Rounding::Pct1DropRemainder,
        Some(other) => bail!("Unknown rounding {other:?} in split {spec:?}"),
    };

    let parse_bound = |name: &str| -> Result<Option<(i64, bool)>> {
        match captures.name(name) {
            None => Ok(None),
            Some(m) => {
                let text = m.as_str();
                let percent = text.ends_with('%');
                let value: i64 = text.trim_end_matches('%').parse()?;
                if percent && !(-100..=100).contains(&value) {
                    bail!("Percent slice boundaries must be > -100 and < 100: {spec:?}");
                }
                Ok(Some((value, percent)))
            }
        }
    };

    let from = parse_bound("from")?;
    let to = parse_bound("to")?;

    let percent = match (from, to) {
        (Some((_, a)), Some((_, b))) if a != b => bail!("'from' and 'to' must use the same unit: {spec:?}"),
        (Some((_, p)), _) | (None, Some((_, p))) => p,
        (None, None) => false,
    };

    let n = num_examples as i64;
    let pct_to_abs = |boundary: i64| -> Result<i64> {
        match rounding {
            Rounding::Closest => Ok((boundary as f64 * n as f64 / 100.0).round_ties_even() as i64),
            Rounding::Pct1DropRemainder => {
                if num_examples < 100 {
                    bail!("Using \"pct1_dropremainder\" rounding on a split with less than 100 elements is forbidden");
                }
                Ok(boundary * (n / 100))
            }
        }
    };

    let mut from_abs = match from {
        None => 0,
        Some((value, true)) => pct_to_abs(value)?,
        Some((value, false)) => value,
    };
    let mut to_abs = match to {
        None => n,
        Some((value, true)) => pct_to_abs(value)?,
        Some((value, false)) => value,
    };
    let _ = percent;

    if from_abs < 0 {
        from_abs = (n + from_abs).max(0);
    }
    if to_abs < 0 {
        to_abs = (n + to_abs).max(0);
    }
    let from_abs = from_abs.min(n) as usize;
    let to_abs = to_abs.min(n) as usize;

    Ok(SplitSlice {
        split,
        from: from_abs,
        to: to_abs,
    })
}

/// The split name of a specification (the part before `[`).
pub fn split_name(spec: &str) -> &str {
    spec.split(['[', '(', '+']).next().unwrap_or(spec).trim()
}

/// Loads the prompts described by a dataset specification. If a Hub dataset isn't
/// pinned to a commit yet, the latest commit is recorded in the specification.
pub fn load_prompts(specification: &mut DatasetSpecification, default_system_prompt: &str) -> Result<Vec<Prompt>> {
    let mut prompts = Vec::new();
    for spec in specification.specifications_mut() {
        prompts.extend(load_prompts_single(spec, default_system_prompt)?);
    }
    Ok(prompts)
}

fn load_prompts_single(spec: &mut SingleDatasetSpecification, default_system_prompt: &str) -> Result<Vec<Prompt>> {
    let path = Path::new(&spec.dataset);

    let mut texts: Vec<String> = if path.is_file() && !is_data_file(&spec.dataset) {
        // Plain text file with one prompt per line. Empty lines are ignored.
        let content = std::fs::read_to_string(path).with_context(|| format!("Failed to read {}", path.display()))?;
        let lines: Vec<String> = content
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(String::from)
            .collect();
        match &spec.split {
            // A synthetic split name is prepended because the split syntax expects a named split.
            Some(split) => {
                let slice = parse_split(&format!("_{split}"), lines.len())?;
                lines[slice.from..slice.to].to_vec()
            }
            None => lines,
        }
    } else {
        let split = spec
            .split
            .clone()
            .ok_or_else(|| anyhow!("The \"split\" field is required for datasets: {}", spec.dataset))?;
        let column = spec
            .column
            .clone()
            .ok_or_else(|| anyhow!("The \"column\" field is required for datasets: {}", spec.dataset))?;

        if path.is_file() {
            // A single data file on disk.
            let rows = read_column(path, &column)?;
            let slice = parse_split(&split, rows.len())?;
            rows[slice.from..slice.to].to_vec()
        } else {
            let repo = if path.is_dir() {
                ResolvedRepo::open(RepoType::Dataset, &spec.dataset, None)?
            } else {
                // Pin to the latest commit if not already set, so the exact dataset
                // version is recorded for reproducibility.
                let repo = ResolvedRepo::open(RepoType::Dataset, &spec.dataset, spec.commit.as_deref())?;
                if spec.commit.is_none() {
                    spec.commit = repo.commit.clone();
                }
                repo
            };
            load_repo_split(&repo, spec.config.as_deref(), &split, &column)?
        }
    };

    if !spec.prefix.is_empty() {
        texts = texts.into_iter().map(|t| format!("{} {t}", spec.prefix)).collect();
    }
    if !spec.suffix.is_empty() {
        texts = texts.into_iter().map(|t| format!("{t} {}", spec.suffix)).collect();
    }

    let system = spec
        .system_prompt
        .clone()
        .unwrap_or_else(|| default_system_prompt.to_string());

    Ok(texts
        .into_iter()
        .map(|user| Prompt {
            system: system.clone(),
            user,
        })
        .collect())
}

const DATA_EXTENSIONS: &[&str] = &["parquet", "jsonl", "json", "csv"];

fn is_data_file(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    DATA_EXTENSIONS.iter().any(|ext| lower.ends_with(&format!(".{ext}")))
        || lower.ends_with(".jsonl.gz")
        || lower.ends_with(".json.gz")
}

/// Loads a split of a dataset repository (or local dataset directory) and returns
/// the values of one column.
pub fn load_repo_split(
    repo: &ResolvedRepo,
    config: Option<&str>,
    split_spec: &str,
    column: &str,
) -> Result<Vec<String>> {
    let split = split_name(split_spec);
    let (source, files) = resolve_split_files(repo, config, split)?;

    let mut rows = Vec::new();
    for file in &files {
        let path = source.get(file, None)?;
        rows.extend(read_column(&path, column).with_context(|| format!("Failed to read {file}"))?);
    }

    let slice = parse_split(split_spec, rows.len())?;
    Ok(rows[slice.from..slice.to].to_vec())
}

/// Determines the data files of a split, using the `configs` section of the dataset
/// card if present, then the default file naming conventions of the `datasets` library,
/// then the automatic Parquet conversion of the Hub. Returns the repository revision
/// that contains the files (the conversion lives in a separate revision) and the files.
pub fn resolve_split_files(
    repo: &ResolvedRepo,
    config: Option<&str>,
    split: &str,
) -> Result<(ResolvedRepo, Vec<String>)> {
    // 1. Explicit configuration in the dataset card.
    if let Some(readme) = repo.get_optional("README.md").ok().flatten()
        && let Ok(text) = std::fs::read_to_string(&readme)
        && let Some(configs) = parse_card_configs(&text)
        && !configs.is_empty()
    {
        let chosen = match config {
            Some(name) => configs.iter().find(|c| c.name == name),
            None => configs
                .iter()
                .find(|c| c.default)
                .or_else(|| configs.iter().find(|c| c.name == "default"))
                .or_else(|| (configs.len() == 1).then(|| &configs[0])),
        };

        if let Some(chosen) = chosen {
            let patterns = chosen.data_files.get(split).ok_or_else(|| {
                anyhow!(
                    "Split {split:?} not found in config {:?} of {} (available: {})",
                    chosen.name,
                    repo.id,
                    chosen.data_files.keys().cloned().collect::<Vec<_>>().join(", ")
                )
            })?;
            let files = match_patterns(&repo.files, patterns)?;
            if files.is_empty() {
                bail!("No data files match {patterns:?} in {}", repo.id);
            }
            return Ok((repo.clone(), files));
        } else if let Some(name) = config {
            bail!(
                "Config {name:?} not found in {} (available: {})",
                repo.id,
                configs.iter().map(|c| c.name.clone()).collect::<Vec<_>>().join(", ")
            );
        }
    }

    // 2. Conventional file names, optionally within a config subdirectory.
    let candidates: Vec<&String> = repo
        .files
        .iter()
        .filter(|f| is_data_file(f))
        .filter(|f| match config {
            Some(name) => f.starts_with(&format!("{name}/")) || f.contains(&format!("/{name}/")),
            None => true,
        })
        .collect();

    if !candidates.is_empty() {
        let mut by_split: BTreeMap<&'static str, Vec<String>> = BTreeMap::new();
        for file in &candidates {
            if let Some(canonical) = detect_split(file) {
                by_split.entry(canonical).or_default().push((*file).clone());
            }
        }

        let canonical = canonical_split(split);
        let files = if by_split.is_empty() {
            // No split keywords at all: everything belongs to "train".
            if canonical == "train" {
                candidates.iter().map(|f| (*f).clone()).collect()
            } else {
                Vec::new()
            }
        } else {
            by_split.get(canonical).cloned().unwrap_or_default()
        };

        if !files.is_empty() {
            return Ok((repo.clone(), prefer_one_format(files)));
        }
    }

    // 3. Automatic Parquet conversion (refs/convert/parquet).
    if !repo.is_local() {
        let config_name = config.unwrap_or("default");
        let converted = ResolvedRepo::open(RepoType::Dataset, &repo.id, Some("refs/convert/parquet"))?;
        let prefix = format!("{config_name}/{split}/");
        let files: Vec<String> = converted
            .files
            .iter()
            .filter(|f| f.starts_with(&prefix) && f.ends_with(".parquet"))
            .cloned()
            .collect();
        if !files.is_empty() {
            return Ok((converted, files));
        }
    }

    bail!("Could not find data files for split {split:?} in {}", repo.id)
}

fn prefer_one_format(mut files: Vec<String>) -> Vec<String> {
    // If several formats are present (e.g. CSV and Parquet), use only the first one by preference.
    for ext in DATA_EXTENSIONS {
        let matching: Vec<String> = files
            .iter()
            .filter(|f| f.to_ascii_lowercase().ends_with(&format!(".{ext}")))
            .cloned()
            .collect();
        if !matching.is_empty() {
            return matching;
        }
    }
    files.sort();
    files
}

fn canonical_split(split: &str) -> &str {
    match split {
        "training" => "train",
        "valid" | "dev" | "val" => "validation",
        "testing" | "eval" | "evaluation" => "test",
        other => other,
    }
}

/// Detects the split of a data file from keywords in its path, like `datasets`
/// does with its default patterns (keywords delimited by `-._ 0-9/`).
fn detect_split(path: &str) -> Option<&'static str> {
    let lower = path.to_ascii_lowercase();
    let tokens: Vec<&str> = lower
        .split(|c: char| matches!(c, '-' | '.' | '_' | ' ' | '/') || c.is_ascii_digit())
        .filter(|t| !t.is_empty())
        .collect();

    let keywords: &[(&str, &[&str])] = &[
        ("train", &["train", "training"]),
        ("validation", &["validation", "valid", "dev", "val"]),
        ("test", &["test", "testing", "eval", "evaluation"]),
    ];

    // The datasets library checks test/validation keywords before train is assigned to
    // the remaining files; any keyword match counts.
    for (canonical, words) in keywords.iter().rev() {
        if tokens.iter().any(|t| words.contains(t)) {
            return Some(canonical);
        }
    }
    None
}

fn match_patterns(files: &[String], patterns: &[String]) -> Result<Vec<String>> {
    let mut builder = globset::GlobSetBuilder::new();
    for pattern in patterns {
        builder.add(
            globset::GlobBuilder::new(pattern)
                .literal_separator(true)
                .build()
                .with_context(|| format!("Invalid data file pattern {pattern:?}"))?,
        );
    }
    let set = builder.build()?;
    let mut matched: Vec<String> = files.iter().filter(|f| set.is_match(f.as_str())).cloned().collect();
    matched.sort();
    Ok(matched)
}

#[derive(Debug, Clone)]
struct CardConfig {
    name: String,
    default: bool,
    data_files: BTreeMap<String, Vec<String>>,
}

/// Parses the `configs` section of a dataset card's YAML front matter.
fn parse_card_configs(readme: &str) -> Option<Vec<CardConfig>> {
    let rest = readme.strip_prefix("---")?;
    let end = rest.find("\n---")?;
    let yaml = &rest[..end];

    let docs = yaml_rust2::YamlLoader::load_from_str(yaml).ok()?;
    let doc = docs.first()?;
    let configs = doc["configs"].as_vec()?;

    let mut result = Vec::new();
    for config in configs {
        let name = config["config_name"].as_str().unwrap_or("default").to_string();
        let default = config["default"].as_bool().unwrap_or(false);
        let mut data_files: BTreeMap<String, Vec<String>> = BTreeMap::new();

        let as_patterns = |value: &yaml_rust2::Yaml| -> Vec<String> {
            match value {
                yaml_rust2::Yaml::String(s) => vec![s.clone()],
                yaml_rust2::Yaml::Array(items) => items.iter().filter_map(|i| i.as_str().map(String::from)).collect(),
                _ => vec![],
            }
        };

        match &config["data_files"] {
            yaml_rust2::Yaml::String(pattern) => {
                data_files.insert("train".into(), vec![pattern.clone()]);
            }
            yaml_rust2::Yaml::Array(entries) => {
                for entry in entries {
                    match entry {
                        yaml_rust2::Yaml::String(pattern) => {
                            data_files.entry("train".into()).or_default().push(pattern.clone());
                        }
                        _ => {
                            let split = entry["split"].as_str().unwrap_or("train").to_string();
                            data_files.entry(split).or_default().extend(as_patterns(&entry["path"]));
                        }
                    }
                }
            }
            yaml_rust2::Yaml::Hash(map) => {
                for (split, value) in map {
                    if let Some(split) = split.as_str() {
                        data_files
                            .entry(split.to_string())
                            .or_default()
                            .extend(as_patterns(value));
                    }
                }
            }
            _ => {}
        }

        result.push(CardConfig {
            name,
            default,
            data_files,
        });
    }

    Some(result)
}

/// Reads all values of a column from a data file (Parquet, JSON Lines, JSON, CSV).
pub fn read_column(path: &Path, column: &str) -> Result<Vec<String>> {
    let name = path.to_string_lossy().to_ascii_lowercase();
    if name.ends_with(".parquet") {
        read_parquet_column(path, column)
    } else if name.ends_with(".jsonl") {
        read_jsonl_column(path, column)
    } else if name.ends_with(".json") {
        read_json_column(path, column)
    } else if name.ends_with(".csv") {
        read_csv_column(path, column)
    } else {
        bail!("Unsupported data file format: {}", path.display())
    }
}

/// Reads all rows of a data file as JSON objects.
pub fn read_rows(path: &Path) -> Result<Vec<serde_json::Map<String, serde_json::Value>>> {
    let name = path.to_string_lossy().to_ascii_lowercase();
    if name.ends_with(".parquet") {
        read_parquet_rows(path)
    } else if name.ends_with(".jsonl") {
        let text = std::fs::read_to_string(path)?;
        text.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| match serde_json::from_str(line)? {
                serde_json::Value::Object(map) => Ok(map),
                _ => bail!("JSON Lines rows must be objects"),
            })
            .collect()
    } else if name.ends_with(".json") {
        json_records(&serde_json::from_str(&std::fs::read_to_string(path)?)?)
    } else if name.ends_with(".csv") {
        let mut reader = csv::Reader::from_path(path)?;
        let headers = reader.headers()?.clone();
        let mut rows = Vec::new();
        for record in reader.records() {
            let record = record?;
            rows.push(
                headers
                    .iter()
                    .zip(record.iter())
                    .map(|(h, v)| (h.to_string(), serde_json::Value::String(v.to_string())))
                    .collect(),
            );
        }
        Ok(rows)
    } else {
        bail!("Unsupported data file format: {}", path.display())
    }
}

fn value_to_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn column_from_rows(
    rows: &[serde_json::Map<String, serde_json::Value>],
    column: &str,
    path: &Path,
) -> Result<Vec<String>> {
    rows.iter()
        .map(|row| {
            row.get(column)
                .map(value_to_string)
                .ok_or_else(|| anyhow!("Column {column:?} not found in {}", path.display()))
        })
        .collect()
}

fn read_jsonl_column(path: &Path, column: &str) -> Result<Vec<String>> {
    column_from_rows(&read_rows(path)?, column, path)
}

fn json_records(value: &serde_json::Value) -> Result<Vec<serde_json::Map<String, serde_json::Value>>> {
    match value {
        serde_json::Value::Array(items) => items
            .iter()
            .map(|item| match item {
                serde_json::Value::Object(map) => Ok(map.clone()),
                _ => bail!("JSON array items must be objects"),
            })
            .collect(),
        // Column-oriented ({"col": [...]}) or {"data": [...]} layouts.
        serde_json::Value::Object(map) => {
            if let Some(data) = map.get("data").filter(|d| d.is_array()) {
                return json_records(data);
            }
            let length = map
                .values()
                .filter_map(|v| v.as_array().map(Vec::len))
                .max()
                .unwrap_or(0);
            Ok((0..length)
                .map(|i| {
                    map.iter()
                        .map(|(k, v)| (k.clone(), v.get(i).cloned().unwrap_or(serde_json::Value::Null)))
                        .collect()
                })
                .collect())
        }
        _ => bail!("Unsupported JSON layout"),
    }
}

fn read_json_column(path: &Path, column: &str) -> Result<Vec<String>> {
    column_from_rows(&read_rows(path)?, column, path)
}

fn read_csv_column(path: &Path, column: &str) -> Result<Vec<String>> {
    let mut reader = csv::Reader::from_path(path)?;
    let index = reader
        .headers()?
        .iter()
        .position(|h| h == column)
        .ok_or_else(|| anyhow!("Column {column:?} not found in {}", path.display()))?;
    let mut values = Vec::new();
    for record in reader.records() {
        values.push(record?.get(index).unwrap_or_default().to_string());
    }
    Ok(values)
}

fn parquet_field_to_json(field: &parquet::record::Field) -> serde_json::Value {
    use parquet::record::Field;
    match field {
        Field::Null => serde_json::Value::Null,
        Field::Bool(b) => serde_json::Value::Bool(*b),
        Field::Byte(v) => (*v).into(),
        Field::Short(v) => (*v).into(),
        Field::Int(v) => (*v).into(),
        Field::Long(v) => (*v).into(),
        Field::UByte(v) => (*v).into(),
        Field::UShort(v) => (*v).into(),
        Field::UInt(v) => (*v).into(),
        Field::ULong(v) => (*v).into(),
        Field::Float(v) => serde_json::Number::from_f64(*v as f64).map_or(serde_json::Value::Null, Into::into),
        Field::Double(v) => serde_json::Number::from_f64(*v).map_or(serde_json::Value::Null, Into::into),
        Field::Str(s) => serde_json::Value::String(s.clone()),
        Field::ListInternal(list) => {
            serde_json::Value::Array(list.elements().iter().map(parquet_field_to_json).collect())
        }
        Field::Group(row) => serde_json::Value::Object(
            row.get_column_iter()
                .map(|(name, value)| (name.clone(), parquet_field_to_json(value)))
                .collect(),
        ),
        Field::MapInternal(map) => serde_json::Value::Object(
            map.entries()
                .iter()
                .map(|(k, v)| (value_to_string(&parquet_field_to_json(k)), parquet_field_to_json(v)))
                .collect(),
        ),
        other => serde_json::Value::String(other.to_string()),
    }
}

fn read_parquet_rows(path: &Path) -> Result<Vec<serde_json::Map<String, serde_json::Value>>> {
    use parquet::file::reader::{FileReader, SerializedFileReader};

    let file = std::fs::File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let reader = SerializedFileReader::new(file).with_context(|| format!("Invalid Parquet file {}", path.display()))?;
    let mut rows = Vec::new();
    for row in reader.get_row_iter(None)? {
        let row = row?;
        rows.push(
            row.get_column_iter()
                .map(|(name, value)| (name.clone(), parquet_field_to_json(value)))
                .collect(),
        );
    }
    Ok(rows)
}

fn read_parquet_column(path: &Path, column: &str) -> Result<Vec<String>> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::schema::types::Type;

    let file = std::fs::File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let reader = SerializedFileReader::new(file).with_context(|| format!("Invalid Parquet file {}", path.display()))?;

    // Project to the requested column to avoid decoding everything.
    let schema = reader.metadata().file_metadata().schema();
    let field = schema
        .get_fields()
        .iter()
        .find(|f| f.name() == column)
        .ok_or_else(|| anyhow!("Column {column:?} not found in {}", path.display()))?
        .clone();
    let projection = Type::group_type_builder(schema.name())
        .with_fields(vec![field])
        .build()?;

    let mut values = Vec::new();
    for row in reader.get_row_iter(Some(projection))? {
        let row = row?;
        let value = row
            .get_column_iter()
            .next()
            .map(|(_, value)| value_to_string(&parquet_field_to_json(value)))
            .unwrap_or_default();
        values.push(value);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slice(spec: &str, n: usize) -> (usize, usize) {
        let s = parse_split(spec, n).unwrap();
        (s.from, s.to)
    }

    #[test]
    fn split_slices() {
        assert_eq!(slice("train", 500), (0, 500));
        assert_eq!(slice("train[:400]", 500), (0, 400));
        assert_eq!(slice("train[:400]", 100), (0, 100));
        assert_eq!(slice("test[10:20]", 500), (10, 20));
        assert_eq!(slice("test[-10:]", 500), (490, 500));
        assert_eq!(slice("test[-1000:]", 500), (0, 500));
        assert_eq!(slice("train[:50%]", 9), (0, 4)); // 4.5 rounds to even
        assert_eq!(slice("train[:50%]", 11), (0, 6)); // 5.5 rounds to even
        assert_eq!(slice("train[10%:20%]", 1000), (100, 200));
        assert_eq!(slice("train[:10%](pct1_dropremainder)", 1050), (0, 100));
        assert_eq!(slice("_[:3]", 5), (0, 3));
        assert_eq!(parse_split("train[:400]", 1).unwrap().split, "train");
    }

    #[test]
    fn invalid_splits() {
        assert!(parse_split("train[:10%](pct1_dropremainder)", 50).is_err());
        assert!(parse_split("train[1:20%]", 50).is_err());
        assert!(parse_split("train[:200%]", 50).is_err());
        assert!(parse_split("tr ain", 50).is_err());
    }

    #[test]
    fn split_detection() {
        assert_eq!(detect_split("data/train-00000-of-00001.parquet"), Some("train"));
        assert_eq!(detect_split("data/test-00000-of-00001.parquet"), Some("test"));
        assert_eq!(detect_split("validation.jsonl"), Some("validation"));
        assert_eq!(detect_split("harmful_behaviors.csv"), None);
        assert_eq!(detect_split("contrast/data.parquet"), None);
    }

    #[test]
    fn card_configs() {
        let readme = "---\ndataset_info:\n  features: []\nconfigs:\n- config_name: default\n  data_files:\n  - split: train\n    path: data/train-*\n  - split: test\n    path: data/test-*\n---\n# Card\n";
        let configs = parse_card_configs(readme).unwrap();
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].data_files["test"], ["data/test-*"]);

        let files = vec![
            "README.md".to_string(),
            "data/test-00000-of-00001.parquet".to_string(),
            "data/train-00000-of-00001.parquet".to_string(),
        ];
        assert_eq!(
            match_patterns(&files, &configs[0].data_files["train"]).unwrap(),
            ["data/train-00000-of-00001.parquet"]
        );
    }

    #[test]
    fn text_file_prompts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prompts.txt");
        std::fs::write(&path, "first\n\n  second  \nthird\n").unwrap();

        let mut spec = DatasetSpecification::Single(SingleDatasetSpecification {
            dataset: path.to_string_lossy().into(),
            split: Some("[1:]".into()),
            prefix: "Q:".into(),
            ..Default::default()
        });
        let prompts = load_prompts(&mut spec, "sys").unwrap();
        assert_eq!(
            prompts.iter().map(|p| p.user.as_str()).collect::<Vec<_>>(),
            ["Q: second", "Q: third"]
        );
        assert_eq!(prompts[0].system, "sys");
    }

    #[test]
    fn local_jsonl_and_csv_datasets() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("train.jsonl"),
            "{\"text\": \"a\"}\n{\"text\": \"b\"}\n{\"text\": \"c\"}\n",
        )
        .unwrap();
        std::fs::write(dir.path().join("test.csv"), "text,other\nx,1\ny,2\n").unwrap();

        let mut spec = DatasetSpecification::Single(SingleDatasetSpecification {
            dataset: dir.path().to_string_lossy().into(),
            split: Some("train[:2]".into()),
            column: Some("text".into()),
            system_prompt: Some("custom".into()),
            ..Default::default()
        });
        let prompts = load_prompts(&mut spec, "sys").unwrap();
        assert_eq!(prompts.iter().map(|p| p.user.as_str()).collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(prompts[0].system, "custom");
        // Local datasets are never pinned.
        assert!(spec.specifications()[0].commit.is_none());

        let mut spec = DatasetSpecification::Single(SingleDatasetSpecification {
            dataset: dir.path().to_string_lossy().into(),
            split: Some("test".into()),
            column: Some("text".into()),
            ..Default::default()
        });
        let prompts = load_prompts(&mut spec, "sys").unwrap();
        assert_eq!(prompts.iter().map(|p| p.user.as_str()).collect::<Vec<_>>(), ["x", "y"]);
    }

    #[test]
    fn missing_split_or_column_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let mut spec = DatasetSpecification::Single(SingleDatasetSpecification {
            dataset: dir.path().to_string_lossy().into(),
            column: Some("text".into()),
            ..Default::default()
        });
        let error = load_prompts(&mut spec, "sys").unwrap_err();
        assert!(error.to_string().contains("\"split\" field is required"));
    }
}
