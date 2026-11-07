// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Heretic settings.
//!
//! The schema mirrors the original `config.py` 1:1, so existing `config.toml`
//! files (including all bundled presets) keep working. Settings are layered
//! from multiple sources, in increasing order of precedence:
//!
//! 1. `config.toml` in the working directory (or the file in `HERETIC_CONFIG_FILE`)
//! 2. Environment variables (`HERETIC_<FIELD_NAME>`)
//! 3. Command line arguments (`--field-name value`)
//! 4. Values passed programmatically (e.g. from the GUI)
//!
//! Tables are deep-merged across sources, like in pydantic-settings.
//!
//! IMPORTANT: Any settings added here must be evaluated for privacy implications
//! and added to [`EXCLUDED_FIELDS`] if appropriate. Excluded fields are never
//! persisted to checkpoints or reproducibility manifests.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, anyhow, bail};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::dataset::{DatasetSpecification, SingleDatasetSpecification};
use crate::util::deep_merge_tables;

/// Fields that are never persisted (checkpoints, reproducibility manifests),
/// either because they are machine-local, private, or already resolved.
pub const EXCLUDED_FIELDS: &[&str] = &[
    "evaluate_model",
    "collect_reproducibles",
    "reproduce",
    "max_batch_size",
    "batch_size_test_prompts",
    "response_prefix_test_prompts",
    "chain_of_thought_blocks",
    "print_debug_information",
    "study_checkpoint_dir",
    "benchmarks",
    "save_directory",
    "upload_repo_id",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationMethod {
    /// No quantization.
    #[default]
    None,
    /// 8-bit block quantization.
    Q8,
    /// 4-bit block quantization.
    /// The value "bnb_4bit" from Heretic 2.x (Python) is accepted as an alias.
    #[serde(alias = "bnb_4bit")]
    Q4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ExportStrategy {
    /// Merge the LoRA adapter into the weights and export the full model.
    Merge,
    /// Export only the LoRA adapter.
    Adapter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Optimization {
    Minimize,
    Maximize,
    /// Compute the score without optimizing for it.
    None,
}

/// Configuration for a scorer plugin:
/// `{ plugin = "<plugin>", optimization = "<optimization>", instance_name = "<optional>" }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ScorerConfig {
    /// Scorer to use, e.g. "KeywordRate". Fully qualified names from Heretic 2.x
    /// (e.g. "heretic.scorers.keyword_rate.KeywordRate") are accepted as aliases.
    pub plugin: String,

    /// Optimization direction for this scorer. "minimize" / "maximize" to include
    /// the scorer as an objective, "none" to compute the score without optimizing for it.
    pub optimization: Optimization,

    /// Optional name to distinguish multiple instances of the same scorer.
    /// Instance-specific settings live under `[scorer.<ClassName>_<instance_name>]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance_name: Option<String>,
}

/// Configuration for a modifier plugin: `{ plugin = "<plugin>", instance_name = "<optional>" }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ModifierConfig {
    /// Modifier to use, e.g. "ARA". Fully qualified names from Heretic 2.x
    /// (e.g. "heretic.modifiers.ara.ARA") are accepted as aliases.
    pub plugin: String,

    /// Optional name to distinguish multiple instances of the same modifier.
    /// Instance-specific settings live under `[modifier.<ClassName>_<instance_name>]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BenchmarkSpecification {
    /// Task ID of the benchmark.
    pub task: String,
    /// Name of the benchmark for presentation purposes.
    pub name: String,
    /// Description of the benchmark for presentation purposes.
    pub description: String,
}

/// Device map: either a single device ("auto", "cpu", "cuda", "cuda:1")
/// or an explicit mapping (not yet supported by the engine).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum DeviceMap {
    Device(String),
    Map(BTreeMap<String, DeviceMapEntry>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum DeviceMapEntry {
    Index(i64),
    Name(String),
}

impl Default for DeviceMap {
    fn default() -> Self {
        DeviceMap::Device("auto".into())
    }
}

/// Maximum shard size, either in bytes or as a string such as "5GB".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum ShardSize {
    Bytes(u64),
    Text(String),
}

impl ShardSize {
    /// Parses the shard size like Transformers does ("5GB" = 5 * 10^9, "5GiB" = 5 * 2^30).
    pub fn to_bytes(&self) -> Result<u64> {
        match self {
            ShardSize::Bytes(bytes) => Ok(*bytes),
            ShardSize::Text(text) => crate::util::parse_size(text),
        }
    }
}

impl Default for ShardSize {
    fn default() -> Self {
        ShardSize::Text("5GB".into())
    }
}

fn default_dtypes() -> Vec<String> {
    // "auto" almost always means bfloat16. If that doesn't work, fall back to
    // float16, then bfloat16 (in case "auto" resolved to float32), then float32.
    ["auto", "float16", "bfloat16", "float32"].map(String::from).to_vec()
}

fn default_true() -> bool {
    true
}

fn default_max_batch_size() -> usize {
    128
}

fn default_max_response_length() -> usize {
    100
}

fn default_n_trials() -> usize {
    100
}

fn default_n_startup_trials() -> usize {
    30
}

fn default_study_checkpoint_dir() -> String {
    "checkpoints".into()
}

fn default_modified_model_type() -> String {
    "decensored".into()
}

fn default_system_prompt() -> String {
    "You are a helpful assistant.".into()
}

fn default_upload_tags() -> Vec<String> {
    ["uncensored", "decensored", "abliterated"].map(String::from).to_vec()
}

fn hf_dataset(dataset: &str, split: &str) -> SingleDatasetSpecification {
    SingleDatasetSpecification {
        dataset: dataset.into(),
        split: Some(split.into()),
        column: Some("text".into()),
        ..Default::default()
    }
}

fn default_batch_size_test_prompts() -> DatasetSpecification {
    DatasetSpecification::Single(hf_dataset("mlabonne/harmless_alpaca", "train[:256]"))
}

fn default_response_prefix_test_prompts() -> DatasetSpecification {
    DatasetSpecification::Multiple(vec![
        hf_dataset("mlabonne/harmless_alpaca", "train[:100]"),
        hf_dataset("mlabonne/harmful_behaviors", "train[:100]"),
    ])
}

fn default_chain_of_thought_blocks() -> Vec<(String, String)> {
    [
        // Most thinking models.
        ("<think>", "</think>"),
        // gpt-oss.
        (
            "<|channel|>analysis<|message|>",
            "<|end|><|start|>assistant<|channel|>final<|message|>",
        ),
        // Unknown, suggested by user.
        ("<thought>", "</thought>"),
        // Unknown, suggested by user.
        ("[THINK]", "[/THINK]"),
    ]
    .map(|(a, b)| (a.to_string(), b.to_string()))
    .to_vec()
}

fn default_scorers() -> Vec<ScorerConfig> {
    vec![
        ScorerConfig {
            plugin: "heretic.scorers.keyword_rate.KeywordRate".into(),
            optimization: Optimization::Minimize,
            instance_name: None,
        },
        ScorerConfig {
            plugin: "heretic.scorers.kl_divergence.KLDivergence".into(),
            optimization: Optimization::Minimize,
            instance_name: None,
        },
    ]
}

fn default_modifiers() -> Vec<ModifierConfig> {
    vec![ModifierConfig {
        plugin: "heretic.modifiers.ara.ARA".into(),
        instance_name: None,
    }]
}

fn default_benchmarks() -> Vec<BenchmarkSpecification> {
    [
        (
            "agieval",
            "AGIEval",
            "A Human-Centric Benchmark for Evaluating Foundation Models",
        ),
        (
            "bbh",
            "BIG-Bench Hard (BBH)",
            "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them",
        ),
        (
            "commonsense_qa",
            "CommonsenseQA",
            "A Question Answering Challenge Targeting Commonsense Knowledge",
        ),
        (
            "eq_bench",
            "EQ-Bench",
            "An Emotional Intelligence Benchmark for Large Language Models",
        ),
        ("gsm8k", "GSM8K", "Training Verifiers to Solve Math Word Problems"),
        ("hellaswag", "HellaSwag", "Can a Machine Really Finish Your Sentence?"),
        (
            "ifeval",
            "IFEval",
            "Instruction-Following Evaluation for Large Language Models",
        ),
        ("mmlu", "MMLU", "Measuring Massive Multitask Language Understanding"),
        (
            "mmlu_pro",
            "MMLU-Pro",
            "A More Robust and Challenging Multi-Task Language Understanding Benchmark",
        ),
        (
            "piqa",
            "PIQA",
            "Reasoning about Physical Commonsense in Natural Language",
        ),
        (
            "winogrande",
            "WinoGrande",
            "An Adversarial Winograd Schema Challenge at Scale",
        ),
    ]
    .map(|(task, name, description)| BenchmarkSpecification {
        task: task.into(),
        name: name.into(),
        description: description.into(),
    })
    .to_vec()
}

/// All Heretic settings. Field names are identical to the TOML keys.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Settings {
    /// Hugging Face model ID, or path to model on disk.
    #[serde(default)]
    pub model: String,

    /// Hugging Face commit hash of the model.
    #[serde(default)]
    pub model_commit: Option<String>,

    /// If this model ID or path is set, then instead of modifying the main model,
    /// evaluate this model relative to the main model.
    #[serde(default)]
    pub evaluate_model: Option<String>,

    /// If this directory path is set, then instead of modifying a model, download all
    /// reproduce.json files from public Heretic model repositories on Hugging Face,
    /// and store them in that directory for archival purposes.
    #[serde(default)]
    pub collect_reproducibles: Option<String>,

    /// If this path or URL to a reproduce.json file is set, load reproduction information
    /// from that file, and attempt to reproduce the modified model it originated from.
    #[serde(default)]
    pub reproduce: Option<String>,

    /// List of dtypes to try when loading model tensors.
    /// If loading with a dtype fails, the next dtype in the list will be tried.
    #[serde(default = "default_dtypes")]
    pub dtypes: Vec<String>,

    /// Quantization method to use when loading the model. Options:
    /// "none" (no quantization), "q8" (8-bit), "q4" (4-bit; "bnb_4bit" is an alias).
    #[serde(default)]
    pub quantization: QuantizationMethod,

    /// Device to load the model on: "auto" (first CUDA device if available, otherwise CPU),
    /// "cpu", "cuda", or "cuda:<index>".
    #[serde(default)]
    pub device_map: DeviceMap,

    /// Maximum memory to allocate per device (e.g., { "0" = "20GB", "cpu" = "64GB" }).
    #[serde(default)]
    pub max_memory: Option<BTreeMap<String, String>>,

    /// Whether to move intermediate analysis tensors (such as residuals and logprobs)
    /// to CPU memory as soon as possible to reduce peak VRAM usage.
    #[serde(default = "default_true")]
    pub offload_outputs_to_cpu: bool,

    /// Number of input sequences to process in parallel (0 = auto).
    #[serde(default)]
    pub batch_size: usize,

    /// Maximum batch size to try when automatically determining the optimal batch size.
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,

    /// Dataset of prompts to use for automatically determining the optimal batch size.
    #[serde(default = "default_batch_size_test_prompts")]
    pub batch_size_test_prompts: DatasetSpecification,

    /// Maximum number of tokens to generate for each response.
    #[serde(default = "default_max_response_length")]
    pub max_response_length: usize,

    /// Common prefix to assume for all responses, so that evaluation happens at the point
    /// where responses start to differ for different prompts. If not set, the prefix is
    /// determined automatically by comparing multiple responses.
    #[serde(default)]
    pub response_prefix: Option<String>,

    /// Dataset of prompts to use for automatically determining the response prefix.
    #[serde(default = "default_response_prefix_test_prompts")]
    pub response_prefix_test_prompts: DatasetSpecification,

    /// List of pairs of the form [cot_initializer, cot_finalizer] used to skip the
    /// Chain-of-Thought block in responses, so that evaluation happens at the start
    /// of the actual response.
    #[serde(default = "default_chain_of_thought_blocks")]
    pub chain_of_thought_blocks: Vec<(String, String)>,

    /// Whether to print additional information that can help with debugging.
    #[serde(default)]
    pub print_debug_information: bool,

    /// List of scorer configs. Each entry is an object
    /// { plugin = <plugin>, optimization = <optimization>, instance_name = <optional> }.
    #[serde(default = "default_scorers")]
    pub scorers: Vec<ScorerConfig>,

    /// List of modifier configs. Each entry is an object { plugin = <plugin>, instance_name = <optional> }.
    /// Only a single modifier can currently be applied, and this list must contain exactly one entry.
    #[serde(default = "default_modifiers")]
    pub modifiers: Vec<ModifierConfig>,

    /// Number of trials to run during optimization.
    #[serde(default = "default_n_trials")]
    pub n_trials: usize,

    /// Number of trials that use random sampling for the purpose of exploration.
    #[serde(default = "default_n_startup_trials")]
    pub n_startup_trials: usize,

    /// Random seed for reproducible optimization.
    #[serde(default)]
    pub seed: Option<u64>,

    /// Directory to save and load study progress to/from.
    #[serde(default = "default_study_checkpoint_dir")]
    pub study_checkpoint_dir: String,

    /// Adjective describing the type of model produced by the configuration.
    #[serde(default = "default_modified_model_type")]
    pub modified_model_type: String,

    /// Benchmarks to offer to the user for evaluating modified models.
    #[serde(default = "default_benchmarks")]
    pub benchmarks: Vec<BenchmarkSpecification>,

    /// Maximum size for individual safetensors files generated when exporting a model.
    #[serde(default)]
    pub max_shard_size: ShardSize,

    /// How to export the model: "merge", "adapter", or unset to prompt the user.
    #[serde(default)]
    pub export_strategy: Option<ExportStrategy>,

    /// Action to take in case a checkpoint exists: "continue", "restart", or unset to prompt the user.
    #[serde(default)]
    pub checkpoint_action: Option<String>,

    /// Index (in the sorted Pareto front) of the trial to use, or unset to prompt the user.
    #[serde(default)]
    pub trial_index: Option<usize>,

    /// Number of additional trials to run, or unset to prompt the user.
    #[serde(default)]
    pub n_additional_trials: Option<usize>,

    /// Action to take with the modified model: "save", "upload", or unset to prompt the user.
    #[serde(default)]
    pub model_action: Option<String>,

    /// Directory to save the model to, or unset to prompt the user.
    #[serde(default)]
    pub save_directory: Option<String>,

    /// Name of the Hugging Face repository to upload the model to, or unset to prompt the user.
    #[serde(default)]
    pub upload_repo_id: Option<String>,

    /// Whether the Hugging Face repository to upload the model to should be private,
    /// or unset to prompt the user.
    #[serde(default)]
    pub upload_repo_private: Option<bool>,

    /// Which reproducibility information to add to the Hugging Face repository:
    /// "full", "basic", "none", or unset to prompt the user.
    #[serde(default)]
    pub upload_reproducibility_information: Option<String>,

    /// Tags to add to the Hugging Face repository when uploading the model.
    #[serde(default = "default_upload_tags")]
    pub upload_tags: Vec<String>,

    /// Whether to attempt to reproduce the model even if there are environment mismatches,
    /// or unset to prompt the user.
    #[serde(default)]
    pub ignore_mismatches: Option<bool>,

    /// System prompt to use when prompting the model.
    #[serde(default = "default_system_prompt")]
    pub system_prompt: String,

    /// Plugin-specific settings (`[scorer.<Class>]`, `[modifier.<Class>]`, ...)
    /// and any other unknown keys.
    #[serde(flatten)]
    #[schemars(skip)]
    pub extra: toml::Table,
}

impl Default for Settings {
    fn default() -> Self {
        // All fields have serde defaults, so deserializing an empty table
        // yields the default settings.
        toml::Table::new().try_into().expect("default settings are valid")
    }
}

/// A configuration problem, reported per field like pydantic does.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigError {
    pub field: String,
    pub message: String,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

fn validate_instance_name(value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err("cannot be empty or whitespace".into());
    }
    if value.contains('.') {
        return Err("'.' is not allowed".into());
    }
    if value.chars().any(char::is_whitespace) {
        return Err("whitespace is not allowed".into());
    }
    Ok(())
}

impl ScorerConfig {
    pub fn validate(&self) -> Result<(), String> {
        match &self.instance_name {
            Some(name) => validate_instance_name(name),
            None => Ok(()),
        }
    }
}

impl ModifierConfig {
    pub fn validate(&self) -> Result<(), String> {
        match &self.instance_name {
            Some(name) => validate_instance_name(name),
            None => Ok(()),
        }
    }
}

/// Where a value should be read from when layering settings sources.
#[derive(Debug, Clone, Default)]
pub struct Sources {
    /// Explicit path to the configuration file. If `None`, `HERETIC_CONFIG_FILE`
    /// is consulted, falling back to `config.toml` in the working directory.
    pub config_file: Option<PathBuf>,
    /// Whether to read `HERETIC_*` environment variables.
    pub environment: bool,
    /// Command line arguments (without the program name), e.g. `["--n-trials", "5"]`.
    pub cli_args: Vec<String>,
    /// Programmatic overrides with the highest precedence.
    pub overrides: toml::Table,
}

impl Settings {
    /// Loads settings from all sources (see module documentation).
    pub fn load(sources: &Sources) -> Result<Settings, Vec<ConfigError>> {
        let to_errors = |error: anyhow::Error| {
            vec![ConfigError {
                field: "config".into(),
                message: format!("{error:#}"),
            }]
        };

        let mut merged = load_config_file(sources.config_file.as_deref()).map_err(to_errors)?;

        if sources.environment {
            let env = environment_table(std::env::vars()).map_err(to_errors)?;
            merged = deep_merge_tables(merged, env);
        }

        let cli = parse_cli_args(&sources.cli_args).map_err(to_errors)?;
        merged = deep_merge_tables(merged, cli);
        merged = deep_merge_tables(merged, sources.overrides.clone());

        Settings::from_table(merged)
    }

    /// Validates a settings table without reading any other source.
    pub fn from_table(table: toml::Table) -> Result<Settings, Vec<ConfigError>> {
        let settings: Settings = toml::Value::Table(table).try_into().map_err(|error: toml::de::Error| {
            vec![ConfigError {
                field: error_field(&error),
                message: error.message().to_string(),
            }]
        })?;
        settings.validate()?;
        Ok(settings)
    }

    /// Checks semantic constraints that serde cannot express.
    pub fn validate(&self) -> Result<(), Vec<ConfigError>> {
        let mut errors = Vec::new();

        for (index, scorer) in self.scorers.iter().enumerate() {
            if let Err(message) = scorer.validate() {
                errors.push(ConfigError {
                    field: format!("scorers.{index}.instance_name"),
                    message,
                });
            }
        }

        for (index, modifier) in self.modifiers.iter().enumerate() {
            if let Err(message) = modifier.validate() {
                errors.push(ConfigError {
                    field: format!("modifiers.{index}.instance_name"),
                    message,
                });
            }
        }

        if self.max_batch_size == 0 {
            errors.push(ConfigError {
                field: "max_batch_size".into(),
                message: "must be greater than 0".into(),
            });
        }

        if self.max_response_length == 0 {
            errors.push(ConfigError {
                field: "max_response_length".into(),
                message: "must be greater than 0".into(),
            });
        }

        if self.n_trials == 0 {
            errors.push(ConfigError {
                field: "n_trials".into(),
                message: "must be greater than 0".into(),
            });
        }

        if self.n_additional_trials == Some(0) {
            errors.push(ConfigError {
                field: "n_additional_trials".into(),
                message: "must be greater than 0".into(),
            });
        }

        if let Err(error) = self.max_shard_size.to_bytes() {
            errors.push(ConfigError {
                field: "max_shard_size".into(),
                message: format!("{error:#}"),
            });
        }

        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    /// The settings as a JSON object, without excluded fields.
    /// This is what gets persisted in study checkpoints and reproducibility manifests.
    pub fn to_persisted_json(&self) -> serde_json::Value {
        let mut value = serde_json::to_value(self).expect("settings are serializable");
        if let serde_json::Value::Object(map) = &mut value {
            for field in EXCLUDED_FIELDS {
                map.remove(*field);
            }
        }
        value
    }

    /// Validates `values` (typically persisted settings) without reading any other
    /// source. Settings missing from `values` get their defaults, except for excluded
    /// settings (such as `save_directory`), which are taken from `current`.
    pub fn from_persisted_json(values: &serde_json::Value, current: &Settings) -> Result<Settings, Vec<ConfigError>> {
        let mut table = json_to_toml_table(values).map_err(|error| {
            vec![ConfigError {
                field: "settings".into(),
                message: format!("{error:#}"),
            }]
        })?;

        let current_table = current.to_toml_table();
        for field in EXCLUDED_FIELDS {
            table.remove(*field);
            if let Some(value) = current_table.get(*field) {
                table.insert(field.to_string(), value.clone());
            }
        }

        Settings::from_table(table)
    }

    /// The settings as a TOML table (`None` values are omitted).
    pub fn to_toml_table(&self) -> toml::Table {
        toml::Table::try_from(self).expect("settings are serializable to TOML")
    }

    /// The settings as a TOML document (`None` values are omitted).
    pub fn to_toml_string(&self) -> String {
        toml::to_string(self).expect("settings are serializable to TOML")
    }

    /// The canonical file name of the study checkpoint for the configured model.
    pub fn study_checkpoint_file(&self) -> PathBuf {
        let name: String = self
            .model
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '_' || c == '-' {
                    c.to_string()
                } else {
                    "--".to_string()
                }
            })
            .collect();
        Path::new(&self.study_checkpoint_dir).join(format!("{name}.jsonl"))
    }

    /// The raw table at a dotted namespace such as "scorer.KeywordRate".
    pub fn plugin_namespace(&self, namespace: &str) -> Result<toml::Table> {
        let mut current = &self.extra;
        let parts: Vec<&str> = namespace.split('.').collect();
        for (index, part) in parts.iter().enumerate() {
            match current.get(*part) {
                None => return Ok(toml::Table::new()),
                Some(toml::Value::Table(table)) => {
                    if index == parts.len() - 1 {
                        return Ok(table.clone());
                    }
                    current = table;
                }
                Some(other) => {
                    bail!(
                        "Plugin namespace [{namespace}] must be a table, got {}",
                        other.type_str()
                    )
                }
            }
        }
        Ok(toml::Table::new())
    }

    /// Builds the settings for a plugin from its base table `[<top>.<Class>]`
    /// and the optional instance table `[<top>.<Class>_<instance>]`. Only keys
    /// that exist in the plugin settings schema are considered.
    pub fn plugin_settings<T>(&self, top_namespace: &str, class_name: &str, instance_name: Option<&str>) -> Result<T>
    where
        T: DeserializeOwned + JsonSchema,
    {
        let allowed = schema_properties::<T>();

        let mut namespaces = vec![format!("{top_namespace}.{class_name}")];
        if let Some(instance) = instance_name {
            namespaces.push(format!("{top_namespace}.{class_name}_{instance}"));
        }

        let mut merged = toml::Table::new();
        for namespace in namespaces {
            let table = self.plugin_namespace(&namespace)?;
            let filtered: toml::Table = table.into_iter().filter(|(key, _)| allowed.contains(key)).collect();
            merged = deep_merge_tables(merged, filtered);
        }

        toml::Value::Table(merged)
            .try_into()
            .map_err(|error: toml::de::Error| anyhow!("Invalid settings for {class_name}: {}", error.message()))
    }
}

fn error_field(error: &toml::de::Error) -> String {
    // toml reports e.g. "unknown variant `foo`, expected ..." with a span;
    // the field name isn't always available, so fall back to a generic label.
    let message = error.message();
    if let Some(start) = message.find("field `") {
        let rest = &message[start + 7..];
        if let Some(end) = rest.find('`') {
            return rest[..end].to_string();
        }
    }
    "config".into()
}

/// Resolves the configuration file to read, honoring `HERETIC_CONFIG_FILE`.
/// A missing implicit `config.toml` is silently ignored, but a missing file that
/// was explicitly configured is an error.
pub fn resolve_config_path(explicit: Option<&Path>) -> Result<Option<PathBuf>> {
    if let Some(path) = explicit {
        if !path.is_file() {
            bail!("heretic: configuration file {} does not exist", path.display());
        }
        return Ok(Some(path.to_path_buf()));
    }

    if let Ok(path) = std::env::var("HERETIC_CONFIG_FILE") {
        let path = PathBuf::from(path);
        if !path.is_file() {
            bail!("heretic: configuration file {} does not exist", path.display());
        }
        return Ok(Some(path));
    }

    let path = PathBuf::from("config.toml");
    Ok(path.is_file().then_some(path))
}

fn load_config_file(explicit: Option<&Path>) -> Result<toml::Table> {
    match resolve_config_path(explicit)? {
        None => Ok(toml::Table::new()),
        Some(path) => {
            let text = std::fs::read_to_string(&path).with_context(|| format!("Failed to read {}", path.display()))?;
            toml::from_str(&text).with_context(|| format!("Failed to parse {}", path.display()))
        }
    }
}

/// Top-level property names declared in a type's JSON schema.
pub fn schema_properties<T: JsonSchema>() -> BTreeSet<String> {
    let schema = schemars::schema_for!(T);
    schema
        .get("properties")
        .and_then(|properties| properties.as_object())
        .map(|properties| properties.keys().cloned().collect())
        .unwrap_or_default()
}

/// How a raw string from an environment variable or the command line should be interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FieldKind {
    /// The value is used verbatim.
    String,
    /// A boolean flag (implicit `true` on the command line).
    Bool,
    /// Parsed as JSON (numbers, lists, objects), falling back to a string.
    Structured,
}

fn field_kinds() -> BTreeMap<String, FieldKind> {
    let schema = schemars::schema_for!(Settings);
    let mut kinds = BTreeMap::new();

    let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) else {
        return kinds;
    };

    for (name, property) in properties {
        let types: Vec<String> = match property.get("type") {
            Some(serde_json::Value::String(t)) => vec![t.clone()],
            Some(serde_json::Value::Array(ts)) => ts.iter().filter_map(|t| t.as_str().map(String::from)).collect(),
            _ => vec![],
        };
        let non_null: Vec<&String> = types.iter().filter(|t| *t != "null").collect();

        let kind = if non_null.len() == 1 && non_null[0] == "string" {
            FieldKind::String
        } else if non_null.len() == 1 && non_null[0] == "boolean" {
            FieldKind::Bool
        } else {
            FieldKind::Structured
        };
        kinds.insert(name.clone(), kind);
    }

    kinds
}

fn parse_raw_value(raw: &str, kind: FieldKind) -> Result<Option<toml::Value>> {
    match kind {
        FieldKind::String => Ok(Some(toml::Value::String(raw.to_string()))),
        FieldKind::Bool => match raw.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => Ok(Some(toml::Value::Boolean(true))),
            "false" | "0" | "no" | "off" => Ok(Some(toml::Value::Boolean(false))),
            "null" | "none" => Ok(None),
            _ => bail!("invalid boolean value: {raw}"),
        },
        FieldKind::Structured => match serde_json::from_str::<serde_json::Value>(raw) {
            Ok(serde_json::Value::Null) => Ok(None),
            Ok(json) => Ok(json_to_toml(&json)),
            // Not valid JSON, so it must be a plain string (e.g. an enum value).
            Err(_) => Ok(Some(toml::Value::String(raw.to_string()))),
        },
    }
}

/// Reads `HERETIC_<FIELD>` variables for all known fields.
pub fn environment_table(vars: impl Iterator<Item = (String, String)>) -> Result<toml::Table> {
    let kinds = field_kinds();
    let mut table = toml::Table::new();

    for (key, raw) in vars {
        let Some(name) = key.strip_prefix("HERETIC_") else {
            continue;
        };
        let name = name.to_ascii_lowercase();
        let Some(kind) = kinds.get(&name) else {
            // E.g. HERETIC_CONFIG_FILE.
            continue;
        };
        if let Some(value) = parse_raw_value(&raw, *kind).with_context(|| format!("Invalid value for {key}"))? {
            table.insert(name, value);
        }
    }

    Ok(table)
}

/// Parses `--field-name value` style arguments (kebab case, like pydantic-settings).
/// Boolean fields can be given as implicit flags (`--flag`, `--no-flag`).
pub fn parse_cli_args(args: &[String]) -> Result<toml::Table> {
    let kinds = field_kinds();
    let mut table = toml::Table::new();
    let mut index = 0;

    while index < args.len() {
        let arg = &args[index];
        let Some(flag) = arg.strip_prefix("--") else {
            bail!("Unexpected argument: {arg}");
        };

        let (flag, inline_value) = match flag.split_once('=') {
            Some((flag, value)) => (flag, Some(value.to_string())),
            None => (flag, None),
        };

        let name = flag.replace('-', "_");

        if let Some(kind) = kinds.get(&name) {
            let raw = match inline_value {
                Some(value) => value,
                None if *kind == FieldKind::Bool => {
                    let next_is_value = args
                        .get(index + 1)
                        .is_some_and(|next| parse_raw_value(next, FieldKind::Bool).is_ok() && !next.starts_with("--"));
                    if next_is_value {
                        index += 1;
                        args[index].clone()
                    } else {
                        "true".into()
                    }
                }
                None => {
                    index += 1;
                    args.get(index)
                        .cloned()
                        .ok_or_else(|| anyhow!("Missing value for --{flag}"))?
                }
            };
            if let Some(value) = parse_raw_value(&raw, *kind).with_context(|| format!("Invalid value for --{flag}"))? {
                table.insert(name, value);
            }
        } else if let Some(negated) = name.strip_prefix("no_")
            && kinds.get(negated) == Some(&FieldKind::Bool)
        {
            table.insert(negated.to_string(), toml::Value::Boolean(false));
        } else {
            bail!("Unknown option: --{flag}");
        }

        index += 1;
    }

    Ok(table)
}

/// Separates positional arguments (e.g. the model ID in `heretic <model>`) from options.
/// A token is positional unless it is the value of a preceding non-boolean option.
pub fn split_positional_args(args: &[String]) -> (Vec<String>, Vec<String>) {
    let kinds = field_kinds();
    let mut positional = Vec::new();
    let mut options = Vec::new();
    let mut index = 0;

    while index < args.len() {
        let arg = &args[index];
        if let Some(flag) = arg.strip_prefix("--") {
            options.push(arg.clone());
            if !flag.contains('=') {
                let name = flag.replace('-', "_");
                let takes_value = match kinds.get(&name) {
                    Some(FieldKind::Bool) => args
                        .get(index + 1)
                        .is_some_and(|next| parse_raw_value(next, FieldKind::Bool).is_ok() && !next.starts_with("--")),
                    Some(_) => true,
                    None => false,
                };
                if takes_value && let Some(value) = args.get(index + 1) {
                    options.push(value.clone());
                    index += 1;
                }
            }
        } else {
            positional.push(arg.clone());
        }
        index += 1;
    }

    (positional, options)
}

/// Converts a JSON value to TOML (`null` has no TOML equivalent and is dropped).
pub fn json_to_toml(value: &serde_json::Value) -> Option<toml::Value> {
    match value {
        serde_json::Value::Null => None,
        serde_json::Value::Bool(b) => Some(toml::Value::Boolean(*b)),
        serde_json::Value::Number(n) => n
            .as_i64()
            .map(toml::Value::Integer)
            .or_else(|| n.as_f64().map(toml::Value::Float)),
        serde_json::Value::String(s) => Some(toml::Value::String(s.clone())),
        serde_json::Value::Array(items) => Some(toml::Value::Array(items.iter().filter_map(json_to_toml).collect())),
        serde_json::Value::Object(map) => Some(toml::Value::Table(
            map.iter()
                .filter_map(|(k, v)| json_to_toml(v).map(|v| (k.clone(), v)))
                .collect(),
        )),
    }
}

pub fn json_to_toml_table(value: &serde_json::Value) -> Result<toml::Table> {
    match json_to_toml(value) {
        Some(toml::Value::Table(table)) => Ok(table),
        Some(_) => bail!("expected an object"),
        None => Ok(toml::Table::new()),
    }
}

/// JSON schema of the main settings, used by the GUI to generate forms.
pub fn settings_schema() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(Settings)).expect("schema is serializable")
}

/// Field documentation (name, description) for `--help` output.
pub fn settings_field_docs() -> Vec<(String, String)> {
    let schema = settings_schema();
    let mut docs = Vec::new();
    if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
        for (name, property) in properties {
            let description = property
                .get("description")
                .and_then(|d| d.as_str())
                .unwrap_or_default()
                .replace('\n', " ");
            docs.push((name.clone(), description));
        }
    }
    docs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(text: &str) -> toml::Table {
        toml::from_str(text).unwrap()
    }

    #[test]
    fn defaults_match_original() {
        let settings = Settings::default();
        assert_eq!(settings.dtypes, ["auto", "float16", "bfloat16", "float32"]);
        assert_eq!(settings.n_trials, 100);
        assert_eq!(settings.n_startup_trials, 30);
        assert_eq!(settings.max_response_length, 100);
        assert_eq!(settings.upload_tags, ["uncensored", "decensored", "abliterated"]);
        assert_eq!(settings.scorers.len(), 2);
        assert_eq!(settings.modifiers[0].plugin, "heretic.modifiers.ara.ARA");
        assert!(settings.offload_outputs_to_cpu);
        assert_eq!(
            settings.chain_of_thought_blocks[0],
            ("<think>".into(), "</think>".into())
        );
    }

    #[test]
    fn accepts_slug_like_instance_name() {
        let config = ScorerConfig {
            plugin: "KeywordRate".into(),
            optimization: Optimization::Minimize,
            instance_name: Some("small-1".into()),
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn rejects_bad_instance_names() {
        assert_eq!(
            validate_instance_name(" \t").unwrap_err(),
            "cannot be empty or whitespace"
        );
        for name in ["small name", "small\tname", "small\nname"] {
            assert_eq!(validate_instance_name(name).unwrap_err(), "whitespace is not allowed");
        }
        assert_eq!(validate_instance_name("small.name").unwrap_err(), "'.' is not allowed");
    }

    #[test]
    fn all_bundled_presets_parse() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../presets");
        let mut count = 0;
        for entry in std::fs::read_dir(&root).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "toml") {
                let text = std::fs::read_to_string(&path).unwrap();
                let result = Settings::from_table(table(&text));
                assert!(result.is_ok(), "{}: {:?}", path.display(), result.err());
                count += 1;
            }
        }
        assert!(count >= 8);
    }

    #[test]
    fn plugin_tables_are_kept_in_extra() {
        let settings = Settings::from_table(table(
            "model = \"m\"\n[scorer.KeywordRate]\nscore_name = \"X\"\n[modifier.Abliteration]\nrow_normalization = \"pre\"\n",
        ))
        .unwrap();
        assert_eq!(
            settings.plugin_namespace("modifier.Abliteration").unwrap(),
            table("row_normalization = \"pre\"")
        );
        assert!(settings.plugin_namespace("scorer.Missing").unwrap().is_empty());
    }

    #[test]
    fn persisted_json_keeps_plugin_tables_and_excludes_private_fields() {
        let current = Settings {
            save_directory: Some("current".into()),
            ..Settings::default()
        };

        let values = serde_json::json!({
            "model": "model",
            "save_directory": "from-values",
            "modifier": {"Abliteration": {"row_normalization": "pre"}},
        });
        let settings = Settings::from_persisted_json(&values, &current).unwrap();

        assert_eq!(
            settings.extra,
            table("[modifier.Abliteration]\nrow_normalization = \"pre\"")
        );
        // Settings with exclusion come from the current settings.
        assert_eq!(settings.save_directory.as_deref(), Some("current"));

        let persisted = settings.to_persisted_json();
        assert!(persisted.get("save_directory").is_none());
        assert!(persisted.get("benchmarks").is_none());
        assert_eq!(persisted["modifier"]["Abliteration"]["row_normalization"], "pre");
    }

    #[test]
    fn persisted_settings_missing_values_get_defaults() {
        let settings =
            Settings::from_persisted_json(&serde_json::json!({"model": "model"}), &Settings::default()).unwrap();
        assert_eq!(settings.n_trials, 100);
        assert!(settings.extra.is_empty());
    }

    #[test]
    fn cli_args_are_parsed_by_kind() {
        let args: Vec<String> = [
            "--n-trials",
            "5",
            "--model",
            "123",
            "--print-debug-information",
            "--upload-tags",
            "[\"cli-tag\", \"test\"]",
            "--no-offload-outputs-to-cpu",
            "--quantization=bnb_4bit",
        ]
        .map(String::from)
        .to_vec();
        let parsed = parse_cli_args(&args).unwrap();
        let settings = Settings::from_table(parsed).unwrap();
        assert_eq!(settings.n_trials, 5);
        // String fields are never parsed as JSON.
        assert_eq!(settings.model, "123");
        assert!(settings.print_debug_information);
        assert_eq!(settings.upload_tags, ["cli-tag", "test"]);
        assert!(!settings.offload_outputs_to_cpu);
        assert_eq!(settings.quantization, QuantizationMethod::Q4);
    }

    #[test]
    fn positional_model_argument() {
        let args = |list: &[&str]| list.iter().map(|s| s.to_string()).collect::<Vec<_>>();

        let (positional, options) = split_positional_args(&args(&["--n-trials", "5", "Qwen/Qwen3-4B"]));
        assert_eq!(positional, ["Qwen/Qwen3-4B"]);
        assert_eq!(options, ["--n-trials", "5"]);

        // Unlike the Python version, a trailing option value is not mistaken for the model.
        let (positional, _) = split_positional_args(&args(&["--n-trials", "5"]));
        assert!(positional.is_empty());

        let (positional, options) = split_positional_args(&args(&["--print-debug-information", "m"]));
        assert_eq!(positional, ["m"]);
        assert_eq!(options, ["--print-debug-information"]);
    }

    #[test]
    fn unknown_cli_option_is_rejected() {
        assert!(parse_cli_args(&["--does-not-exist".into(), "1".into()]).is_err());
    }

    #[test]
    fn environment_overrides() {
        let env = environment_table(
            [
                ("HERETIC_MAX_RESPONSE_LENGTH".to_string(), "9".to_string()),
                ("HERETIC_CONFIG_FILE".to_string(), "x.toml".to_string()),
                ("PATH".to_string(), "/bin".to_string()),
            ]
            .into_iter(),
        )
        .unwrap();
        assert_eq!(env, table("max_response_length = 9"));
    }

    #[test]
    fn sources_are_layered() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("custom.toml");
        std::fs::write(
            &path,
            "n_trials = 42\nmax_response_length = 7\n[scorer.KeywordRate]\na = 1\n",
        )
        .unwrap();

        let sources = Sources {
            config_file: Some(path),
            environment: false,
            cli_args: vec!["--max-response-length".into(), "8".into()],
            overrides: table("[scorer.KeywordRate]\nb = 2"),
        };
        let settings = Settings::load(&sources).unwrap();
        assert_eq!(settings.n_trials, 42);
        assert_eq!(settings.max_response_length, 8);
        assert_eq!(
            settings.plugin_namespace("scorer.KeywordRate").unwrap(),
            table("a = 1\nb = 2")
        );
    }

    #[test]
    fn missing_explicit_config_file_aborts() {
        let error = resolve_config_path(Some(Path::new("definitely-missing.toml"))).unwrap_err();
        assert!(error.to_string().contains("does not exist"));
    }

    #[test]
    fn study_checkpoint_file_name() {
        let settings = Settings {
            model: "Qwen/Qwen3-4B-Instruct-2507".into(),
            ..Settings::default()
        };
        assert_eq!(
            settings.study_checkpoint_file(),
            Path::new("checkpoints").join("Qwen--Qwen3-4B-Instruct-2507.jsonl")
        );
    }

    #[derive(Debug, Deserialize, JsonSchema)]
    #[serde(default)]
    struct DemoPluginSettings {
        score_name: String,
        threshold: i64,
    }

    impl Default for DemoPluginSettings {
        fn default() -> Self {
            Self {
                score_name: "Demo".into(),
                threshold: 3,
            }
        }
    }

    #[test]
    fn plugin_settings_merge_instance_overrides() {
        let settings = Settings::from_table(table(
            "[scorer.Demo]\nscore_name = \"Base\"\nunrelated = true\n[scorer.Demo_small]\nthreshold = 9\n",
        ))
        .unwrap();

        let base: DemoPluginSettings = settings.plugin_settings("scorer", "Demo", None).unwrap();
        assert_eq!((base.score_name.as_str(), base.threshold), ("Base", 3));

        let small: DemoPluginSettings = settings.plugin_settings("scorer", "Demo", Some("small")).unwrap();
        assert_eq!((small.score_name.as_str(), small.threshold), ("Base", 9));
    }
}
