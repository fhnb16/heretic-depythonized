// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Backend of the Heretic desktop application.
//!
//! A single worker thread owns the processing session; commands from the frontend are
//! sent to it through a channel. Stopping and cancelling bypass the channel through
//! shared flags, because the worker is busy while a run is in progress. Log messages,
//! progress and session events are forwarded to the frontend as Tauri events.

mod worker;

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use heretic_core::Settings;
use heretic_core::config::{ConfigError, settings_schema};
use heretic_core::hub::{self, RepoType};
use heretic_core::prompt::Message;
use heretic_session::{CheckpointAction, Session};
use serde::Serialize;
use serde_json::{Value, json};
use tauri::{AppHandle, Emitter, State};

use crate::worker::{Command, Worker};

/// Built-in configuration presets (the `config.*.toml` files of Heretic 2.x).
const PRESETS: &[(&str, &str)] = &[
    ("default", include_str!("../../../../presets/default.toml")),
    ("noslop", include_str!("../../../../presets/noslop.toml")),
    ("nohumor", include_str!("../../../../presets/nohumor.toml")),
    ("nosyco", include_str!("../../../../presets/nosyco.toml")),
    ("emotions", include_str!("../../../../presets/emotions.toml")),
    ("terse", include_str!("../../../../presets/terse.toml")),
    ("classifier", include_str!("../../../../presets/classifier.toml")),
    ("piqa", include_str!("../../../../presets/piqa.toml")),
];

pub struct AppState {
    worker: Worker,
    /// Access token entered in this session (never written to disk).
    hf_token: Mutex<Option<String>>,
}

type CommandResult<T> = Result<T, String>;

fn error_string(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn format_errors(errors: Vec<ConfigError>) -> String {
    errors.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n")
}

fn parse_settings(value: &Value) -> CommandResult<Settings> {
    let table = heretic_core::config::json_to_toml_table(value).map_err(error_string)?;
    Settings::from_table(table).map_err(format_errors)
}

#[derive(Serialize)]
struct AppInfo {
    version: &'static str,
    cuda_compiled: bool,
    hf_cache: String,
    has_hf_token: bool,
    /// Default folder for run checkpoints (the working directory of an installed app
    /// is usually not writable).
    default_checkpoint_dir: String,
}

fn default_checkpoint_dir() -> PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    home.join("Documents").join("Heretic").join("checkpoints")
}

#[tauri::command]
fn app_info(state: State<AppState>) -> AppInfo {
    AppInfo {
        version: env!("CARGO_PKG_VERSION"),
        cuda_compiled: heretic_engine::device::cuda_compiled(),
        hf_cache: hub::cache_dir().display().to_string(),
        has_hf_token: hub::token().is_some() || state.hf_token.lock().unwrap().is_some(),
        default_checkpoint_dir: default_checkpoint_dir().display().to_string(),
    }
}

#[tauri::command]
fn preset_ids() -> Vec<&'static str> {
    PRESETS.iter().map(|(id, _)| *id).collect()
}

/// The full settings of a preset (with defaults filled in), as JSON.
#[tauri::command]
fn preset_settings(id: String) -> CommandResult<Value> {
    let (_, text) = PRESETS
        .iter()
        .find(|(preset, _)| *preset == id)
        .ok_or_else(|| format!("Unknown preset {id}"))?;
    let table: heretic_core::toml::Table = heretic_core::toml::from_str(text).map_err(error_string)?;
    let settings = Settings::from_table(table).map_err(format_errors)?;
    serde_json::to_value(settings).map_err(error_string)
}

#[tauri::command]
fn get_settings_schema() -> Value {
    settings_schema()
}

/// Validates settings; returns a list of problems (empty if valid).
#[tauri::command]
fn validate_settings(settings: Value) -> Vec<String> {
    match parse_settings(&settings) {
        Ok(_) => Vec::new(),
        Err(errors) => errors.lines().map(String::from).collect(),
    }
}

#[tauri::command]
fn settings_to_toml(settings: Value) -> CommandResult<String> {
    Ok(parse_settings(&settings)?.to_toml_string())
}

#[tauri::command]
fn settings_from_toml(text: String) -> CommandResult<Value> {
    let table: heretic_core::toml::Table = heretic_core::toml::from_str(&text).map_err(error_string)?;
    let settings = Settings::from_table(table).map_err(format_errors)?;
    serde_json::to_value(settings).map_err(error_string)
}

#[derive(Serialize)]
struct ModelSearchResult {
    id: String,
    downloads: u64,
    likes: u64,
}

/// Searches text generation models on the Hugging Face Hub.
#[tauri::command]
async fn search_models(query: String) -> CommandResult<Vec<ModelSearchResult>> {
    tauri::async_runtime::spawn_blocking(move || {
        let query = query.trim();
        if query.len() < 2 {
            return Ok(Vec::new());
        }
        let url = format!(
            "{}/api/models?search={}&pipeline_tag=text-generation&sort=downloads&direction=-1&limit=12",
            hub::endpoint(),
            query.replace(' ', "%20")
        );
        let mut response = ureq::get(&url).call().map_err(error_string)?;
        let items: Vec<Value> = response.body_mut().read_json().map_err(error_string)?;
        Ok(items
            .into_iter()
            .filter_map(|item| {
                Some(ModelSearchResult {
                    id: item.get("id")?.as_str()?.to_string(),
                    downloads: item.get("downloads").and_then(Value::as_u64).unwrap_or(0),
                    likes: item.get("likes").and_then(Value::as_u64).unwrap_or(0),
                })
            })
            .collect())
    })
    .await
    .map_err(error_string)?
}

#[derive(Serialize)]
struct ModelInspection {
    id: String,
    supported: bool,
    message: Option<String>,
    model_type: Option<String>,
    architectures: Vec<String>,
    layers: Option<usize>,
    multimodal: bool,
    checkpoint_dtype: Option<String>,
    /// Total size of the safetensors weights in bytes, if known.
    weights_bytes: Option<u64>,
    has_chat_template: bool,
}

/// Checks whether a model (Hub ID or local path) is supported, and estimates its size.
#[tauri::command]
async fn inspect_model(id: String) -> CommandResult<ModelInspection> {
    tauri::async_runtime::spawn_blocking(move || -> CommandResult<ModelInspection> {
        let id = id.trim().to_string();
        let local = Path::new(&id);

        let (config_text, weights_bytes, files): (String, Option<u64>, Vec<String>) = if local.is_dir() {
            let config =
                std::fs::read_to_string(local.join("config.json")).map_err(|_| "config.json not found".to_string())?;
            let mut total = 0u64;
            let mut files = Vec::new();
            for entry in std::fs::read_dir(local).map_err(error_string)? {
                let entry = entry.map_err(error_string)?;
                let name = entry.file_name().to_string_lossy().to_string();
                if name.ends_with(".safetensors") {
                    total += entry.metadata().map(|m| m.len()).unwrap_or(0);
                }
                files.push(name);
            }
            (config, Some(total), files)
        } else {
            let info = hub::repo_info(RepoType::Model, &id, None).map_err(|e| format!("{e:#}"))?;
            let weights: u64 = info
                .siblings
                .iter()
                .filter(|s| s.rfilename.ends_with(".safetensors"))
                .filter_map(|s| s.size.or(s.lfs.as_ref().and_then(|l| l.size)))
                .sum();
            let files: Vec<String> = info.siblings.iter().map(|s| s.rfilename.clone()).collect();
            let path = hub::download(RepoType::Model, &id, Some(&info.sha), "config.json", None)
                .map_err(|e| format!("{e:#}"))?;
            (
                std::fs::read_to_string(path).map_err(error_string)?,
                (weights > 0).then_some(weights),
                files,
            )
        };

        let config: Value = serde_json::from_str(&config_text).map_err(error_string)?;
        let text = heretic_engine::config::text_config_json(&config);
        let architectures = config
            .get("architectures")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect())
            .unwrap_or_default();

        let parsed = heretic_engine::config::TextConfig::from_json(&config);
        let has_chat_template = files
            .iter()
            .any(|f| f == "chat_template.jinja" || f == "tokenizer_config.json");

        Ok(ModelInspection {
            id,
            supported: parsed.is_ok() && files.iter().any(|f| f.ends_with(".safetensors")),
            message: match &parsed {
                Err(error) => Some(format!("{error:#}")),
                Ok(_) if !files.iter().any(|f| f.ends_with(".safetensors")) => {
                    Some("The model has no safetensors weights".into())
                }
                Ok(_) => None,
            },
            model_type: text.get("model_type").and_then(Value::as_str).map(String::from),
            architectures,
            layers: text
                .get("num_hidden_layers")
                .and_then(Value::as_u64)
                .map(|n| n as usize),
            multimodal: heretic_engine::config::is_multimodal(&config),
            checkpoint_dtype: parsed.ok().and_then(|c| c.torch_dtype),
            weights_bytes,
            has_chat_template,
        })
    })
    .await
    .map_err(error_string)?
}

#[derive(Serialize)]
struct Project {
    model: String,
    path: String,
    finished: bool,
    completed_trials: usize,
    n_trials: Option<usize>,
    modified: Option<u64>,
    settings: Option<Value>,
}

/// Lists the study checkpoints in a directory.
#[tauri::command]
fn list_projects(checkpoint_dir: String) -> Vec<Project> {
    let mut projects = Vec::new();
    let Ok(entries) = std::fs::read_dir(&checkpoint_dir) else {
        return projects;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "jsonl") {
            continue;
        }
        let Ok((_, attrs, trials)) = heretic_optim::Study::peek(&path) else {
            continue;
        };
        let settings = attrs.get("settings").cloned();
        projects.push(Project {
            model: settings
                .as_ref()
                .and_then(|s| s.get("model"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            path: path.display().to_string(),
            finished: attrs.get("finished").and_then(Value::as_bool).unwrap_or(false),
            completed_trials: trials
                .iter()
                .filter(|t| t.state == heretic_optim::TrialState::Complete)
                .count(),
            n_trials: settings
                .as_ref()
                .and_then(|s| s.get("n_trials"))
                .and_then(Value::as_u64)
                .map(|n| n as usize),
            modified: entry
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs()),
            settings,
        });
    }
    projects.sort_by_key(|p| std::cmp::Reverse(p.modified));
    projects
}

#[tauri::command]
fn delete_project(path: String) -> CommandResult<()> {
    let path = PathBuf::from(path);
    if path.extension().is_none_or(|e| e != "jsonl") {
        return Err("Not a checkpoint file".into());
    }
    std::fs::remove_file(path).map_err(error_string)
}

/// Starts a run in the background. Progress is reported through events.
#[tauri::command]
fn start_run(state: State<AppState>, settings: Value, checkpoint_action: Option<String>) -> CommandResult<()> {
    let settings = parse_settings(&settings)?;
    if settings.model.trim().is_empty() {
        return Err("No model selected".into());
    }
    let action = checkpoint_action.as_deref().and_then(CheckpointAction::parse);
    state.worker.send_async(Command::Start {
        settings: Box::new(settings),
        action,
    })
}

/// Checks whether a checkpoint exists for the given settings.
#[tauri::command]
fn checkpoint_info(settings: Value) -> CommandResult<Option<heretic_session::CheckpointInfo>> {
    let settings = parse_settings(&settings)?;
    let session = Session::new(
        settings,
        Arc::new(heretic_core::report::NullReporter),
        heretic_session::no_events(),
    );
    Ok(session.existing_checkpoint())
}

#[tauri::command]
fn run_more_trials(state: State<AppState>, count: usize) -> CommandResult<()> {
    if count == 0 {
        return Err("The number of trials must be greater than 0".into());
    }
    state.worker.send_async(Command::RunMore { count })
}

/// Stops the run after the current trial.
#[tauri::command]
fn stop_run(state: State<AppState>) {
    if let Some(stop) = state.worker.stop_flag() {
        stop.store(true, Ordering::SeqCst);
    }
}

/// Aborts the current operation (the running trial is discarded).
#[tauri::command]
fn cancel_run(state: State<AppState>) {
    if let Some(stop) = state.worker.stop_flag() {
        stop.store(true, Ordering::SeqCst);
    }
    if let Some(cancel) = state.worker.cancel_flag() {
        cancel.store(true, Ordering::SeqCst);
    }
}

#[tauri::command]
async fn get_trials(state: State<'_, AppState>) -> CommandResult<Value> {
    state.worker.request(Command::Trials).await
}

#[tauri::command]
async fn restore_trial(state: State<'_, AppState>, number: usize) -> CommandResult<Value> {
    state.worker.request(Command::Restore { number }).await
}

#[tauri::command]
async fn chat_send(state: State<'_, AppState>, messages: Vec<Message>, use_adapters: bool) -> CommandResult<Value> {
    state.worker.request(Command::Chat { messages, use_adapters }).await
}

#[tauri::command]
fn chat_stop(state: State<AppState>) {
    if let Some(cancel) = state.worker.cancel_flag() {
        cancel.store(true, Ordering::SeqCst);
    }
}

#[tauri::command]
async fn save_model(state: State<'_, AppState>, directory: String, strategy: String) -> CommandResult<Value> {
    let strategy = parse_strategy(&strategy)?;
    state
        .worker
        .request(Command::Save {
            directory: PathBuf::from(directory),
            strategy,
        })
        .await
}

fn parse_strategy(strategy: &str) -> CommandResult<heretic_core::config::ExportStrategy> {
    match strategy {
        "merge" => Ok(heretic_core::config::ExportStrategy::Merge),
        "adapter" => Ok(heretic_core::config::ExportStrategy::Adapter),
        other => Err(format!("Unknown export strategy {other}")),
    }
}

fn resolve_token(state: &AppState, token: Option<String>) -> CommandResult<String> {
    if let Some(token) = token.map(|t| t.trim().to_string()).filter(|t| !t.is_empty()) {
        *state.hf_token.lock().unwrap() = Some(token.clone());
        return Ok(token);
    }
    if let Some(token) = state.hf_token.lock().unwrap().clone() {
        return Ok(token);
    }
    hub::token().ok_or_else(|| "A Hugging Face access token is required".to_string())
}

#[derive(Serialize)]
struct HubUser {
    name: String,
    fullname: Option<String>,
    email: Option<String>,
}

#[tauri::command]
async fn hf_whoami(state: State<'_, AppState>, token: Option<String>) -> CommandResult<HubUser> {
    let token = resolve_token(&state, token)?;
    tauri::async_runtime::spawn_blocking(move || {
        heretic_export::hub_upload::whoami(&token)
            .map(|u| HubUser {
                name: u.name,
                fullname: u.fullname,
                email: u.email,
            })
            .map_err(|e| format!("{e:#}"))
    })
    .await
    .map_err(error_string)?
}

#[tauri::command]
async fn upload_model(
    state: State<'_, AppState>,
    token: Option<String>,
    repo_id: String,
    private: bool,
    strategy: String,
) -> CommandResult<Value> {
    let token = resolve_token(&state, token)?;
    let strategy = parse_strategy(&strategy)?;
    state
        .worker
        .request(Command::Upload {
            token,
            repo_id,
            private,
            strategy,
        })
        .await
}

#[tauri::command]
async fn model_card_preview(state: State<'_, AppState>) -> CommandResult<Value> {
    state.worker.request(Command::ModelCard).await
}

#[tauri::command]
fn forget_token(state: State<AppState>) {
    *state.hf_token.lock().unwrap() = None;
}

pub(crate) fn emit(app: &AppHandle, event: &str, payload: Value) {
    let _ = app.emit(event, payload);
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let handle = app.handle().clone();
            tauri::Manager::manage(
                app,
                AppState {
                    worker: Worker::spawn(handle),
                    hf_token: Mutex::new(None),
                },
            );
            emit(app.handle(), "heretic://ready", json!({}));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            app_info,
            preset_ids,
            preset_settings,
            get_settings_schema,
            validate_settings,
            settings_to_toml,
            settings_from_toml,
            search_models,
            inspect_model,
            list_projects,
            delete_project,
            checkpoint_info,
            start_run,
            run_more_trials,
            stop_run,
            cancel_run,
            get_trials,
            restore_trial,
            chat_send,
            chat_stop,
            save_model,
            hf_whoami,
            upload_model,
            model_card_preview,
            forget_token,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Heretic");
}
