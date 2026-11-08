// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! A processing session: loading the model, preparing evaluation, running the
//! optimization, and working with the resulting trials.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Context as _, Result, anyhow, bail};
use heretic_core::Settings;
use heretic_core::config::ExportStrategy;
use heretic_core::dataset::load_prompts;
use heretic_core::prompt::{Message, Prompt};
use heretic_core::report::{ReporterExt, SharedReporter};
use heretic_core::util::{common_prefix, format_duration};
use heretic_engine::{Model, is_cancelled};
use heretic_optim::{FrozenTrial, Study, TpeConfig};
use heretic_plugins::evaluator::Evaluator;
use heretic_plugins::{Context, Modifier, Score, create_modifier};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::events::{EventSink, Phase, ScoreRecord, SessionEvent, TrialSummary};

/// What to do when a checkpoint for the model already exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointAction {
    /// Continue the previous run (or show its results if it is finished).
    Continue,
    /// Delete the checkpoint and start from scratch.
    Restart,
}

impl CheckpointAction {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "continue" => Some(CheckpointAction::Continue),
            "restart" => Some(CheckpointAction::Restart),
            _ => None,
        }
    }
}

/// Information about an existing study checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointInfo {
    pub path: PathBuf,
    pub finished: bool,
    pub completed_trials: usize,
    pub n_trials: Option<usize>,
}

/// Why `run_trials` returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunOutcome {
    /// The requested number of trials was reached.
    Completed,
    /// A stop was requested (graceful, between trials).
    Stopped,
    /// The running trial was cancelled.
    Cancelled,
}

pub struct Session {
    pub settings: Settings,
    reporter: SharedReporter,
    events: EventSink,
    model: Option<Model>,
    evaluator: Option<Evaluator>,
    modifier: Option<Box<dyn Modifier>>,
    study: Option<Study>,
    stop: Arc<AtomicBool>,
    /// Number of the trial whose modification is currently applied to the model.
    pub applied_trial: Option<usize>,
}

fn study_seed(settings: &Settings) -> u64 {
    settings.seed.unwrap_or(0)
}

fn tpe_config(settings: &Settings) -> TpeConfig {
    TpeConfig {
        n_startup_trials: settings.n_startup_trials,
        n_ei_candidates: 128,
        ..TpeConfig::default()
    }
}

impl Session {
    pub fn new(settings: Settings, reporter: SharedReporter, events: EventSink) -> Self {
        Self {
            settings,
            reporter,
            events,
            model: None,
            evaluator: None,
            modifier: None,
            study: None,
            stop: Arc::new(AtomicBool::new(false)),
            applied_trial: None,
        }
    }

    pub fn reporter(&self) -> &SharedReporter {
        &self.reporter
    }

    fn emit(&self, event: SessionEvent) {
        (self.events)(&event);
    }

    fn phase(&self, phase: Phase) {
        self.emit(SessionEvent::Phase { phase });
    }

    pub fn model(&self) -> Option<&Model> {
        self.model.as_ref()
    }

    pub fn study(&self) -> Option<&Study> {
        self.study.as_ref()
    }

    pub fn modifier(&self) -> Option<&dyn Modifier> {
        self.modifier.as_deref()
    }

    pub fn evaluator(&self) -> Option<&Evaluator> {
        self.evaluator.as_ref()
    }

    /// Requests a graceful stop after the current trial.
    pub fn stop_handle(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }

    /// Flag that aborts the running operation (available once the model is loaded).
    pub fn cancel_handle(&self) -> Option<Arc<AtomicBool>> {
        self.model.as_ref().map(|m| m.cancel_flag().clone())
    }

    /// Information about an existing checkpoint for the configured model, if any.
    pub fn existing_checkpoint(&self) -> Option<CheckpointInfo> {
        let path = self.settings.study_checkpoint_file();
        if !path.is_file() {
            return None;
        }
        let (_, attrs, trials) = Study::peek(&path).ok()?;
        let n_trials = attrs
            .get("settings")
            .and_then(|s| s.get("n_trials"))
            .and_then(Value::as_u64)
            .map(|n| n as usize);
        Some(CheckpointInfo {
            finished: attrs.get("finished").and_then(Value::as_bool).unwrap_or(false),
            completed_trials: trials
                .iter()
                .filter(|t| t.state == heretic_optim::TrialState::Complete)
                .count(),
            n_trials,
            path,
        })
    }

    /// Prepares everything needed for optimization: applies the checkpoint action, loads
    /// the model, determines the batch size and response prefix, initializes scorers and
    /// the modifier, and opens the study.
    pub fn prepare(&mut self, checkpoint_action: Option<CheckpointAction>) -> Result<()> {
        let checkpoint_path = self.settings.study_checkpoint_file();
        let mut resume = false;

        if let Some(info) = self.existing_checkpoint() {
            match checkpoint_action {
                Some(CheckpointAction::Continue) => {
                    let (_, attrs, _) = Study::peek(&info.path)?;
                    if let Some(persisted) = attrs.get("settings") {
                        self.settings = Settings::from_persisted_json(persisted, &self.settings).map_err(|errors| {
                            anyhow!(
                                "The checkpoint contains invalid settings: {}",
                                errors.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; ")
                            )
                        })?;
                    }
                    resume = true;
                }
                Some(CheckpointAction::Restart) => {
                    std::fs::remove_file(&info.path)
                        .with_context(|| format!("Failed to delete {}", info.path.display()))?;
                }
                None => bail!(
                    "A checkpoint exists for this model ({}); choose whether to continue or restart",
                    info.path.display()
                ),
            }
        }

        if self.settings.seed.is_none() {
            use rand::RngExt;
            self.settings.seed = Some(rand::rng().random::<u32>() as u64);
        }

        self.phase(Phase::LoadingModel);
        let model = Model::load(&self.settings, self.reporter.clone())?;
        self.model = Some(model);

        if self.settings.batch_size == 0 {
            self.phase(Phase::BatchSize);
            let batch_size = self.determine_batch_size()?;
            self.settings.batch_size = batch_size;
        }
        self.model_mut()?.options.batch_size = self.settings.batch_size;

        if self.settings.response_prefix.is_none() {
            self.phase(Phase::ResponsePrefix);
            let prefix = self.determine_response_prefix()?;
            self.settings.response_prefix = Some(prefix);
        }
        let prefix = self.settings.response_prefix.clone().unwrap_or_default();
        self.model_mut()?.options.response_prefix = prefix;

        self.phase(Phase::Scorers);
        let evaluator = {
            let model = self.model.as_ref().expect("model is loaded");
            let ctx = Context::new(&self.settings, model, &self.reporter);
            Evaluator::new(&ctx)?
        };
        self.emit(SessionEvent::Baseline {
            scores: evaluator.baseline_scores.clone(),
            objective_names: evaluator.objective_names(),
        });

        if evaluator.objective_names().is_empty() {
            bail!(
                "No optimization objectives configured. At least one scorer must set \"optimization\" \
                 to \"maximize\" or \"minimize\"."
            );
        }
        self.evaluator = Some(evaluator);

        self.phase(Phase::Modifier);
        self.reporter.heading("Loading and initializing modifiers...");
        if self.settings.modifiers.len() != 1 {
            bail!("Exactly one modifier must be configured (multiple modifiers are not yet supported)");
        }
        let mut modifier = create_modifier(&self.settings.modifiers[0], &self.settings)?;
        self.reporter.info(format!("* Loaded: **{}**", modifier.class_name()));
        {
            let model = self.model.as_ref().expect("model is loaded");
            let ctx = Context::new(&self.settings, model, &self.reporter);
            modifier.init(&ctx)?;
        }
        self.modifier = Some(modifier);

        let directions = self.evaluator.as_ref().expect("evaluator").objective_directions();
        let mut study = if resume {
            let study = Study::open(&checkpoint_path, tpe_config(&self.settings), study_seed(&self.settings))?;
            if study.directions != directions {
                bail!("The checkpoint's optimization objectives don't match the configured scorers");
            }
            study
        } else {
            Study::create(
                &checkpoint_path,
                directions,
                tpe_config(&self.settings),
                study_seed(&self.settings),
            )?
        };
        study.set_user_attr("settings", self.settings.to_persisted_json())?;
        if !resume || study.user_attrs.get("finished").is_none() {
            study.set_user_attr("finished", Value::Bool(false))?;
        }
        if resume && !study.trials().is_empty() {
            self.reporter.heading("Resuming existing study.");
        }
        self.study = Some(study);

        Ok(())
    }

    fn model_mut(&mut self) -> Result<&mut Model> {
        self.model.as_mut().ok_or_else(|| anyhow!("No model is loaded"))
    }

    fn require_model(&self) -> Result<&Model> {
        self.model.as_ref().ok_or_else(|| anyhow!("No model is loaded"))
    }

    /// Benchmarks increasing batch sizes and returns the one with the highest throughput.
    fn determine_batch_size(&mut self) -> Result<usize> {
        let reporter = self.reporter.clone();
        reporter.heading(format!(
            "Loading batch size test prompts from **{}**...",
            self.settings.batch_size_test_prompts.describe()
        ));
        let test_prompts = load_prompts(&mut self.settings.batch_size_test_prompts, &self.settings.system_prompt)?;
        reporter.info(format!("* **{}** prompts loaded", test_prompts.len()));
        if test_prompts.is_empty() {
            bail!("No batch size test prompts");
        }

        reporter.heading("Determining optimal batch size...");
        let max_batch_size = self.settings.max_batch_size;
        let model = self.model_mut()?;

        let mut batch_size = 1;
        let mut best_batch_size = 1;
        let mut best_performance = -1.0;

        while batch_size <= max_batch_size {
            let prompts: Vec<Prompt> = test_prompts.iter().cycle().take(batch_size).cloned().collect();
            model.options.batch_size = batch_size;

            let attempt = || -> Result<(Vec<String>, f64)> {
                // Warmup run so that one-time costs aren't benchmarked.
                model.get_responses(&prompts, false)?;
                let start = Instant::now();
                let responses = model.get_responses(&prompts, false)?;
                Ok((responses, start.elapsed().as_secs_f64()))
            };

            match attempt() {
                Ok((responses, seconds)) => {
                    let tokens: usize = responses.iter().map(|r| model.count_tokens(r).unwrap_or(0)).sum();
                    let performance = tokens as f64 / seconds.max(1e-9);
                    reporter.info(format!(
                        "* Trying batch size **{batch_size}**... Ok (**{performance:.0}** tokens/s)"
                    ));
                    if performance > best_performance {
                        best_batch_size = batch_size;
                        best_performance = performance;
                    }
                }
                Err(error) => {
                    if is_cancelled(&error) || batch_size == 1 {
                        return Err(error);
                    }
                    reporter.warning(format!("* Trying batch size **{batch_size}**... Failed ({error:#})"));
                    break;
                }
            }

            batch_size *= 2;
        }

        reporter.info(format!("* Chosen batch size: **{best_batch_size}**"));
        Ok(best_batch_size)
    }

    /// Detects a closed chain-of-thought block or a common response prefix, so that
    /// evaluation happens where responses start to differ.
    fn determine_response_prefix(&mut self) -> Result<String> {
        let reporter = self.reporter.clone();
        reporter.heading(format!(
            "Loading response prefix test prompts from **{}**...",
            self.settings.response_prefix_test_prompts.describe()
        ));
        let test_prompts = load_prompts(
            &mut self.settings.response_prefix_test_prompts,
            &self.settings.system_prompt,
        )?;
        reporter.info(format!("* **{}** prompts loaded", test_prompts.len()));

        reporter.heading("Checking chat template for a closed Chain-of-Thought block or an initializer...");
        let cot_blocks = self.settings.chain_of_thought_blocks.clone();
        let model = self.model_mut()?;
        model.options.response_prefix.clear();

        // If the model supports system messages, a custom one overrides default system
        // instructions, which may contain CoT examples that would cause false positives.
        let mut dummy = Vec::new();
        if model.tokenizer.supports_system_role() {
            dummy.push(Message::system("You are a helpful assistant."));
        }
        dummy.push(Message::user("This is a dummy prompt."));
        let dummy_prompt = model
            .tokenizer
            .apply_chat_template(&dummy, true, heretic_core::chat::TemplateDate::Now)?;

        let mut prefix: Option<String> = None;
        let mut cot_skip_applied = false;

        for (initializer, finalizer) in &cot_blocks {
            // The template may already insert the full closed block (thinking disabled).
            let pattern = format!("(?s){}.*{}", regex::escape(initializer), regex::escape(finalizer));
            if Regex::new(&pattern)?.is_match(&dummy_prompt) {
                reporter.info(format!(
                    "* Chat template inserts full closed CoT block **{initializer:?}{finalizer:?}**"
                ));
                reporter.info("* Response prefix is not required");
                prefix = Some(String::new());
                cot_skip_applied = true;
                break;
            }
            // The template inserts only the initializer (thinking enabled by default), so
            // the finalizer completes the closed block.
            if dummy_prompt.contains(initializer.as_str()) {
                reporter.info(format!("* Chat template inserts CoT initializer: **{initializer:?}**"));
                reporter.info(format!("* Remaining CoT finalizer: **{finalizer:?}**"));
                prefix = Some(finalizer.clone());
                cot_skip_applied = true;
                break;
            }
        }

        if !cot_skip_applied {
            reporter.info("* None found");
        }

        if prefix.is_none() {
            // Models instructed to think by their system prompt (or trained to always think)
            // generate the CoT initializer themselves, which is then closed with the finalizer.
            reporter.heading("Checking for common response prefix...");
            let responses = model.get_responses_batched(&test_prompts, false)?;
            // Trailing spaces are removed because different tokens starting with a space
            // would otherwise lead to an uncommon tokenization.
            let mut found = common_prefix(&responses).trim_end_matches(' ').to_string();
            if found.is_empty() {
                reporter.info("* None found");
            } else {
                reporter.info(format!("* Prefix found: **{found:?}**"));
                for (initializer, finalizer) in &cot_blocks {
                    if found.starts_with(initializer.as_str()) {
                        found = format!("{initializer}{finalizer}");
                        reporter.info(format!("* Closed Chain-of-Thought block: **{found:?}**"));
                        cot_skip_applied = true;
                        break;
                    }
                }
            }
            prefix = Some(found);
        }

        let mut prefix = prefix.unwrap_or_default();

        // With a CoT skip, check that the prefix is complete (e.g. not missing a trailing newline).
        if cot_skip_applied && !prefix.is_empty() {
            reporter.info("* Rechecking with prefix...");
            model.options.response_prefix = prefix.clone();
            let responses = model.get_responses_batched(&test_prompts, false)?;
            let additional = common_prefix(&responses).trim_end_matches(' ').to_string();
            if !additional.is_empty() {
                prefix.push_str(&additional);
                reporter.info(format!("* Extended prefix found: **{prefix:?}**"));
            }
        }

        model.options.response_prefix = prefix.clone();
        Ok(prefix)
    }

    /// Runs trials until the study contains `target_total` completed trials, a stop is
    /// requested, or the running trial is cancelled.
    pub fn run_trials(&mut self, target_total: usize) -> Result<RunOutcome> {
        self.stop.store(false, Ordering::SeqCst);
        if let Some(model) = &self.model {
            model.cancel_flag().store(false, Ordering::SeqCst);
        }
        self.phase(Phase::Optimizing);

        let start = Instant::now();
        let start_count = self.study.as_ref().map_or(0, |s| s.completed_trials().count());
        let mut outcome = RunOutcome::Completed;

        loop {
            let completed = self
                .study
                .as_ref()
                .ok_or_else(|| anyhow!("No study"))?
                .completed_trials()
                .count();
            if completed >= target_total {
                break;
            }
            if self.stop.load(Ordering::SeqCst) {
                outcome = RunOutcome::Stopped;
                break;
            }

            match self.run_trial(target_total, start, start_count) {
                Ok(()) => {}
                Err(error) if is_cancelled(&error) => {
                    self.reporter.warning("Trial cancelled.");
                    outcome = RunOutcome::Cancelled;
                    break;
                }
                Err(error) => return Err(error),
            }
        }

        // Any cancellation is consumed here so that the model remains usable.
        if let Some(model) = &self.model {
            model.cancel_flag().store(false, Ordering::SeqCst);
        }

        // The run is finished once the configured number of trials is reached
        // (which may be larger than this call's target).
        let goal = target_total.max(self.settings.n_trials);
        let study = self.study.as_mut().ok_or_else(|| anyhow!("No study"))?;
        let mut persisted = self.settings.to_persisted_json();
        persisted["n_trials"] = json!(goal);
        study.set_user_attr("settings", persisted)?;
        let finished = study.completed_trials().count() >= goal;
        study.set_user_attr("finished", Value::Bool(finished))?;

        if finished {
            self.phase(Phase::Finished);
        }
        Ok(outcome)
    }

    fn run_trial(&mut self, total: usize, start: Instant, start_count: usize) -> Result<()> {
        let model = self.model.as_ref().ok_or_else(|| anyhow!("No model is loaded"))?;
        let modifier = self.modifier.as_ref().ok_or_else(|| anyhow!("No modifier"))?;
        let evaluator = self.evaluator.as_mut().ok_or_else(|| anyhow!("No evaluator"))?;
        let study = self.study.as_mut().ok_or_else(|| anyhow!("No study"))?;
        let ctx = Context::new(&self.settings, model, &self.reporter);

        let mut trial = study.ask();
        let index = trial.number + 1;
        trial.set_user_attr("index", json!(index));

        let parameters = modifier.suggest_parameters(&ctx, &mut trial)?;
        trial.set_user_attr("parameters", parameters.clone());
        let presented = modifier.present_parameters(&parameters)?;

        self.reporter
            .heading(format!("Running trial **{index}** of **{total}**..."));
        self.reporter.info("* Parameters:");
        for (name, value) in &presented {
            self.reporter.info(format!("  * {name} = **{value}**"));
        }
        (self.events)(&SessionEvent::TrialStarted {
            index,
            total,
            parameters: presented.clone(),
        });

        let result = (|| -> Result<Vec<(String, Score)>> {
            self.reporter.info("* Resetting model...");
            modifier.reset_model(&ctx)?;
            self.reporter
                .info(format!("* Modifying model using {}...", modifier.modifier_name()));
            modifier.modify_model(&ctx, &parameters)?;
            self.reporter.info("* Evaluating...");
            evaluator.get_scores(&ctx)
        })();

        let scores = match result {
            Ok(scores) => scores,
            Err(error) => {
                // The model may be partially modified; the next trial or restore resets it.
                self.applied_trial = None;
                if is_cancelled(&error) {
                    (self.events)(&SessionEvent::TrialCancelled { index });
                }
                return Err(error);
            }
        };

        for (name, score) in &scores {
            self.reporter.info(format!("  * **{name}:** {}", score.display));
        }

        let values = evaluator.objective_values(&scores);
        trial.set_user_attr("scores", evaluator.paired_score_records(&scores));
        let frozen = trial.complete(values)?;
        self.applied_trial = Some(frozen.number);

        let done = study.completed_trials().count();
        let elapsed = start.elapsed().as_secs_f64();
        let run_count = done.saturating_sub(start_count).max(1);
        let remaining = (done < total).then(|| elapsed / run_count as f64 * (total - done) as f64);

        self.reporter
            .detail(format!("Elapsed time: **{}**", format_duration(elapsed)));
        if let Some(remaining) = remaining {
            self.reporter
                .detail(format!("Estimated remaining time: **{}**", format_duration(remaining)));
        }

        let pareto: Vec<usize> = study.best_trials().iter().map(|t| t.number).collect();
        let summary = summarize(&frozen, modifier.as_ref(), pareto.contains(&frozen.number))?;
        (self.events)(&SessionEvent::TrialFinished {
            trial: summary,
            pareto,
            total,
            elapsed_seconds: elapsed,
            remaining_seconds: remaining,
        });

        Ok(())
    }

    /// All completed trials as summaries, in trial order.
    pub fn trials(&self) -> Result<Vec<TrialSummary>> {
        let study = self.study.as_ref().ok_or_else(|| anyhow!("No study"))?;
        let modifier = self.modifier.as_ref().ok_or_else(|| anyhow!("No modifier"))?;
        let pareto: Vec<usize> = study.best_trials().iter().map(|t| t.number).collect();
        study
            .completed_trials()
            .map(|t| summarize(t, modifier.as_ref(), pareto.contains(&t.number)))
            .collect()
    }

    /// The Pareto-optimal trials, sorted by their scores in objective order.
    pub fn pareto_front(&self) -> Result<Vec<TrialSummary>> {
        let mut front: Vec<TrialSummary> = self.trials()?.into_iter().filter(|t| t.pareto_optimal).collect();
        front.sort_by(|a, b| {
            for (x, y) in a.values.iter().zip(&b.values) {
                match x.total_cmp(y) {
                    std::cmp::Ordering::Equal => continue,
                    other => return other,
                }
            }
            std::cmp::Ordering::Equal
        });
        Ok(front)
    }

    fn find_trial(&self, number: usize) -> Result<&FrozenTrial> {
        self.study
            .as_ref()
            .ok_or_else(|| anyhow!("No study"))?
            .trials()
            .iter()
            .find(|t| t.number == number)
            .ok_or_else(|| anyhow!("Trial {number} does not exist"))
    }

    /// Applies the modification of a trial to the model.
    pub fn restore_trial(&mut self, number: usize) -> Result<()> {
        let parameters = self
            .find_trial(number)?
            .user_attrs
            .get("parameters")
            .cloned()
            .ok_or_else(|| anyhow!("Trial {number} has no parameters"))?;
        self.apply_parameters(&parameters)?;
        self.applied_trial = Some(number);
        Ok(())
    }

    /// Applies modifier parameters directly (e.g. from a reproducibility manifest).
    pub fn apply_parameters(&mut self, parameters: &Value) -> Result<()> {
        self.phase(Phase::RestoringTrial);
        let model = self.require_model()?;
        model.cancel_flag().store(false, Ordering::SeqCst);
        let modifier = self.modifier.as_ref().ok_or_else(|| anyhow!("No modifier"))?;
        let ctx = Context::new(&self.settings, model, &self.reporter);
        self.reporter.info("* Parameters:");
        for (name, value) in modifier.present_parameters(parameters)? {
            self.reporter.info(format!("  * {name} = **{value}**"));
        }
        self.reporter.info("* Resetting model...");
        modifier.reset_model(&ctx)?;
        self.reporter
            .info(format!("* Modifying model using {}...", modifier.modifier_name()));
        modifier.modify_model(&ctx, parameters)?;
        Ok(())
    }

    /// Generates a chat response with the (modified or original) model.
    pub fn chat(&self, messages: &[Message], use_adapters: bool, on_text: &mut dyn FnMut(&str)) -> Result<String> {
        let model = self.require_model()?;
        model.cancel_flag().store(false, Ordering::SeqCst);
        let result = model.stream_chat(messages, use_adapters, 4096, Some(model.cancel_flag()), on_text);
        model.cancel_flag().store(false, Ordering::SeqCst);
        result
    }

    /// Evaluates another model relative to the main model's baseline (`evaluate_model` mode).
    pub fn evaluate_other_model(&mut self, model_id: &str) -> Result<Vec<(String, Score)>> {
        // Free the main model's memory first.
        self.model = None;
        let mut settings = self.settings.clone();
        settings.model = model_id.to_string();
        settings.model_commit = None;
        let mut model = Model::load(&settings, self.reporter.clone())?;
        model.options.batch_size = self.settings.batch_size.max(1);
        model.options.response_prefix = self.settings.response_prefix.clone().unwrap_or_default();
        let evaluator = self.evaluator.as_mut().ok_or_else(|| anyhow!("No evaluator"))?;
        let ctx = Context::new(&settings, &model, &self.reporter);
        self.reporter.info("* Evaluating...");
        let scores = evaluator.get_scores(&ctx)?;
        drop(ctx);
        self.model = Some(model);
        Ok(scores)
    }

    /// Saves the model with the currently applied trial.
    pub fn save_model(&self, dir: &std::path::Path, strategy: ExportStrategy) -> Result<heretic_export::SavedModel> {
        self.phase(Phase::Exporting);
        let model = self.require_model()?;
        match strategy {
            ExportStrategy::Adapter => {
                self.reporter.info("Saving LoRA adapter...");
                heretic_export::save_adapter(model, dir)
            }
            ExportStrategy::Merge => {
                self.reporter.info("Saving merged model...");
                heretic_export::save_merged(model, dir, self.settings.max_shard_size.to_bytes()?, &self.reporter)
            }
        }
    }

    /// Saves the model with the currently applied trial as a GGUF file. `output` is a `.gguf`
    /// file or a folder; quantized types use `llama_cpp_dir`, or the setting of the same name.
    pub fn save_gguf(
        &self,
        output: &std::path::Path,
        kind: heretic_export::gguf::GgufType,
        llama_cpp_dir: Option<&std::path::Path>,
    ) -> Result<heretic_export::gguf::GgufExport> {
        self.phase(Phase::Exporting);
        let model = self.require_model()?;
        let configured = self.settings.llama_cpp_dir.as_deref().map(std::path::Path::new);
        let name = format!(
            "{}-heretic",
            std::path::Path::new(&self.settings.model)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "model".into())
        );
        heretic_export::gguf::export_gguf(model, &name, output, kind, llama_cpp_dir.or(configured), &self.reporter)
    }

    /// The model card for the currently applied trial.
    pub fn model_card(&self) -> Result<String> {
        let model = self.require_model()?;
        let modifier = self.modifier.as_ref().ok_or_else(|| anyhow!("No modifier"))?;
        let number = self
            .applied_trial
            .ok_or_else(|| anyhow!("No trial is applied to the model"))?;
        let trial = self.find_trial(number)?;

        let parameters = trial
            .user_attrs
            .get("parameters")
            .map(|p| modifier.present_parameters(p))
            .transpose()?
            .unwrap_or_default();
        let scores: Vec<heretic_export::model_card::CardScore> = trial
            .user_attrs
            .get("scores")
            .map(|s| serde_json::from_value(s.clone()))
            .transpose()?
            .unwrap_or_default();

        let hub_model_id = model.repo.commit.is_some().then_some(self.settings.model.as_str());
        let intro = heretic_export::model_card::readme_intro(&heretic_export::model_card::CardInfo {
            hub_model_id,
            modified_model_type: &self.settings.modified_model_type,
            modifier_name: &modifier.modifier_name(),
            parameters: &parameters,
            scores: &scores,
            reproducible: false,
        });

        let original = match model.repo.get_optional("README.md") {
            Ok(Some(path)) => std::fs::read_to_string(path).unwrap_or_default(),
            _ => String::new(),
        };

        let mut tags = vec!["heretic".to_string()];
        tags.extend(self.settings.upload_tags.iter().cloned());

        Ok(heretic_export::model_card::build_model_card(
            &original,
            &intro,
            &tags,
            hub_model_id,
        ))
    }

    /// Exports the model and uploads it, with a model card, to a Hugging Face repository.
    pub fn upload_model(
        &self,
        token: &str,
        repo_id: &str,
        private: bool,
        strategy: ExportStrategy,
        progress: &dyn Fn(usize, usize, &str),
    ) -> Result<()> {
        heretic_core::hub::validate_repo_id(repo_id)?;

        // The system temp directory may be too small for a model, so stage next to the checkpoints.
        let staging = std::path::Path::new(&self.settings.study_checkpoint_dir).join("upload-staging");
        if staging.exists() {
            std::fs::remove_dir_all(&staging)?;
        }

        let result = (|| -> Result<()> {
            self.save_model(&staging, strategy)?;
            std::fs::write(staging.join("README.md"), self.model_card()?)?;

            self.reporter.info(format!("Creating repository **{repo_id}**..."));
            heretic_export::hub_upload::create_repo(token, repo_id, private)?;

            let files = heretic_export::hub_upload::directory_files(&staging)?;
            self.reporter.info(format!("Uploading **{}** files...", files.len()));
            let message = format!("Upload {} model made with Heretic", self.settings.modified_model_type);
            heretic_export::hub_upload::upload_files(token, repo_id, &files, &message, progress)
        })();

        let _ = std::fs::remove_dir_all(&staging);
        result
    }

    /// Prepares only the model and scorers (for `evaluate_model` mode).
    pub fn prepare_evaluation(&mut self) -> Result<()> {
        if self.settings.seed.is_none() {
            self.settings.seed = Some(0);
        }
        self.phase(Phase::LoadingModel);
        self.model = Some(Model::load(&self.settings, self.reporter.clone())?);
        if self.settings.batch_size == 0 {
            self.phase(Phase::BatchSize);
            self.settings.batch_size = self.determine_batch_size()?;
        }
        self.model_mut()?.options.batch_size = self.settings.batch_size;
        if self.settings.response_prefix.is_none() {
            self.phase(Phase::ResponsePrefix);
            self.settings.response_prefix = Some(self.determine_response_prefix()?);
        }
        let prefix = self.settings.response_prefix.clone().unwrap_or_default();
        self.model_mut()?.options.response_prefix = prefix;
        self.phase(Phase::Scorers);
        let model = self.model.as_ref().expect("model is loaded");
        let ctx = Context::new(&self.settings, model, &self.reporter);
        self.evaluator = Some(Evaluator::new(&ctx)?);
        Ok(())
    }
}

/// Builds a summary of a completed trial from its user attributes.
pub fn summarize(trial: &FrozenTrial, modifier: &dyn Modifier, pareto_optimal: bool) -> Result<TrialSummary> {
    let parameters = trial
        .user_attrs
        .get("parameters")
        .map(|p| modifier.present_parameters(p))
        .transpose()?
        .unwrap_or_default();
    let scores: Vec<ScoreRecord> = trial
        .user_attrs
        .get("scores")
        .map(|s| serde_json::from_value(s.clone()))
        .transpose()
        .context("Invalid score records")?
        .unwrap_or_default();
    Ok(TrialSummary {
        number: trial.number,
        index: trial
            .user_attrs
            .get("index")
            .and_then(Value::as_u64)
            .map_or(trial.number + 1, |i| i as usize),
        values: trial.values.clone().unwrap_or_default(),
        scores,
        parameters,
        pareto_optimal,
    })
}
