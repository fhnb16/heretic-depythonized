// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! A loaded model with the high-level operations Heretic needs.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use anyhow::{Context as _, Result, anyhow, bail};
use candle_core::{DType, Device, Tensor};
use heretic_core::Settings;
use heretic_core::chat::TemplateDate;
use heretic_core::hub::{RepoType, ResolvedRepo};
use heretic_core::prompt::{Message, Prompt};
use heretic_core::report::{ReporterExt, SharedReporter};
use heretic_core::tokenizer::{ModelTokenizer, TOKENIZER_FILES};
use heretic_core::util::batchify;
use serde_json::Value;

use crate::config::{TextConfig, is_multimodal};
use crate::device::{device_name, dtype_name, resolve_device, resolve_dtype};
use crate::generate::{Decoding, GenerateRequest, GenerationConfig, generate};
use crate::lora::LoraLinear;
use crate::transformer::{CaptureRequest, Captured, ForwardInput, ModuleIo, Transformer, prefill_inputs};
use crate::weights::{WeightStore, safetensors_files};

/// Runtime options that are determined after loading (batch size, response prefix).
#[derive(Debug, Clone)]
pub struct InferenceOptions {
    pub batch_size: usize,
    pub max_response_length: usize,
    /// Appended to every rendered prompt so that evaluation happens at the point where
    /// responses start to differ (e.g. after a closed chain-of-thought block).
    pub response_prefix: String,
}

pub struct Model {
    pub model_id: String,
    pub repo: ResolvedRepo,
    pub config_json: Value,
    pub tokenizer: ModelTokenizer,
    transformer: Transformer,
    pub generation_config: GenerationConfig,
    pub dtype: DType,
    pub device: Device,
    pub weight_files: Vec<PathBuf>,
    /// Checkpoint prefix of the text decoder (e.g. "model.").
    pub text_prefix: String,
    pub multimodal: bool,
    pub options: InferenceOptions,
    /// Rank of the LoRA adapters (0 until `apply_lora` has been called).
    lora_rank: AtomicUsize,
    cancel: Arc<AtomicBool>,
    reporter: SharedReporter,
}

/// Returned when an operation was cancelled through [`Model::cancel_flag`].
#[derive(Debug, thiserror::Error)]
#[error("Operation cancelled")]
pub struct Cancelled;

/// Whether an error (or one of its causes) is a cancellation.
pub fn is_cancelled(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| cause.is::<Cancelled>())
}

fn read_json(path: &std::path::Path) -> Result<Value> {
    serde_json::from_str(&std::fs::read_to_string(path)?).with_context(|| format!("Invalid JSON in {}", path.display()))
}

impl Model {
    /// Loads a model, trying the configured dtypes in order until one works.
    pub fn load(settings: &Settings, reporter: SharedReporter) -> Result<Model> {
        reporter.heading(format!("Loading model **{}**...", settings.model));

        let repo = ResolvedRepo::open(RepoType::Model, &settings.model, settings.model_commit.as_deref())?;

        let config_path = repo.get("config.json", None)?;
        let config_json = read_json(&config_path)?;
        let text_config = TextConfig::from_json(&config_json)?;

        for file in TOKENIZER_FILES {
            if repo.has_file(file) {
                repo.get(file, None)?;
            }
        }
        let tokenizer = ModelTokenizer::from_repo(&repo)?;
        if !tokenizer.has_chat_template() {
            bail!(
                "{} has no chat template; Heretic requires an instruction-tuned model",
                settings.model
            );
        }

        let generation_config = match repo.get_optional("generation_config.json")? {
            Some(path) => GenerationConfig::from_json(&read_json(&path)?),
            None => GenerationConfig::default(),
        };

        let mut weight_files = Vec::new();
        let files = safetensors_files(&repo)?;
        let count = files.len();
        for (index, file) in files.iter().enumerate() {
            let task = format!("Downloading {file} ({}/{count})", index + 1);
            let progress_reporter = reporter.clone();
            let progress = move |done: u64, total: Option<u64>| progress_reporter.progress(&task, done, total);
            weight_files.push(repo.get(file, Some(&progress))?);
        }
        let weights = WeightStore::open(weight_files.clone())?;

        let device = resolve_device(&settings.device_map)?;
        reporter.info(format!("* Device: **{}**", device_name(&device)));

        let multimodal = is_multimodal(&config_json);
        let options = InferenceOptions {
            batch_size: settings.batch_size.max(1),
            max_response_length: settings.max_response_length,
            response_prefix: settings.response_prefix.clone().unwrap_or_default(),
        };

        let mut errors = Vec::new();
        for dtype_spec in &settings.dtypes {
            reporter.info(format!("* Trying dtype **{dtype_spec}**..."));

            let attempt = || -> Result<Model> {
                let dtype = resolve_dtype(dtype_spec, text_config.torch_dtype.as_deref())?;
                let transformer = Transformer::load(text_config.clone(), &weights, dtype, &device)?;
                let model = Model {
                    model_id: settings.model.clone(),
                    repo: repo.clone(),
                    config_json: config_json.clone(),
                    tokenizer: ModelTokenizer::from_repo(&repo)?,
                    transformer,
                    generation_config: generation_config.clone(),
                    dtype,
                    device: device.clone(),
                    weight_files: weight_files.clone(),
                    text_prefix: weights.text_prefix.clone(),
                    multimodal,
                    options: options.clone(),
                    lora_rank: AtomicUsize::new(0),
                    cancel: Arc::new(AtomicBool::new(false)),
                    reporter: reporter.clone(),
                };
                // A test run can reveal dtype-related problems such as overflowing activations.
                model.check_dtype(&settings.system_prompt)?;
                Ok(model)
            };

            match attempt() {
                Ok(model) => {
                    if dtype_spec == "auto" {
                        reporter.info(format!("* Resolved to **{}**", dtype_name(model.dtype)));
                    }
                    reporter.info(format!("* Transformer model with **{}** layers", model.num_layers()));
                    reporter.info("* Modifiable components:");
                    for (component, count) in model.component_counts() {
                        reporter.info(format!("  * **{component}**: **{count}** modules total"));
                    }
                    drop(tokenizer);
                    return Ok(model);
                }
                Err(error) => {
                    reporter.warning(format!("* Failed ({error:#})"));
                    errors.push(format!("{dtype_spec}: {error:#}"));
                }
            }
        }

        bail!(
            "Failed to load model with all configured dtypes:\n{}",
            errors.join("\n")
        )
    }

    fn check_dtype(&self, system_prompt: &str) -> Result<()> {
        let logits = self.get_logits(&[Prompt::new(system_prompt, "What is 1+1?")])?;
        let values: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        if values.iter().any(|v| !v.is_finite()) {
            bail!("the model produces non-finite logits in this dtype");
        }
        Ok(())
    }

    pub fn reporter(&self) -> &SharedReporter {
        &self.reporter
    }

    /// Setting this flag makes running and future operations fail with [`Cancelled`]
    /// until it is cleared again.
    pub fn cancel_flag(&self) -> &Arc<AtomicBool> {
        &self.cancel
    }

    pub fn check_cancelled(&self) -> Result<()> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(Cancelled.into());
        }
        Ok(())
    }

    pub fn config(&self) -> &TextConfig {
        &self.transformer.config
    }

    pub fn num_layers(&self) -> usize {
        self.transformer.num_layers()
    }

    /// The modifiable modules of a layer, by component name.
    pub fn layer_modules(&self, layer: usize) -> BTreeMap<String, Vec<Arc<LoraLinear>>> {
        self.transformer.layer_modules(layer)
    }

    /// All modifiable modules of the model.
    pub fn all_modules(&self) -> Vec<Arc<LoraLinear>> {
        (0..self.num_layers())
            .flat_map(|layer| self.layer_modules(layer).into_values().flatten())
            .collect()
    }

    /// Sorted names of the components that occur in any layer.
    pub fn modifiable_components(&self) -> Vec<String> {
        self.component_counts().into_keys().collect()
    }

    fn component_counts(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for layer in 0..self.num_layers() {
            for (component, modules) in self.layer_modules(layer) {
                *counts.entry(component).or_insert(0) += modules.len();
            }
        }
        counts
    }

    /// Enables LoRA adapters of the given rank on all modifiable modules (initially identity).
    pub fn apply_lora(&self, rank: usize) {
        self.lora_rank.store(rank, Ordering::SeqCst);
        self.reset_adapters();
    }

    /// Rank of the LoRA adapters, once `apply_lora` has been called.
    pub fn lora_rank(&self) -> Option<usize> {
        match self.lora_rank.load(Ordering::SeqCst) {
            0 => None,
            rank => Some(rank),
        }
    }

    /// Resets all adapters to the identity transformation.
    pub fn reset_adapters(&self) {
        for module in self.all_modules() {
            module.clear_adapter();
        }
    }

    /// Renders prompts with the chat template and appends the response prefix.
    pub fn render_prompts(&self, prompts: &[Prompt]) -> Result<Vec<String>> {
        prompts
            .iter()
            .map(|prompt| Ok(self.tokenizer.render_prompt(prompt)? + &self.options.response_prefix))
            .collect()
    }

    /// Generates greedy responses for one batch of prompts.
    pub fn get_responses(&self, prompts: &[Prompt], skip_special_tokens: bool) -> Result<Vec<String>> {
        self.get_responses_with(prompts, skip_special_tokens, true)
    }

    pub fn get_responses_with(
        &self,
        prompts: &[Prompt],
        skip_special_tokens: bool,
        use_adapters: bool,
    ) -> Result<Vec<String>> {
        self.check_cancelled()?;
        let texts = self.render_prompts(prompts)?;
        let batch = self.tokenizer.encode_batch(&texts)?;
        let generated = generate(
            &self.transformer,
            GenerateRequest {
                input_ids: &batch.input_ids,
                attention_mask: &batch.attention_mask,
                max_new_tokens: self.options.max_response_length,
                eos_token_ids: &self.tokenizer.eos_token_ids,
                pad_token_id: self.tokenizer.pad_token_id,
                config: &self.generation_config,
                decoding: Decoding::Greedy,
                use_adapters,
                cancel: Some(&self.cancel),
                on_token: None,
            },
        )?;
        self.check_cancelled()?;
        generated
            .iter()
            .map(|ids| self.tokenizer.decode(ids, skip_special_tokens))
            .collect()
    }

    pub fn get_responses_batched(&self, prompts: &[Prompt], skip_special_tokens: bool) -> Result<Vec<String>> {
        let mut responses = Vec::with_capacity(prompts.len());
        for batch in batchify(prompts, self.options.batch_size) {
            responses.extend(self.get_responses(&batch, skip_special_tokens)?);
        }
        Ok(responses)
    }

    /// Runs a single forward pass over the rendered prompts.
    fn forward_prompts(
        &self,
        prompts: &[Prompt],
        request: CaptureRequest,
        use_adapters: bool,
    ) -> Result<(Tensor, Captured)> {
        self.check_cancelled()?;
        let texts = self.render_prompts(prompts)?;
        let batch = self.tokenizer.encode_batch(&texts)?;
        let (b, seq) = (batch.len(), batch.seq_len());
        let flat: Vec<u32> = batch.input_ids.iter().flatten().copied().collect();
        let input_ids = Tensor::from_vec(flat, (b, seq), &self.device)?;
        let (positions, mask) = prefill_inputs(&batch.attention_mask, None, &self.device)?;
        let window = self.transformer.config.sliding_window.iter().flatten().min().copied();
        let sliding_mask = match window {
            Some(w) if seq > w => Some(prefill_inputs(&batch.attention_mask, Some(w), &self.device)?.1),
            _ => None,
        };

        let mut captured = Captured::default();
        let logits = self.transformer.forward(
            &ForwardInput {
                input_ids: &input_ids,
                positions: &positions,
                mask: &mask,
                sliding_mask: sliding_mask.as_ref(),
                use_adapters,
                all_logits: false,
            },
            None,
            request,
            &mut captured,
        )?;
        Ok((logits, captured))
    }

    /// Raw logits over the vocabulary for the first generated token, shape (prompt, token),
    /// float32, on the CPU.
    pub fn get_logits(&self, prompts: &[Prompt]) -> Result<Tensor> {
        let (logits, _) = self.forward_prompts(prompts, CaptureRequest::default(), true)?;
        Ok(logits.to_dtype(DType::F32)?.to_device(&Device::Cpu)?)
    }

    pub fn get_logits_batched(&self, prompts: &[Prompt]) -> Result<Tensor> {
        let mut parts = Vec::new();
        for batch in batchify(prompts, self.options.batch_size) {
            parts.push(self.get_logits(&batch)?);
        }
        Ok(Tensor::cat(&parts, 0)?)
    }

    /// Residual vectors at the position of the first generated token, for each prompt and
    /// layer (the first entry is the embeddings), shape (prompt, layer + 1, component),
    /// float32, on the CPU.
    ///
    /// With `winsorization_quantile` < 1, symmetric winsorization is applied to each
    /// layer of the per-prompt residuals: components are clamped to the given quantile
    /// of their absolute values.
    pub fn get_residuals(&self, prompts: &[Prompt], winsorization_quantile: f64) -> Result<Tensor> {
        let request = CaptureRequest {
            hidden_states: true,
            module_io: false,
        };
        let (_, captured) = self.forward_prompts(prompts, request, true)?;

        let residuals = Tensor::stack(&captured.hidden_states, 1)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?;

        if (0.0..1.0).contains(&winsorization_quantile) {
            winsorize(&residuals, winsorization_quantile)
        } else {
            Ok(residuals)
        }
    }

    pub fn get_residuals_batched(&self, prompts: &[Prompt], winsorization_quantile: f64) -> Result<Tensor> {
        let mut parts = Vec::new();
        for batch in batchify(prompts, self.options.batch_size) {
            parts.push(self.get_residuals(&batch, winsorization_quantile)?);
        }
        Ok(Tensor::cat(&parts, 0)?)
    }

    /// Mean of the residuals over all prompts, shape (layer + 1, component), accumulated
    /// in float64 to reduce rounding errors.
    pub fn get_residuals_mean(&self, prompts: &[Prompt], winsorization_quantile: f64) -> Result<Tensor> {
        if prompts.is_empty() {
            bail!("prompts must not be empty");
        }
        let mut sum: Option<Tensor> = None;
        let mut count = 0usize;
        for batch in batchify(prompts, self.options.batch_size) {
            let residuals = self.get_residuals(&batch, winsorization_quantile)?;
            let batch_sum = residuals.to_dtype(DType::F64)?.sum(0)?;
            count += residuals.dim(0)?;
            sum = Some(match sum {
                None => batch_sum,
                Some(sum) => (sum + batch_sum)?,
            });
        }
        let sum = sum.ok_or_else(|| anyhow!("no residuals"))?;
        Ok((sum / count as f64)?.to_dtype(DType::F32)?)
    }

    /// Inputs and outputs of all modifiable modules at the position of the first generated token.
    pub fn get_module_io(&self, prompts: &[Prompt]) -> Result<ModuleIo> {
        let request = CaptureRequest {
            hidden_states: false,
            module_io: true,
        };
        let (_, captured) = self.forward_prompts(prompts, request, true)?;
        Ok(captured.module_io)
    }

    /// Like [`get_module_io`](Self::get_module_io), but batched. Results are aggregated
    /// generically, because which modules activate can depend on the prompt (MoE models).
    pub fn get_module_io_batched(&self, prompts: &[Prompt]) -> Result<ModuleIo> {
        let batches: Vec<ModuleIo> = batchify(prompts, self.options.batch_size)
            .iter()
            .map(|batch| self.get_module_io(batch))
            .collect::<Result<_>>()?;

        let mut merged: ModuleIo = Vec::new();
        for layer in 0..self.num_layers() {
            type Parts = (Vec<Tensor>, Vec<Tensor>);
            let mut layer_map: BTreeMap<String, BTreeMap<usize, Parts>> = BTreeMap::new();
            for batch in &batches {
                let Some(components) = batch.get(layer) else { continue };
                for (component, modules) in components {
                    for (index, (input, output)) in modules {
                        let entry = layer_map
                            .entry(component.clone())
                            .or_default()
                            .entry(*index)
                            .or_default();
                        entry.0.push(input.clone());
                        entry.1.push(output.clone());
                    }
                }
            }

            let mut layer_io = BTreeMap::new();
            for (component, modules) in layer_map {
                let mut component_io = BTreeMap::new();
                for (index, (inputs, outputs)) in modules {
                    component_io.insert(index, (Tensor::cat(&inputs, 0)?, Tensor::cat(&outputs, 0)?));
                }
                layer_io.insert(component, component_io);
            }
            merged.push(layer_io);
        }
        Ok(merged)
    }

    /// Number of tokens in a text (with special tokens, like `tokenizer.encode`).
    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(self.tokenizer.encode(text, true)?.len())
    }

    /// Generates a chat response, streaming decoded text through `on_text`.
    /// Uses the model's own sampling settings and the actual current date.
    pub fn stream_chat(
        &self,
        messages: &[Message],
        use_adapters: bool,
        max_new_tokens: usize,
        cancel: Option<&AtomicBool>,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let text = self.tokenizer.apply_chat_template(messages, true, TemplateDate::Now)?;
        let batch = self.tokenizer.encode_batch(&[text])?;

        let decoding = if self.generation_config.do_sample {
            Decoding::Sample {
                seed: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0),
            }
        } else {
            Decoding::Greedy
        };

        let mut tokens: Vec<u32> = Vec::new();
        let mut emitted = 0usize;
        let tokenizer = &self.tokenizer;
        let mut on_token = |token: u32| {
            tokens.push(token);
            if let Ok(decoded) = tokenizer.decode(&tokens, true) {
                // Hold back incomplete multi-byte characters.
                if !decoded.ends_with('\u{FFFD}') && decoded.len() > emitted && decoded.is_char_boundary(emitted) {
                    on_text(&decoded[emitted..]);
                    emitted = decoded.len();
                }
            }
        };

        let generated = generate(
            &self.transformer,
            GenerateRequest {
                input_ids: &batch.input_ids,
                attention_mask: &batch.attention_mask,
                max_new_tokens,
                eos_token_ids: &self.tokenizer.eos_token_ids,
                pad_token_id: self.tokenizer.pad_token_id,
                config: &self.generation_config,
                decoding,
                use_adapters,
                cancel,
                on_token: Some(&mut on_token),
            },
        )?;

        self.tokenizer.decode(&generated[0], true)
    }
}

/// Symmetric winsorization of (prompt, layer, component) residuals: for each prompt and
/// layer, components are clamped to the given quantile of their absolute values
/// (linear interpolation between order statistics, like `torch.quantile`).
pub fn winsorize(residuals: &Tensor, quantile: f64) -> Result<Tensor> {
    let (prompts, layers, components) = residuals.dims3()?;
    let mut values: Vec<f32> = residuals.flatten_all()?.to_vec1()?;
    let mut sorted = vec![0f32; components];
    for row in values.chunks_mut(components) {
        for (s, v) in sorted.iter_mut().zip(row.iter()) {
            *s = v.abs();
        }
        sorted.sort_by(f32::total_cmp);
        let position = quantile * (components - 1) as f64;
        let lower = position.floor() as usize;
        let upper = position.ceil() as usize;
        let fraction = (position - lower as f64) as f32;
        let threshold = sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
        for v in row.iter_mut() {
            *v = v.clamp(-threshold, threshold);
        }
    }
    Ok(Tensor::from_vec(values, (prompts, layers, components), &Device::Cpu)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn winsorization_clamps_to_quantile() -> Result<()> {
        let residuals = Tensor::new(&[[[1f32, -2.0, 3.0, -10.0, 5.0]]], &Device::Cpu)?;
        // |x| sorted: 1 2 3 5 10; the 0.75 quantile is 5.
        let result = winsorize(&residuals, 0.75)?;
        assert_eq!(result.flatten_all()?.to_vec1::<f32>()?, vec![1.0, -2.0, 3.0, -5.0, 5.0]);
        // 0.5 quantile is 3, interpolation at 0.6: 3 + 0.4 * (5 - 3) = 3.8.
        let result = winsorize(&residuals, 0.6)?;
        assert_eq!(result.flatten_all()?.to_vec1::<f32>()?, vec![1.0, -2.0, 3.0, -3.8, 3.8]);
        Ok(())
    }
}
