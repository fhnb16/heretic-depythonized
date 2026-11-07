// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Arbitrary-Rank Ablation (ARA) (Weidmann 2026).
//! See https://github.com/p-e-w/heretic/pull/211 for more information.

use anyhow::{Context as _, Result, anyhow};
use candle_core::{DType, Device, Tensor, Var};
use heretic_core::dataset::DatasetSpecification;
use heretic_core::report::ReporterExt;
use heretic_engine::ModuleIo;
use heretic_optim::Trial;
use heretic_optim::lbfgs::{Lbfgs, LbfgsConfig};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{default_bad_prompts, default_good_prompts, row_norms};
use crate::{Context, Modifier};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct Settings {
    /// Dataset of prompts that tend to produce desirable responses.
    pub good_prompts: DatasetSpecification,
    /// Dataset of prompts that tend to produce undesirable responses.
    pub bad_prompts: DatasetSpecification,
    /// Whether to renormalize the rows of the modified matrices to preserve the original
    /// matrices' row magnitudes. This is believed to improve intelligence retention.
    pub preserve_row_magnitudes: bool,
    /// The rank of the LoRA adapter to use.
    pub lora_rank: usize,
    /// Number of (outer) L-BFGS optimization steps to perform.
    pub n_optimization_steps: usize,
    /// Learning rate to use in the L-BFGS optimizer.
    pub learning_rate: f64,
    /// Maximum number of (inner) iterations to perform per (outer) L-BFGS optimization step.
    pub max_iter: usize,
    /// Number of past updates to store for approximating the Hessian matrix in the L-BFGS optimizer.
    pub history_size: usize,
    /// Whether to print the loss value for each L-BFGS optimization step.
    pub print_loss: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            good_prompts: default_good_prompts(),
            bad_prompts: default_bad_prompts(),
            preserve_row_magnitudes: true,
            lora_rank: 50,
            n_optimization_steps: 5,
            learning_rate: 1.0,
            max_iter: 20,
            history_size: 10,
            print_loss: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Parameters {
    pub start_layer_index: usize,
    pub end_layer_index: usize,
    pub preserve_good_behavior_weight: f64,
    pub steer_bad_behavior_weight: f64,
    pub overcorrect_relative_weight: f64,
    pub neighbor_count: usize,
}

pub struct Ara {
    pub settings: Settings,
    seed: u64,
    good_prompt_count: usize,
    bad_prompt_count: usize,
    good_module_io: ModuleIo,
    bad_module_io: ModuleIo,
}

/// Pairwise Euclidean distances, computed like `torch.cdist` for larger inputs
/// (`|a|² + |b|² - 2ab`). Distances below 1e-6 get a zero gradient, matching PyTorch's
/// special handling of zero distances.
fn cdist(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    const EPS: f64 = 1e-12;
    let a2 = a.sqr()?.sum_keepdim(1)?; // (n, 1)
    let b2 = b.sqr()?.sum_keepdim(1)?.t()?; // (1, m)
    let ab = a.matmul(&b.t()?)?; // (n, m)
    let d2 = (a2.broadcast_add(&b2)? - (ab * 2.0)?)?;
    Ok(((d2 - EPS)?.relu()? + EPS)?.sqrt()?)
}

/// For each row of `a`, the mean distance to its `k` nearest neighbors among the rows of `b`.
fn mean_distances_to_knn(a: &Tensor, b: &Tensor, k: usize) -> Result<Tensor> {
    let distances = cdist(a, b)?;
    let (n, m) = distances.dims2()?;
    let k = k.min(m).max(1);

    // Neighbor selection doesn't need gradients, so it is done on the CPU.
    let values: Vec<f32> = distances.detach().to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?;
    let mut indices = Vec::with_capacity(n * k);
    for row in values.chunks(m) {
        let mut order: Vec<u32> = (0..m as u32).collect();
        order.sort_by(|&x, &y| row[x as usize].total_cmp(&row[y as usize]));
        indices.extend_from_slice(&order[..k]);
    }
    let indices = Tensor::from_vec(indices, (n, k), distances.device())?;
    Ok(distances.gather(&indices, 1)?.mean(1)?)
}

/// The objective function at the heart of ARA.
fn ara_loss(
    good_output: &Tensor,
    bad_output: &Tensor,
    new_good_output: &Tensor,
    new_bad_output: &Tensor,
    parameters: &Parameters,
) -> Result<Tensor> {
    // The outputs for "good" prompts should change as little as possible.
    let preserve_good_behavior = (new_good_output - good_output)?.sqr()?.mean_all()?;

    // Pull the outputs for "bad" prompts towards the original outputs for "good" prompts,
    // and push them away from the original outputs for "bad" prompts. In combination,
    // this overcorrects away from the original residuals, which results in stronger
    // steering that can overcome more complex refusal mechanisms.
    let pull = mean_distances_to_knn(new_bad_output, good_output, parameters.neighbor_count)?.mean_all()?;
    let push = mean_distances_to_knn(new_bad_output, bad_output, parameters.neighbor_count)?.mean_all()?;
    let steer_bad_behavior = (pull - (push * parameters.overcorrect_relative_weight)?)?;

    Ok(((preserve_good_behavior * parameters.preserve_good_behavior_weight)?
        + (steer_bad_behavior * parameters.steer_bad_behavior_weight)?)?)
}

/// Module data needed by the optimization, on the adapter's device, as float32.
struct ModuleData {
    w_base: Tensor,
    w_row_norms: Tensor,
    bias: Option<Tensor>,
    good_input: Tensor,
    good_output: Tensor,
    bad_input: Tensor,
    bad_output: Tensor,
    good_product: Tensor,
    bad_product: Tensor,
}

impl Ara {
    pub fn new(settings: Settings, seed: u64) -> Self {
        Self {
            settings,
            seed,
            good_prompt_count: 0,
            bad_prompt_count: 0,
            good_module_io: Vec::new(),
            bad_module_io: Vec::new(),
        }
    }

    fn objective(&self, data: &ModuleData, a: &Tensor, b: &Tensor, parameters: &Parameters) -> Result<Tensor> {
        // input @ (W_base + B @ A).T = input @ W_base.T + (input @ A.T) @ B.T
        // The first term is precomputed, so only the low-rank term is recomputed.
        let mut new_good = (&data.good_product + data.good_input.matmul(&a.t()?)?.matmul(&b.t()?)?)?;
        let mut new_bad = (&data.bad_product + data.bad_input.matmul(&a.t()?)?.matmul(&b.t()?)?)?;

        if self.settings.preserve_row_magnitudes {
            // Row normalization rescales each output row by (original row norm / effective
            // row norm), which applies directly to the outputs as a per-column factor.
            let effective = row_norms(&(&data.w_base + b.matmul(a)?)?)?;
            let scale = (&data.w_row_norms / effective)?.t()?;
            new_good = new_good.broadcast_mul(&scale)?;
            new_bad = new_bad.broadcast_mul(&scale)?;
        }

        // The captured outputs include the bias, which must not be rescaled.
        if let Some(bias) = &data.bias {
            new_good = new_good.broadcast_add(bias)?;
            new_bad = new_bad.broadcast_add(bias)?;
        }

        ara_loss(&data.good_output, &data.bad_output, &new_good, &new_bad, parameters)
    }

    /// Optimizes the adapter of one module and returns (A, B).
    fn optimize_module(
        &self,
        data: &ModuleData,
        initial_a: Tensor,
        rank: usize,
        parameters: &Parameters,
        label: &str,
        ctx: &Context,
    ) -> Result<(Tensor, Tensor)> {
        let (d_out, d_in) = data.w_base.dims2()?;
        let device = data.w_base.device().clone();
        let n_a = rank * d_in;
        let n_b = d_out * rank;

        // B starts at zero (identity adapter); A is initialized randomly so that
        // gradients don't vanish.
        let b0 = Tensor::zeros(n_b, DType::F32, &device)?;
        let x0 = Tensor::cat(&[&initial_a.flatten_all()?, &b0], 0)?;

        let mut objective = |x: &Tensor| -> Result<(f64, Tensor)> {
            let var = Var::from_tensor(x)?;
            let flat = var.as_tensor();
            let a = flat.narrow(0, 0, n_a)?.reshape((rank, d_in))?;
            let b = flat.narrow(0, n_a, n_b)?.reshape((d_out, rank))?;
            let loss = self.objective(data, &a, &b, parameters)?;
            let grads = loss.backward()?;
            let grad = grads
                .get(flat)
                .cloned()
                .unwrap_or_else(|| flat.zeros_like().expect("zeros"));
            Ok((loss.to_dtype(DType::F64)?.to_scalar::<f64>()?, grad))
        };

        let mut optimizer = Lbfgs::new(
            x0,
            LbfgsConfig {
                lr: self.settings.learning_rate,
                max_iter: self.settings.max_iter,
                history_size: self.settings.history_size,
                ..LbfgsConfig::default()
            },
        );

        for step in 0..self.settings.n_optimization_steps {
            ctx.model.check_cancelled()?;
            let loss = optimizer.step(&mut objective)?;
            if self.settings.print_loss {
                ctx.reporter
                    .detail(format!("[{label}] Step: {}, Loss: {loss:.6}", step + 1));
            }
        }

        let x = optimizer.x;
        let a = x.narrow(0, 0, n_a)?.reshape((rank, d_in))?;
        let b = x.narrow(0, n_a, n_b)?.reshape((d_out, rank))?;
        Ok((a, b))
    }

    fn module_data(
        &self,
        layer: usize,
        component: &str,
        index: usize,
        module: &heretic_engine::LoraLinear,
    ) -> Result<ModuleData> {
        let device = module.weight().device().clone();
        let lookup = |io: &ModuleIo| -> Result<(Tensor, Tensor)> {
            let (input, output) = io
                .get(layer)
                .and_then(|l| l.get(component))
                .and_then(|c| c.get(&index))
                .ok_or_else(|| anyhow!("No module I/O for layer {layer}, {component}[{index}]"))?;
            Ok((
                input.to_dtype(DType::F32)?.to_device(&device)?,
                output.to_dtype(DType::F32)?.to_device(&device)?,
            ))
        };
        let (good_input, good_output) = lookup(&self.good_module_io)?;
        let (bad_input, bad_output) = lookup(&self.bad_module_io)?;

        let w_base = module.weight_f32()?;
        let w_row_norms = row_norms(&w_base)?.detach();
        // The base product doesn't depend on the adapter, so it is computed once, in float32
        // from the float32 weight (not taken from the captured output, which was computed
        // in the model's dtype).
        let good_product = good_input.matmul(&w_base.t()?)?;
        let bad_product = bad_input.matmul(&w_base.t()?)?;
        let bias = match module.bias() {
            Some(bias) => Some(bias.to_dtype(DType::F32)?.reshape((1, w_base.dim(0)?))?),
            None => None,
        };

        Ok(ModuleData {
            w_base,
            w_row_norms,
            bias,
            good_input,
            good_output,
            bad_input,
            bad_output,
            good_product,
            bad_product,
        })
    }
}

/// Kaiming-uniform initialization with a = sqrt(5) (PyTorch's default for linear layers):
/// U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
fn kaiming_uniform(rows: usize, cols: usize, rng: &mut ChaCha8Rng, device: &Device) -> Result<Tensor> {
    let bound = 1.0 / (cols as f32).sqrt();
    let data: Vec<f32> = (0..rows * cols).map(|_| rng.random_range(-bound..bound)).collect();
    Ok(Tensor::from_vec(data, (rows, cols), device)?)
}

impl Modifier for Ara {
    fn class_name(&self) -> &'static str {
        "ARA"
    }

    fn modifier_name(&self) -> String {
        "Arbitrary-Rank Ablation (ARA)".into()
    }

    fn init(&mut self, ctx: &Context) -> Result<()> {
        let reporter = ctx.reporter;
        reporter.heading(format!(
            "Loading good prompts from **{}**...",
            self.settings.good_prompts.describe()
        ));
        let good_prompts = ctx.load_prompts(&mut self.settings.good_prompts)?;
        reporter.info(format!("* **{}** prompts loaded", good_prompts.len()));

        reporter.heading(format!(
            "Loading bad prompts from **{}**...",
            self.settings.bad_prompts.describe()
        ));
        let bad_prompts = ctx.load_prompts(&mut self.settings.bad_prompts)?;
        reporter.info(format!("* **{}** prompts loaded", bad_prompts.len()));

        self.good_prompt_count = good_prompts.len();
        self.bad_prompt_count = bad_prompts.len();

        reporter.heading("Obtaining module I/O for good prompts...");
        self.good_module_io = ctx.model.get_module_io_batched(&good_prompts)?;
        reporter.heading("Obtaining module I/O for bad prompts...");
        self.bad_module_io = ctx.model.get_module_io_batched(&bad_prompts)?;

        ctx.model.apply_lora(self.settings.lora_rank);
        Ok(())
    }

    fn suggest_parameters(&self, ctx: &Context, trial: &mut Trial) -> Result<Value> {
        let layer_count = ctx.model.num_layers() as i64;
        let start_layer_index = trial.suggest_int("start_layer_index", 0, layer_count / 2)? as usize;
        // The end index is exclusive, so starting its range one above the start range's
        // upper bound ensures that the layer range is never empty.
        let end_layer_index = trial.suggest_int("end_layer_index", layer_count / 2 + 1, layer_count)? as usize;
        let preserve_good_behavior_weight = trial.suggest_float("preserve_good_behavior_weight", 0.0, 1.0)?;
        let steer_bad_behavior_weight = trial.suggest_log_float("steer_bad_behavior_weight", 0.0001, 1.0)?;
        let overcorrect_relative_weight = trial.suggest_float("overcorrect_relative_weight", 0.0, 1.3)?;
        // There can't be more neighbors than prompts.
        let max_neighbors = 15.min(self.good_prompt_count).min(self.bad_prompt_count).max(1) as i64;
        let neighbor_count = trial.suggest_int("neighbor_count", 1, max_neighbors)? as usize;

        Ok(serde_json::to_value(Parameters {
            start_layer_index,
            end_layer_index,
            preserve_good_behavior_weight,
            steer_bad_behavior_weight,
            overcorrect_relative_weight,
            neighbor_count,
        })?)
    }

    fn modify_model(&self, ctx: &Context, parameters: &Value) -> Result<()> {
        let parameters: Parameters = serde_json::from_value(parameters.clone()).context("Invalid ARA parameters")?;
        let rank = ctx.model.lora_rank().unwrap_or(self.settings.lora_rank);

        // A single generator, seeded per modification, keeps the initialization of A
        // independent of the device and of previous trials.
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        let end = parameters.end_layer_index.min(ctx.model.num_layers());
        for layer in parameters.start_layer_index..end {
            for (component, modules) in ctx.model.layer_modules(layer) {
                for (index, module) in modules.iter().enumerate() {
                    ctx.model.check_cancelled()?;
                    let data = self.module_data(layer, &component, index, module)?;
                    let initial_a = kaiming_uniform(rank, module.in_features(), &mut rng, module.weight().device())?;
                    let label = format!("{layer}/{component}/{index}");
                    let (a, b) = self.optimize_module(&data, initial_a, rank, &parameters, &label, ctx)?;
                    module.set_adapter(a, b)?;
                }
            }
        }
        Ok(())
    }

    fn present_parameters(&self, parameters: &Value) -> Result<Vec<(String, String)>> {
        let p: Parameters = serde_json::from_value(parameters.clone())?;
        Ok(vec![
            ("start_layer_index".into(), p.start_layer_index.to_string()),
            ("end_layer_index".into(), p.end_layer_index.to_string()),
            (
                "preserve_good_behavior_weight".into(),
                format!("{:.4}", p.preserve_good_behavior_weight),
            ),
            (
                "steer_bad_behavior_weight".into(),
                format!("{:.4}", p.steer_bad_behavior_weight),
            ),
            (
                "overcorrect_relative_weight".into(),
                format!("{:.4}", p.overcorrect_relative_weight),
            ),
            ("neighbor_count".into(), p.neighbor_count.to_string()),
        ])
    }

    fn dataset_specifications(&self) -> Vec<DatasetSpecification> {
        vec![self.settings.good_prompts.clone(), self.settings.bad_prompts.clone()]
    }

    fn settings_json(&self) -> Value {
        serde_json::to_value(&self.settings).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random(rows: usize, cols: usize, rng: &mut ChaCha8Rng) -> Tensor {
        let data: Vec<f32> = (0..rows * cols).map(|_| rng.random::<f32>() - 0.5).collect();
        Tensor::from_vec(data, (rows, cols), &Device::Cpu).unwrap()
    }

    #[test]
    fn cdist_matches_direct_computation() -> Result<()> {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let a = random(5, 7, &mut rng);
        let b = random(4, 7, &mut rng);
        let d = cdist(&a, &b)?.to_vec2::<f32>()?;
        let av = a.to_vec2::<f32>()?;
        let bv = b.to_vec2::<f32>()?;
        for i in 0..5 {
            for j in 0..4 {
                let direct: f32 = av[i]
                    .iter()
                    .zip(&bv[j])
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt();
                assert!((d[i][j] - direct).abs() < 1e-5);
            }
        }
        Ok(())
    }

    #[test]
    fn knn_gradient_is_finite_at_zero_distance() -> Result<()> {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let b = random(6, 4, &mut rng);
        let var = Var::from_tensor(&b)?;
        // Distances of the points to themselves are zero.
        let loss = mean_distances_to_knn(var.as_tensor(), &b, 2)?.mean_all()?;
        let grads = loss.backward()?;
        let grad: Vec<f32> = grads.get(var.as_tensor()).unwrap().flatten_all()?.to_vec1()?;
        assert!(grad.iter().all(|g| g.is_finite()));
        Ok(())
    }

    #[test]
    fn optimization_steers_bad_outputs() -> Result<()> {
        // Synthetic module: good inputs and bad inputs come from different clusters.
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let (d_in, d_out, n) = (12, 10, 16);
        let w = random(d_out, d_in, &mut rng);
        let good_input = random(n, d_in, &mut rng);
        let bad_input = (random(n, d_in, &mut rng) + 1.0)?;
        let good_output = good_input.matmul(&w.t()?)?;
        let bad_output = bad_input.matmul(&w.t()?)?;
        let data = ModuleData {
            w_row_norms: row_norms(&w)?,
            w_base: w.clone(),
            bias: None,
            good_product: good_output.clone(),
            bad_product: bad_output.clone(),
            good_input,
            good_output: good_output.clone(),
            bad_input,
            bad_output: bad_output.clone(),
        };

        let ara = Ara::new(
            Settings {
                lora_rank: 4,
                ..Settings::default()
            },
            0,
        );
        let parameters = Parameters {
            start_layer_index: 0,
            end_layer_index: 1,
            preserve_good_behavior_weight: 0.5,
            steer_bad_behavior_weight: 1.0,
            overcorrect_relative_weight: 0.5,
            neighbor_count: 3,
        };

        let a0 = kaiming_uniform(4, d_in, &mut rng, &Device::Cpu)?;
        let b0 = Tensor::zeros((d_out, 4), DType::F32, &Device::Cpu)?;
        let initial_loss: f32 = ara.objective(&data, &a0, &b0, &parameters)?.to_scalar()?;

        // Run the optimizer directly (without a model context).
        let n_a = 4 * d_in;
        let n_b = d_out * 4;
        let mut objective = |x: &Tensor| -> Result<(f64, Tensor)> {
            let var = Var::from_tensor(x)?;
            let flat = var.as_tensor();
            let a = flat.narrow(0, 0, n_a)?.reshape((4, d_in))?;
            let b = flat.narrow(0, n_a, n_b)?.reshape((d_out, 4))?;
            let loss = ara.objective(&data, &a, &b, &parameters)?;
            let grad = loss.backward()?.get(flat).unwrap().clone();
            Ok((loss.to_dtype(DType::F64)?.to_scalar::<f64>()?, grad))
        };
        let x0 = Tensor::cat(&[&a0.flatten_all()?, &b0.flatten_all()?], 0)?;
        let mut optimizer = Lbfgs::new(
            x0,
            LbfgsConfig {
                history_size: 10,
                ..LbfgsConfig::default()
            },
        );
        for _ in 0..5 {
            optimizer.step(&mut objective)?;
        }
        let (final_loss, _) = objective(&optimizer.x)?;
        assert!(
            (final_loss as f32) < initial_loss - 0.05,
            "{initial_loss} -> {final_loss}"
        );
        assert!(final_loss.is_finite());
        Ok(())
    }
}
