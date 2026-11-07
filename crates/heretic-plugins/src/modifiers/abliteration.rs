// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Directional ablation ("abliteration", Arditi et al. 2024) with a flexible ablation
//! weight kernel over the layers, projected abliteration and norm preservation (MPOA).

use std::collections::BTreeMap;

use anyhow::{Context as _, Result, anyhow};
use candle_core::{DType, IndexOp, Tensor};
use heretic_core::dataset::DatasetSpecification;
use heretic_core::report::ReporterExt;
use heretic_optim::Trial;
use heretic_optim::lowrank::svd_lowrank;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{default_bad_prompts, default_good_prompts, normalize, row_norms};
use crate::{Context, Modifier};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RowNormalization {
    /// No normalization.
    None,
    /// Compute the LoRA adapter relative to row-normalized weights.
    Pre,
    /// Like "pre", but renormalizes to preserve original row magnitudes.
    Full,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct Settings {
    /// Dataset of prompts that tend to produce desirable responses.
    pub good_prompts: DatasetSpecification,
    /// Dataset of prompts that tend to produce undesirable responses.
    pub bad_prompts: DatasetSpecification,
    /// Whether to adjust the residual directions so that only the component that is
    /// orthogonal to the good direction is subtracted during abliteration.
    pub orthogonalize_direction: bool,
    /// How to apply row normalization of the weights: "none", "pre" or "full".
    pub row_normalization: RowNormalization,
    /// The rank of the LoRA adapter to use when "full" row normalization is used.
    /// Row magnitude preservation is approximate due to non-linear effects, and this
    /// determines the rank of that approximation.
    pub full_normalization_lora_rank: usize,
    /// The symmetric winsorization to apply to the per-prompt, per-layer residual vectors,
    /// expressed as the quantile to clamp to (between 0 and 1). Disabled by default.
    pub winsorization_quantile: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            good_prompts: default_good_prompts(),
            bad_prompts: default_bad_prompts(),
            orthogonalize_direction: true,
            row_normalization: RowNormalization::Full,
            full_normalization_lora_rank: 3,
            winsorization_quantile: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WeightDistribution {
    pub max_weight: f64,
    pub max_weight_position: f64,
    pub min_weight: f64,
    pub min_weight_distance: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Parameters {
    /// Fractional layer index of the global direction, or `None` for per-layer directions.
    pub direction_index: Option<f64>,
    pub weight_distributions: BTreeMap<String, WeightDistribution>,
}

pub struct Abliteration {
    pub settings: Settings,
    seed: u64,
    /// Shape (layers + 1, hidden), float32, CPU. Index 0 is the embedding direction.
    residual_directions: Option<Tensor>,
    lora_rank: usize,
}

impl Abliteration {
    pub fn new(settings: Settings, seed: u64) -> Self {
        let lora_rank = if settings.row_normalization == RowNormalization::Full {
            settings.full_normalization_lora_rank.max(1)
        } else {
            // Rank 1 is sufficient for directional ablation without renormalization.
            1
        };
        Self {
            settings,
            seed,
            residual_directions: None,
            lora_rank,
        }
    }

    /// Computes the (normalized) refusal directions from the mean residuals.
    pub fn compute_directions(&self, good_means: &Tensor, bad_means: &Tensor) -> Result<Tensor> {
        let mut directions = normalize(&(bad_means - good_means)?, 1)?;
        if self.settings.orthogonalize_direction {
            // Implements https://huggingface.co/blog/grimjim/projected-abliteration:
            // only the component orthogonal to the good direction is subtracted.
            let good_directions = normalize(good_means, 1)?;
            let projection = (&directions * &good_directions)?.sum_keepdim(1)?;
            directions = (directions - projection.broadcast_mul(&good_directions)?)?;
            directions = normalize(&directions, 1)?;
        }
        Ok(directions)
    }

    fn directions(&self) -> Result<&Tensor> {
        self.residual_directions
            .as_ref()
            .ok_or_else(|| anyhow!("Abliteration modifier was not initialized"))
    }

    /// Computes the LoRA factors (A, B) that ablate direction `v` from weight `w` with strength `weight`.
    pub fn ablation_adapter(&self, w: &Tensor, v: &Tensor, weight: f64) -> Result<(Tensor, Tensor)> {
        // LoRA abliteration: delta W = -lambda * v * (v^T W), with
        // lora_B = -lambda * v and lora_A = v^T W.
        let w_org = w.clone();
        let mut w = w.clone();
        let norms = row_norms(&w)?;
        if self.settings.row_normalization != RowNormalization::None {
            w = normalize(&w, 1)?;
        }

        let v_row = v.reshape((1, v.elem_count()))?;
        let mut lora_a = v_row.matmul(&w)?; // (1, d_in)
        let mut lora_b = (v.reshape((v.elem_count(), 1))? * -weight)?; // (d_out, 1)

        match self.settings.row_normalization {
            RowNormalization::None => {}
            RowNormalization::Pre => {
                // Make the adapter apply to the original weight matrix.
                lora_b = norms.broadcast_mul(&lora_b)?;
            }
            RowNormalization::Full => {
                // Approximates https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
                let adjusted = (w + lora_b.matmul(&lora_a)?)?;
                let renormalized = normalize(&adjusted, 1)?.broadcast_mul(&norms)?;
                let delta = (renormalized - w_org)?;

                let r = self.lora_rank;
                // svd_lowrank is randomized; reseeding immediately before the call makes
                // restoring a trial independent of RNG history.
                let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
                let (u, s, v_mat) = svd_lowrank(&delta, 2 * r + 4, 6, &mut rng)?;
                let r = r.min(s.len());
                let u = u.narrow(1, 0, r)?;
                let vh = v_mat.narrow(1, 0, r)?.t()?.contiguous()?;
                // Split the singular values evenly between the factors for numerical stability.
                let sqrt_s: Vec<f32> = s[..r].iter().map(|x| x.sqrt()).collect();
                let sqrt_s = Tensor::from_vec(sqrt_s, r, u.device())?;
                lora_b = u.broadcast_mul(&sqrt_s.reshape((1, r))?)?;
                lora_a = vh.broadcast_mul(&sqrt_s.reshape((r, 1))?)?;
            }
        }

        Ok((lora_a, lora_b))
    }
}

impl Modifier for Abliteration {
    fn class_name(&self) -> &'static str {
        "Abliteration"
    }

    fn modifier_name(&self) -> String {
        match (self.settings.orthogonalize_direction, self.settings.row_normalization) {
            (true, RowNormalization::Full) => "Magnitude-Preserving Orthogonal Ablation (MPOA)".into(),
            (true, _) => "Projected Abliteration".into(),
            _ => "Abliteration".into(),
        }
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

        reporter.heading("Calculating per-layer residual directions...");
        reporter.info("* Obtaining residual mean for good prompts...");
        let good_means = ctx
            .model
            .get_residuals_mean(&good_prompts, self.settings.winsorization_quantile)?;
        reporter.info("* Obtaining residual mean for bad prompts...");
        let bad_means = ctx
            .model
            .get_residuals_mean(&bad_prompts, self.settings.winsorization_quantile)?;

        self.residual_directions = Some(self.compute_directions(&good_means, &bad_means)?);
        ctx.model.apply_lora(self.lora_rank);
        Ok(())
    }

    fn suggest_parameters(&self, ctx: &Context, trial: &mut Trial) -> Result<Value> {
        let last_layer_index = (ctx.model.num_layers() - 1) as f64;

        let direction_scope = trial.suggest_categorical("direction_scope", &["global", "per layer"])?;

        // Discrimination between "harmful" and "harmless" inputs is usually strongest in
        // layers slightly past the midpoint of the layer stack. This parameter is always
        // sampled because multivariate TPE doesn't work with conditional parameters.
        let direction_index = trial.suggest_float("direction_index", 0.4 * last_layer_index, 0.9 * last_layer_index)?;
        let direction_index = (direction_scope != "per layer").then_some(direction_index);

        let mut weight_distributions = BTreeMap::new();
        for component in ctx.model.modifiable_components() {
            // The MLP gets a negative lower bound that is then clamped to 0, so the optimizer
            // can fully disable its ablation, which is often best for model intelligence.
            let lower_bound = if component == heretic_engine::MLP_DOWN_PROJ {
                -0.25
            } else {
                0.8
            };
            let max_weight = trial
                .suggest_float(&format!("{component}.max_weight"), lower_bound, 1.5)?
                .max(0.0);
            let max_weight_position = trial.suggest_float(
                &format!("{component}.max_weight_position"),
                0.6 * last_layer_index,
                last_layer_index,
            )?;
            // min_weight is sampled as a fraction of max_weight because multivariate TPE
            // doesn't support variable-range parameters.
            let min_weight = trial.suggest_float(&format!("{component}.min_weight"), 0.0, 1.0)?;
            let min_weight_distance = trial.suggest_float(
                &format!("{component}.min_weight_distance"),
                1.0,
                (0.6 * last_layer_index).max(1.0),
            )?;

            weight_distributions.insert(
                component,
                WeightDistribution {
                    max_weight,
                    max_weight_position,
                    min_weight: min_weight * max_weight,
                    min_weight_distance,
                },
            );
        }

        Ok(serde_json::to_value(Parameters {
            direction_index,
            weight_distributions,
        })?)
    }

    fn modify_model(&self, ctx: &Context, parameters: &Value) -> Result<()> {
        let parameters: Parameters =
            serde_json::from_value(parameters.clone()).context("Invalid abliteration parameters")?;
        let directions = self.directions()?;
        let device = &ctx.model.device;

        let global_direction = match parameters.direction_index {
            None => None,
            Some(index) => {
                // Shifted by 1 because the first direction is that of the embeddings.
                let position = index + 1.0;
                let whole = position.trunc() as usize;
                let fraction = position.fract();
                let a = directions.i(whole)?;
                let b = directions.i(whole + 1)?;
                let lerped = (&a + ((b - &a)? * fraction)?)?;
                Some(normalize(&lerped.unsqueeze(0)?, 1)?.squeeze(0)?)
            }
        };

        for layer in 0..ctx.model.num_layers() {
            ctx.model.check_cancelled()?;
            for (component, modules) in ctx.model.layer_modules(layer) {
                let distribution = parameters
                    .weight_distributions
                    .get(&component)
                    .ok_or_else(|| anyhow!("No weight distribution for component {component}"))?;

                let distance = (layer as f64 - distribution.max_weight_position).abs();
                // Don't modify layers that are more than min_weight_distance away from max_weight_position.
                if distance > distribution.min_weight_distance {
                    continue;
                }

                // Interpolate linearly between max_weight and min_weight over min_weight_distance.
                let weight = distribution.max_weight
                    + (distance / distribution.min_weight_distance)
                        * (distribution.min_weight - distribution.max_weight);

                // A weight of 0 disables this component's ablation.
                if weight == 0.0 {
                    continue;
                }

                let direction = match &global_direction {
                    Some(direction) => direction.clone(),
                    None => directions.i(layer + 1)?,
                };
                let v = direction.to_device(device)?.to_dtype(DType::F32)?;

                for module in modules {
                    let w = module.weight_f32()?;
                    let (a, b) = self.ablation_adapter(&w, &v, weight)?;
                    module.set_adapter(a, b)?;
                }
            }
        }
        Ok(())
    }

    fn present_parameters(&self, parameters: &Value) -> Result<Vec<(String, String)>> {
        let parameters: Parameters = serde_json::from_value(parameters.clone())?;
        let mut presented = vec![(
            "direction_index".to_string(),
            match parameters.direction_index {
                None => "per layer".to_string(),
                Some(index) => format!("{index:.2}"),
            },
        )];
        for (component, d) in &parameters.weight_distributions {
            presented.push((format!("{component}.max_weight"), format!("{:.2}", d.max_weight)));
            presented.push((
                format!("{component}.max_weight_position"),
                format!("{:.2}", d.max_weight_position),
            ));
            presented.push((format!("{component}.min_weight"), format!("{:.2}", d.min_weight)));
            presented.push((
                format!("{component}.min_weight_distance"),
                format!("{:.2}", d.min_weight_distance),
            ));
        }
        Ok(presented)
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
    use candle_core::Device;

    fn random(rows: usize, cols: usize, seed: u64) -> Tensor {
        use rand::RngExt;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let data: Vec<f32> = (0..rows * cols).map(|_| rng.random::<f32>() - 0.5).collect();
        Tensor::from_vec(data, (rows, cols), &Device::Cpu).unwrap()
    }

    fn modifier(row_normalization: RowNormalization) -> Abliteration {
        Abliteration::new(
            Settings {
                row_normalization,
                ..Settings::default()
            },
            1234,
        )
    }

    #[test]
    fn plain_ablation_removes_direction() -> Result<()> {
        let w = random(16, 8, 0);
        let v = normalize(&random(1, 16, 1), 1)?.squeeze(0)?;
        let (a, b) = modifier(RowNormalization::None).ablation_adapter(&w, &v, 1.0)?;
        let modified = (&w + b.matmul(&a)?)?;
        // The output of the modified matrix has no component along v.
        let projection = v.reshape((1, 16))?.matmul(&modified)?;
        let max: f32 = projection.abs()?.max_all()?.to_scalar()?;
        assert!(max < 1e-5, "{max}");
        Ok(())
    }

    /// Maximum deviation of the row norm ratios from 1 after applying an adapter.
    fn norm_deviation(w: &Tensor, a: &Tensor, b: &Tensor) -> Result<f32> {
        let modified = (w + b.matmul(a)?)?;
        let ratio = (row_norms(&modified)? / row_norms(w)?)?;
        Ok((ratio - 1.0)?.abs()?.max_all()?.to_scalar()?)
    }

    #[test]
    fn full_normalization_preserves_row_norms() -> Result<()> {
        let w = random(32, 24, 2);
        let v = normalize(&random(1, 32, 3), 1)?.squeeze(0)?;

        let m = modifier(RowNormalization::Full);
        let (a, b) = m.ablation_adapter(&w, &v, 1.0)?;
        assert_eq!(a.dims(), &[3, 24]);
        assert_eq!(b.dims(), &[32, 3]);

        // With full rank, the low-rank approximation is exact and norms are preserved.
        let exact = Abliteration::new(
            Settings {
                full_normalization_lora_rank: 24,
                ..Settings::default()
            },
            1234,
        );
        let (a_full, b_full) = exact.ablation_adapter(&w, &v, 1.0)?;
        assert!(norm_deviation(&w, &a_full, &b_full)? < 1e-3);

        // The rank-3 approximation is much closer than plain ablation.
        let (a_plain, b_plain) = modifier(RowNormalization::None).ablation_adapter(&w, &v, 1.0)?;
        assert!(norm_deviation(&w, &a, &b)? < norm_deviation(&w, &a_plain, &b_plain)?);

        // Deterministic: the same inputs give the same adapter.
        let (a2, _) = m.ablation_adapter(&w, &v, 1.0)?;
        assert_eq!(a.to_vec2::<f32>()?, a2.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn directions_are_normalized_and_orthogonal() -> Result<()> {
        let good = random(4, 10, 4);
        let bad = random(4, 10, 5);
        let directions = modifier(RowNormalization::Full).compute_directions(&good, &bad)?;
        let norms: Vec<f32> = row_norms(&directions)?.flatten_all()?.to_vec1()?;
        assert!(norms.iter().all(|n| (n - 1.0).abs() < 1e-5));
        let good_directions = normalize(&good, 1)?;
        let dots: Vec<f32> = (&directions * &good_directions)?.sum(1)?.to_vec1()?;
        assert!(dots.iter().all(|d| d.abs() < 1e-5), "{dots:?}");
        Ok(())
    }

    #[test]
    fn parameter_presentation() -> Result<()> {
        let mut distributions = BTreeMap::new();
        distributions.insert(
            "attn.o_proj".to_string(),
            WeightDistribution {
                max_weight: 1.0,
                max_weight_position: 10.0,
                min_weight: 0.5,
                min_weight_distance: 4.0,
            },
        );
        let parameters = serde_json::to_value(Parameters {
            direction_index: None,
            weight_distributions: distributions,
        })?;
        let presented = modifier(RowNormalization::Full).present_parameters(&parameters)?;
        assert_eq!(presented[0], ("direction_index".into(), "per layer".into()));
        assert_eq!(presented[1], ("attn.o_proj.max_weight".into(), "1.00".into()));
        Ok(())
    }
}
