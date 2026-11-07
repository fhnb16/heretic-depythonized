// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Tree-structured Parzen Estimator, ported from Optuna's `TPESampler` with
//! `multivariate=True` (including multi-objective support, formerly MOTPE).

use std::collections::BTreeMap;

use rand::RngExt;
use rand_chacha::ChaCha8Rng;

use crate::distributions::Distribution;
use crate::hypervolume::{hypervolume, non_domination_ranks, reference_point, solve_hssp};
use crate::special::{log_gauss_mass, truncnorm_logpdf, truncnorm_ppf};
use crate::study::{Direction, FrozenTrial, TrialState};

#[derive(Debug, Clone)]
pub struct TpeConfig {
    /// Number of initial trials that use random sampling.
    pub n_startup_trials: usize,
    /// Number of candidates sampled from the "good" estimator to choose from.
    pub n_ei_candidates: usize,
    pub prior_weight: f64,
    pub consider_prior: bool,
    pub consider_magic_clip: bool,
}

impl Default for TpeConfig {
    fn default() -> Self {
        Self {
            n_startup_trials: 10,
            n_ei_candidates: 24,
            prior_weight: 1.0,
            consider_prior: true,
            consider_magic_clip: true,
        }
    }
}

/// Optuna's `default_gamma`: number of trials in the "good" group.
pub fn default_gamma(n: usize) -> usize {
    ((0.1 * n as f64).ceil() as usize).min(25)
}

/// Optuna's `default_weights`: older observations get lower weights once there are more than 25.
pub fn default_weights(n: usize) -> Vec<f64> {
    if n == 0 {
        Vec::new()
    } else if n < 25 {
        vec![1.0; n]
    } else {
        let ramp_count = n - 25;
        let mut weights: Vec<f64> = if ramp_count == 1 {
            vec![1.0 / n as f64]
        } else {
            let start = 1.0 / n as f64;
            (0..ramp_count)
                .map(|i| start + (1.0 - start) * i as f64 / (ramp_count - 1) as f64)
                .collect()
        };
        weights.extend(std::iter::repeat_n(1.0, 25));
        weights
    }
}

#[derive(Debug, Clone)]
enum Kernel {
    Categorical {
        /// (n_kernels, n_choices), rows sum to 1.
        weights: Vec<Vec<f64>>,
    },
    TruncNorm {
        mus: Vec<f64>,
        sigmas: Vec<f64>,
        low: f64,
        high: f64,
    },
    DiscreteTruncNorm {
        mus: Vec<f64>,
        sigmas: Vec<f64>,
        low: f64,
        high: f64,
        step: f64,
    },
}

/// A mixture of product distributions over a search space.
#[derive(Debug, Clone)]
pub struct ParzenEstimator {
    weights: Vec<f64>,
    names: Vec<String>,
    distributions: Vec<Distribution>,
    kernels: Vec<Kernel>,
}

fn is_log(distribution: &Distribution) -> bool {
    matches!(
        distribution,
        Distribution::Float { log: true, .. } | Distribution::Int { log: true, .. }
    )
}

impl ParzenEstimator {
    /// Builds the estimator from observations (internal representation, in trial order).
    pub fn new(
        observations: &BTreeMap<String, Vec<f64>>,
        search_space: &BTreeMap<String, Distribution>,
        config: &TpeConfig,
        predetermined_weights: Option<Vec<f64>>,
    ) -> Self {
        let n_observations = observations.values().next().map_or(0, Vec::len);

        let mut weights = if n_observations == 0 {
            vec![1.0]
        } else {
            let mut w = predetermined_weights.unwrap_or_else(|| default_weights(n_observations));
            w.truncate(n_observations);
            if config.consider_prior {
                w.push(config.prior_weight);
            }
            w
        };
        let sum: f64 = weights.iter().sum();
        for w in weights.iter_mut() {
            *w /= sum;
        }

        let n_params = search_space.len();
        let mut names = Vec::new();
        let mut distributions = Vec::new();
        let mut kernels = Vec::new();

        for (name, distribution) in search_space {
            let raw = observations.get(name).cloned().unwrap_or_default();
            let transformed: Vec<f64> = if is_log(distribution) {
                raw.iter().map(|x| x.ln()).collect()
            } else {
                raw
            };

            let kernel = match distribution {
                Distribution::Categorical { choices } => {
                    let n_choices = choices.len();
                    if transformed.is_empty() {
                        Kernel::Categorical {
                            weights: vec![vec![1.0 / n_choices as f64; n_choices]],
                        }
                    } else {
                        let n_kernels = transformed.len() + usize::from(config.consider_prior);
                        let mut rows = vec![vec![config.prior_weight / n_kernels as f64; n_choices]; n_kernels];
                        for (row, &index) in rows.iter_mut().zip(&transformed) {
                            row[index as usize] += 1.0;
                        }
                        for row in rows.iter_mut() {
                            let total: f64 = row.iter().sum();
                            for v in row.iter_mut() {
                                *v /= total;
                            }
                        }
                        Kernel::Categorical { weights: rows }
                    }
                }
                Distribution::Float { low, high, log, step } => {
                    let (low, high, step) = if *log {
                        (low.ln(), high.ln(), None)
                    } else {
                        (*low, *high, *step)
                    };
                    numerical_kernel(&transformed, low, high, step, n_params, config)
                }
                Distribution::Int { low, high, log, step } => {
                    let (low, high, step) = if *log {
                        ((*low as f64).ln(), (*high as f64).ln(), None)
                    } else {
                        (*low as f64, *high as f64, Some(*step as f64))
                    };
                    numerical_kernel(&transformed, low, high, step, n_params, config)
                }
            };

            names.push(name.clone());
            distributions.push(distribution.clone());
            kernels.push(kernel);
        }

        Self {
            weights,
            names,
            distributions,
            kernels,
        }
    }

    /// Draws samples in the transformed space, shape (count, n_params).
    fn sample_transformed(&self, rng: &mut ChaCha8Rng, count: usize) -> Vec<Vec<f64>> {
        let cumulative: Vec<f64> = self
            .weights
            .iter()
            .scan(0.0, |acc, w| {
                *acc += w;
                Some(*acc)
            })
            .collect();
        let active: Vec<usize> = (0..count)
            .map(|_| {
                let r = rng.random::<f64>() * cumulative.last().copied().unwrap_or(1.0);
                cumulative.iter().position(|&c| r < c).unwrap_or(cumulative.len() - 1)
            })
            .collect();

        let mut samples = vec![vec![0.0; self.kernels.len()]; count];
        for (param, kernel) in self.kernels.iter().enumerate() {
            match kernel {
                Kernel::Categorical { weights } => {
                    for (sample, &k) in samples.iter_mut().zip(&active) {
                        let row = &weights[k.min(weights.len() - 1)];
                        let r = rng.random::<f64>();
                        let mut cumulative = 0.0;
                        let mut index = 0;
                        for (i, w) in row.iter().enumerate() {
                            cumulative += w;
                            // The last entry is treated as exactly 1 to avoid rounding errors.
                            if i == row.len() - 1 || cumulative >= r {
                                index = i;
                                break;
                            }
                        }
                        sample[param] = index as f64;
                    }
                }
                Kernel::TruncNorm { mus, sigmas, low, high } => {
                    for (sample, &k) in samples.iter_mut().zip(&active) {
                        let (mu, sigma) = (mus[k], sigmas[k]);
                        let q = rng.random::<f64>();
                        sample[param] = truncnorm_ppf(q, (low - mu) / sigma, (high - mu) / sigma) * sigma + mu;
                    }
                }
                Kernel::DiscreteTruncNorm {
                    mus,
                    sigmas,
                    low,
                    high,
                    step,
                } => {
                    for (sample, &k) in samples.iter_mut().zip(&active) {
                        let (mu, sigma) = (mus[k], sigmas[k]);
                        let q = rng.random::<f64>();
                        let x = truncnorm_ppf(q, (low - step / 2.0 - mu) / sigma, (high + step / 2.0 - mu) / sigma)
                            * sigma
                            + mu;
                        sample[param] = (low + ((x - low) / step).round_ties_even() * step).clamp(*low, *high);
                    }
                }
            }
        }
        samples
    }

    /// Log density of samples given in the transformed space.
    fn log_pdf_transformed(&self, samples: &[Vec<f64>]) -> Vec<f64> {
        let n_kernels = self.weights.len();
        let log_weights: Vec<f64> = self.weights.iter().map(|w| w.ln()).collect();

        samples
            .iter()
            .map(|sample| {
                let mut per_kernel = log_weights.clone();
                for (param, kernel) in self.kernels.iter().enumerate() {
                    let x = sample[param];
                    match kernel {
                        Kernel::Categorical { weights } => {
                            for (k, total) in per_kernel.iter_mut().enumerate() {
                                *total += weights[k.min(weights.len() - 1)][x as usize].ln();
                            }
                        }
                        Kernel::TruncNorm { mus, sigmas, low, high } => {
                            for k in 0..n_kernels {
                                let (mu, sigma) = (mus[k], sigmas[k]);
                                per_kernel[k] +=
                                    truncnorm_logpdf(x, (low - mu) / sigma, (high - mu) / sigma, mu, sigma);
                            }
                        }
                        Kernel::DiscreteTruncNorm {
                            mus,
                            sigmas,
                            low,
                            high,
                            step,
                        } => {
                            let lower_limit = low - step / 2.0;
                            let upper_limit = high + step / 2.0;
                            let x_lower = (x - step / 2.0).max(lower_limit);
                            let x_upper = (x + step / 2.0).min(upper_limit);
                            for k in 0..n_kernels {
                                let (mu, sigma) = (mus[k], sigmas[k]);
                                let mass = log_gauss_mass((x_lower - mu) / sigma, (x_upper - mu) / sigma);
                                let accept = log_gauss_mass((lower_limit - mu) / sigma, (upper_limit - mu) / sigma);
                                per_kernel[k] += mass - accept;
                            }
                        }
                    }
                }
                let max = per_kernel.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                // Avoid (-inf) - (-inf) when the probability is zero.
                let max = if max == f64::NEG_INFINITY { 0.0 } else { max };
                per_kernel.iter().map(|v| (v - max).exp()).sum::<f64>().ln() + max
            })
            .collect()
    }

    /// Converts transformed samples back to the internal representation.
    fn untransform(&self, sample: &[f64]) -> BTreeMap<String, f64> {
        self.names
            .iter()
            .zip(&self.distributions)
            .zip(sample)
            .map(|((name, distribution), &x)| {
                let value = match distribution {
                    Distribution::Float {
                        low, high, log: true, ..
                    } => x.exp().clamp(*low, *high),
                    Distribution::Int {
                        low, high, log: true, ..
                    } => x.exp().round().clamp(*low as f64, *high as f64),
                    _ => x,
                };
                (name.clone(), value)
            })
            .collect()
    }
}

fn numerical_kernel(
    observations: &[f64],
    low: f64,
    high: f64,
    step: Option<f64>,
    n_params: usize,
    config: &TpeConfig,
) -> Kernel {
    let step_or_zero = step.unwrap_or(0.0);
    let range = high - low + step_or_zero;
    let n = observations.len();

    // Multivariate TPE uses a fixed bandwidth (Scott's rule-like scaling).
    let sigma = 0.2 * (n.max(1) as f64).powf(-1.0 / (n_params as f64 + 4.0)) * range;
    let mut sigmas = vec![sigma; n];
    if config.consider_magic_clip {
        let max_sigma = range;
        let min_sigma = range / (1.0 + n as f64).min(100.0);
        for s in sigmas.iter_mut() {
            *s = s.clamp(min_sigma, max_sigma);
        }
    }

    let mut mus = observations.to_vec();
    if config.consider_prior {
        mus.push(0.5 * (low + high));
        sigmas.push(range);
    }

    match step {
        None => Kernel::TruncNorm { mus, sigmas, low, high },
        Some(step) => Kernel::DiscreteTruncNorm {
            mus,
            sigmas,
            low,
            high,
            step,
        },
    }
}

fn sign(direction: Direction) -> f64 {
    match direction {
        Direction::Minimize => 1.0,
        Direction::Maximize => -1.0,
    }
}

/// Objective values as losses (to be minimized).
fn losses(trial: &FrozenTrial, directions: &[Direction]) -> Vec<f64> {
    let values = trial.values.as_deref().unwrap_or_default();
    values.iter().zip(directions).map(|(v, d)| v * sign(*d)).collect()
}

/// Splits completed trials into the "good" (below) and "bad" (above) groups.
fn split_trials<'a>(
    trials: &[&'a FrozenTrial],
    directions: &[Direction],
    n_below: usize,
) -> (Vec<&'a FrozenTrial>, Vec<&'a FrozenTrial>) {
    let n_below = n_below.min(trials.len());

    if directions.len() <= 1 {
        let mut sorted: Vec<&FrozenTrial> = trials.to_vec();
        sorted.sort_by(|a, b| losses(a, directions)[0].total_cmp(&losses(b, directions)[0]));
        let above = sorted.split_off(n_below);
        return (sorted, above);
    }

    if n_below == 0 {
        return (Vec::new(), trials.to_vec());
    }

    let loss_values: Vec<Vec<f64>> = trials.iter().map(|t| losses(t, directions)).collect();
    let ranks = non_domination_ranks(&loss_values, Some(n_below));

    let mut below_indices: Vec<usize> = Vec::with_capacity(n_below);
    let mut rank = 0;
    loop {
        let rank_indices: Vec<usize> = (0..trials.len()).filter(|&i| ranks[i] == rank).collect();
        if rank_indices.is_empty() || below_indices.len() + rank_indices.len() > n_below {
            // Hypervolume subset selection within the rank that doesn't fit completely.
            let subset_size = n_below - below_indices.len();
            if subset_size > 0 && !rank_indices.is_empty() {
                let rank_losses: Vec<Vec<f64>> = rank_indices.iter().map(|&i| loss_values[i].clone()).collect();
                let reference = reference_point(&rank_losses);
                for selected in solve_hssp(&rank_losses, subset_size, &reference) {
                    below_indices.push(rank_indices[selected]);
                }
            }
            break;
        }
        below_indices.extend(rank_indices);
        if below_indices.len() == n_below {
            break;
        }
        rank += 1;
    }

    let mut below = Vec::new();
    let mut above = Vec::new();
    for (index, trial) in trials.iter().enumerate() {
        if below_indices.contains(&index) {
            below.push(*trial);
        } else {
            above.push(*trial);
        }
    }
    (below, above)
}

/// Weights of the "good" trials in multi-objective optimization: their (normalized)
/// contribution to the hypervolume of the group.
fn multi_objective_below_weights(below: &[&FrozenTrial], directions: &[Direction]) -> Vec<f64> {
    let n = below.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let loss_values: Vec<Vec<f64>> = below.iter().map(|t| losses(t, directions)).collect();
    let reference = reference_point(&loss_values);
    let total = hypervolume(&loss_values, &reference);
    let contributions: Vec<f64> = (0..n)
        .map(|i| {
            let others: Vec<Vec<f64>> = loss_values
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, v)| v.clone())
                .collect();
            total - hypervolume(&others, &reference)
        })
        .collect();
    let max = contributions.iter().copied().fold(0.0, f64::max).max(f64::EPSILON);
    contributions.iter().map(|c| (c / max).max(f64::EPSILON)).collect()
}

fn observations(trials: &[&FrozenTrial], search_space: &BTreeMap<String, Distribution>) -> BTreeMap<String, Vec<f64>> {
    search_space
        .keys()
        .map(|name| {
            let values = trials
                .iter()
                .map(|t| t.params.get(name).map_or(f64::NAN, |(_, v)| *v))
                .collect();
            (name.clone(), values)
        })
        .collect()
}

/// The search space shared by all completed trials (params with identical distributions),
/// excluding single-valued distributions, like Optuna's `IntersectionSearchSpace`.
pub fn intersection_search_space(trials: &[FrozenTrial]) -> BTreeMap<String, Distribution> {
    let mut space: Option<BTreeMap<String, Distribution>> = None;
    for trial in trials.iter().filter(|t| t.state == TrialState::Complete) {
        let trial_space: BTreeMap<String, Distribution> = trial
            .params
            .iter()
            .map(|(name, (d, _))| (name.clone(), d.clone()))
            .collect();
        space = Some(match space {
            None => trial_space,
            Some(current) => current
                .into_iter()
                .filter(|(name, d)| trial_space.get(name) == Some(d))
                .collect(),
        });
    }
    space
        .unwrap_or_default()
        .into_iter()
        .filter(|(_, d)| !d.single())
        .collect()
}

/// Samples values for the whole search space jointly. Returns an empty map during
/// the startup phase (parameters are then sampled randomly and independently).
pub fn sample_relative(
    config: &TpeConfig,
    rng: &mut ChaCha8Rng,
    trials: &[FrozenTrial],
    directions: &[Direction],
    search_space: &BTreeMap<String, Distribution>,
) -> BTreeMap<String, f64> {
    if search_space.is_empty() {
        return BTreeMap::new();
    }

    let complete: Vec<&FrozenTrial> = trials.iter().filter(|t| t.state == TrialState::Complete).collect();
    if complete.len() < config.n_startup_trials || complete.is_empty() {
        return BTreeMap::new();
    }

    let n_below = default_gamma(complete.len());
    let (below, above) = split_trials(&complete, directions, n_below);

    let below_weights = (directions.len() > 1).then(|| multi_objective_below_weights(&below, directions));
    let estimator_below =
        ParzenEstimator::new(&observations(&below, search_space), search_space, config, below_weights);
    let estimator_above = ParzenEstimator::new(&observations(&above, search_space), search_space, config, None);

    let candidates = estimator_below.sample_transformed(rng, config.n_ei_candidates.max(1));
    let log_below = estimator_below.log_pdf_transformed(&candidates);
    let log_above = estimator_above.log_pdf_transformed(&candidates);

    let mut best = 0;
    let mut best_score = f64::NEG_INFINITY;
    for (i, (b, a)) in log_below.iter().zip(&log_above).enumerate() {
        let score = b - a;
        if score > best_score {
            best_score = score;
            best = i;
        }
    }

    estimator_below.untransform(&candidates[best])
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn gamma_and_weights() {
        assert_eq!(default_gamma(1), 1);
        assert_eq!(default_gamma(30), 3);
        assert_eq!(default_gamma(1000), 25);
        assert_eq!(default_weights(3), vec![1.0; 3]);
        let w = default_weights(30);
        assert_eq!(w.len(), 30);
        assert!((w[0] - 1.0 / 30.0).abs() < 1e-12);
        assert_eq!(w[4], 1.0);
        assert!(w[0] < w[3]);
    }

    #[test]
    fn estimator_density_is_normalized() {
        let mut space = BTreeMap::new();
        space.insert("x".to_string(), Distribution::float(0.0, 1.0));
        let mut obs = BTreeMap::new();
        obs.insert("x".to_string(), vec![0.2, 0.25, 0.8]);
        let estimator = ParzenEstimator::new(&obs, &space, &TpeConfig::default(), None);

        let n = 10_000;
        let grid: Vec<Vec<f64>> = (0..n).map(|i| vec![(i as f64 + 0.5) / n as f64]).collect();
        let integral: f64 = estimator
            .log_pdf_transformed(&grid)
            .iter()
            .map(|l| l.exp() / n as f64)
            .sum();
        assert!((integral - 1.0).abs() < 1e-3, "{integral}");

        let mut rng = ChaCha8Rng::seed_from_u64(0);
        for sample in estimator.sample_transformed(&mut rng, 500) {
            assert!((0.0..=1.0).contains(&sample[0]));
        }
    }

    #[test]
    fn discrete_and_categorical_masses_sum_to_one() {
        let mut space = BTreeMap::new();
        space.insert("k".to_string(), Distribution::int(1, 5));
        space.insert("c".to_string(), Distribution::categorical(["a", "b", "c"]));
        let mut obs = BTreeMap::new();
        obs.insert("k".to_string(), vec![2.0, 2.0, 5.0]);
        obs.insert("c".to_string(), vec![0.0, 2.0, 2.0]);
        let estimator = ParzenEstimator::new(&obs, &space, &TpeConfig::default(), None);

        let mut points = Vec::new();
        for c in 0..3 {
            for k in 1..=5 {
                points.push(vec![c as f64, k as f64]); // BTreeMap order: "c" before "k"
            }
        }
        let total: f64 = estimator.log_pdf_transformed(&points).iter().map(|l| l.exp()).sum();
        assert!((total - 1.0).abs() < 1e-9, "{total}");

        let mut rng = ChaCha8Rng::seed_from_u64(3);
        for sample in estimator.sample_transformed(&mut rng, 300) {
            assert!(sample[0] == sample[0].round() && (0.0..3.0).contains(&sample[0]));
            assert!(sample[1] == sample[1].round() && (1.0..=5.0).contains(&sample[1]));
        }
    }
}
