// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Parameter distributions, with Optuna's internal representation:
//! floats and ints are stored as their value, categoricals as the choice index.

use anyhow::{Result, bail};
use rand::RngExt;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Distribution {
    Float {
        low: f64,
        high: f64,
        #[serde(default)]
        log: bool,
        #[serde(default)]
        step: Option<f64>,
    },
    Int {
        low: i64,
        high: i64,
        #[serde(default)]
        log: bool,
        #[serde(default = "one")]
        step: i64,
    },
    Categorical {
        choices: Vec<Value>,
    },
}

fn one() -> i64 {
    1
}

impl Distribution {
    pub fn float(low: f64, high: f64) -> Self {
        Distribution::Float {
            low,
            high,
            log: false,
            step: None,
        }
    }

    pub fn log_float(low: f64, high: f64) -> Self {
        Distribution::Float {
            low,
            high,
            log: true,
            step: None,
        }
    }

    pub fn int(low: i64, high: i64) -> Self {
        Distribution::Int {
            low,
            high,
            log: false,
            step: 1,
        }
    }

    pub fn categorical<T: Into<Value>>(choices: impl IntoIterator<Item = T>) -> Self {
        Distribution::Categorical {
            choices: choices.into_iter().map(Into::into).collect(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Distribution::Float { low, high, log, step } => {
                if low.partial_cmp(high).is_none_or(|o| o.is_gt()) {
                    bail!("low ({low}) must be <= high ({high})");
                }
                if *log && *low <= 0.0 {
                    bail!("low must be > 0 for a log distribution");
                }
                if step.is_some_and(|s| s <= 0.0) {
                    bail!("step must be > 0");
                }
                if *log && step.is_some() {
                    bail!("log and step cannot be used together");
                }
            }
            Distribution::Int { low, high, log, step } => {
                if low > high {
                    bail!("low ({low}) must be <= high ({high})");
                }
                if *log && *low < 1 {
                    bail!("low must be >= 1 for a log distribution");
                }
                if *step <= 0 {
                    bail!("step must be > 0");
                }
            }
            Distribution::Categorical { choices } => {
                if choices.is_empty() {
                    bail!("categorical distributions need at least one choice");
                }
            }
        }
        Ok(())
    }

    /// Whether the distribution has only one possible value.
    pub fn single(&self) -> bool {
        match self {
            Distribution::Float { low, high, step, .. } => match step {
                Some(step) => (high - low) < *step,
                None => low == high,
            },
            Distribution::Int { low, high, step, .. } => high - low < *step,
            Distribution::Categorical { choices } => choices.len() == 1,
        }
    }

    /// Converts an internal value to the external value returned to the user.
    pub fn to_external(&self, internal: f64) -> Value {
        match self {
            Distribution::Float { .. } => serde_json::json!(internal),
            Distribution::Int { .. } => serde_json::json!(internal.round() as i64),
            Distribution::Categorical { choices } => choices[(internal as usize).min(choices.len() - 1)].clone(),
        }
    }

    pub fn contains(&self, internal: f64) -> bool {
        match self {
            Distribution::Float { low, high, .. } => (*low..=*high).contains(&internal),
            Distribution::Int { low, high, .. } => (*low as f64..=*high as f64).contains(&internal),
            Distribution::Categorical { choices } => internal >= 0.0 && (internal as usize) < choices.len(),
        }
    }

    /// Samples uniformly, like Optuna's `RandomSampler` (via `_SearchSpaceTransform`).
    pub fn sample_random(&self, rng: &mut ChaCha8Rng) -> f64 {
        match self {
            Distribution::Float { low, high, log, step } => {
                if low == high {
                    return *low;
                }
                match (log, step) {
                    (true, _) => rng.random_range(low.ln()..high.ln()).exp().clamp(*low, *high),
                    (false, Some(step)) => {
                        let x = rng.random_range((low - step / 2.0)..(high + step / 2.0));
                        let snapped = low + ((x - low) / step).round() * step;
                        snapped.clamp(*low, *high)
                    }
                    (false, None) => rng.random_range(*low..*high),
                }
            }
            Distribution::Int { low, high, log, step } => {
                if low == high {
                    return *low as f64;
                }
                let x = if *log {
                    rng.random_range((*low as f64 - 0.5).ln()..(*high as f64 + 0.5).ln())
                        .exp()
                } else {
                    rng.random_range((*low as f64 - 0.5)..(*high as f64 + 0.5))
                };
                let snapped = *low as f64 + ((x - *low as f64) / *step as f64).round() * *step as f64;
                snapped.clamp(*low as f64, *high as f64)
            }
            Distribution::Categorical { choices } => rng.random_range(0..choices.len()) as f64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn random_samples_stay_in_range() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let distributions = [
            Distribution::float(0.4, 0.9),
            Distribution::log_float(0.0001, 1.0),
            Distribution::int(1, 15),
            Distribution::categorical(["global", "per layer"]),
        ];
        let mut seen_ends = [false; 2];
        for distribution in &distributions {
            for _ in 0..2000 {
                let x = distribution.sample_random(&mut rng);
                assert!(distribution.contains(x), "{distribution:?}: {x}");
                if let Distribution::Int { .. } = distribution {
                    assert_eq!(x, x.round());
                    if x == 1.0 {
                        seen_ends[0] = true;
                    }
                    if x == 15.0 {
                        seen_ends[1] = true;
                    }
                }
            }
        }
        assert_eq!(seen_ends, [true, true]);
    }

    #[test]
    fn external_representation() {
        let categorical = Distribution::categorical(["global", "per layer"]);
        assert_eq!(categorical.to_external(1.0), serde_json::json!("per layer"));
        assert_eq!(Distribution::int(0, 5).to_external(3.0), serde_json::json!(3));
        assert!(Distribution::int(2, 2).single());
        assert!(!Distribution::float(0.0, 1.0).single());
    }

    #[test]
    fn serialization() {
        let d = Distribution::log_float(0.0001, 1.0);
        let json = serde_json::to_string(&d).unwrap();
        assert_eq!(serde_json::from_str::<Distribution>(&json).unwrap(), d);
    }
}
