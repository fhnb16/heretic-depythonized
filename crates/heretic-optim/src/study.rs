// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Optimization studies with a persistent journal, modeled after Optuna's API.

use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, anyhow, bail};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::distributions::Distribution;
use crate::hypervolume::dominates;
use crate::tpe::{TpeConfig, intersection_search_space, sample_relative};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    Minimize,
    Maximize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialState {
    Complete,
    Pruned,
    Fail,
}

/// A finished trial.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrozenTrial {
    pub number: usize,
    pub state: TrialState,
    /// Parameter name -> (distribution, internal value).
    pub params: BTreeMap<String, (Distribution, f64)>,
    pub values: Option<Vec<f64>>,
    #[serde(default)]
    pub user_attrs: Map<String, Value>,
}

impl FrozenTrial {
    /// External parameter values.
    pub fn param_values(&self) -> BTreeMap<String, Value> {
        self.params
            .iter()
            .map(|(name, (d, v))| (name.clone(), d.to_external(*v)))
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum JournalEntry {
    Study { version: u32, directions: Vec<Direction> },
    StudyAttr { key: String, value: Value },
    Trial { trial: FrozenTrial },
}

const JOURNAL_VERSION: u32 = 1;

/// Directions, study attributes and trials stored in a journal.
pub type JournalContents = (Vec<Direction>, Map<String, Value>, Vec<FrozenTrial>);

/// An optimization study. If created with a journal path, every change is appended to
/// the journal immediately, so an interrupted run can be resumed.
pub struct Study {
    pub directions: Vec<Direction>,
    pub user_attrs: Map<String, Value>,
    trials: Vec<FrozenTrial>,
    pub config: TpeConfig,
    pub seed: u64,
    journal: Option<PathBuf>,
}

impl std::fmt::Debug for Study {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Study")
            .field("directions", &self.directions)
            .field("trials", &self.trials.len())
            .field("journal", &self.journal)
            .finish()
    }
}

impl Study {
    /// Creates an in-memory study.
    pub fn new(directions: Vec<Direction>, config: TpeConfig, seed: u64) -> Self {
        Self {
            directions,
            user_attrs: Map::new(),
            trials: Vec::new(),
            config,
            seed,
            journal: None,
        }
    }

    /// Creates a new study with a journal file (overwriting any existing file).
    pub fn create(path: &Path, directions: Vec<Direction>, config: TpeConfig, seed: u64) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, "").with_context(|| format!("Failed to create {}", path.display()))?;
        let mut study = Self::new(directions.clone(), config, seed);
        study.journal = Some(path.to_path_buf());
        study.append(&JournalEntry::Study {
            version: JOURNAL_VERSION,
            directions,
        })?;
        Ok(study)
    }

    /// Reads only the metadata of a journal (directions, attributes, trials) without
    /// attaching it, e.g. to list previous runs.
    pub fn peek(path: &Path) -> Result<JournalContents> {
        let file = std::fs::File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
        let mut directions = None;
        let mut attrs = Map::new();
        let mut trials = Vec::new();
        for (index, line) in std::io::BufReader::new(file).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let entry: JournalEntry = match serde_json::from_str(&line) {
                Ok(entry) => entry,
                // A partially written last line (e.g. after a crash) is ignored.
                Err(_) if index > 0 => continue,
                Err(error) => return Err(anyhow!("{} is not a Heretic study journal: {error}", path.display())),
            };
            match entry {
                JournalEntry::Study { version, directions: d } => {
                    if version > JOURNAL_VERSION {
                        bail!("{} was written by a newer version of Heretic", path.display());
                    }
                    directions = Some(d);
                }
                JournalEntry::StudyAttr { key, value } => {
                    attrs.insert(key, value);
                }
                JournalEntry::Trial { trial } => trials.push(trial),
            }
        }
        let directions = directions.ok_or_else(|| anyhow!("{} contains no study", path.display()))?;
        Ok((directions, attrs, trials))
    }

    /// Opens an existing journal.
    pub fn open(path: &Path, config: TpeConfig, seed: u64) -> Result<Self> {
        let (directions, user_attrs, trials) = Self::peek(path)?;
        Ok(Self {
            directions,
            user_attrs,
            trials,
            config,
            seed,
            journal: Some(path.to_path_buf()),
        })
    }

    fn append(&self, entry: &JournalEntry) -> Result<()> {
        if let Some(path) = &self.journal {
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(path)
                .with_context(|| format!("Failed to open {}", path.display()))?;
            let mut line = serde_json::to_string(entry)?;
            line.push('\n');
            file.write_all(line.as_bytes())?;
            file.flush()?;
        }
        Ok(())
    }

    pub fn journal_path(&self) -> Option<&Path> {
        self.journal.as_deref()
    }

    pub fn set_user_attr(&mut self, key: &str, value: Value) -> Result<()> {
        self.user_attrs.insert(key.to_string(), value.clone());
        self.append(&JournalEntry::StudyAttr {
            key: key.to_string(),
            value,
        })
    }

    pub fn trials(&self) -> &[FrozenTrial] {
        &self.trials
    }

    pub fn completed_trials(&self) -> impl Iterator<Item = &FrozenTrial> {
        self.trials.iter().filter(|t| t.state == TrialState::Complete)
    }

    /// Starts a new trial. Parameters for the joint search space of previous trials are
    /// sampled up front (multivariate TPE); others are sampled when suggested.
    pub fn ask(&mut self) -> Trial<'_> {
        let number = self.trials.len();
        // Deriving the RNG from the trial number makes sampling independent of whether
        // the study was resumed in between.
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed ^ (number as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let search_space = intersection_search_space(&self.trials);
        let relative = sample_relative(&self.config, &mut rng, &self.trials, &self.directions, &search_space);
        Trial {
            study: self,
            number,
            rng,
            relative,
            params: BTreeMap::new(),
            user_attrs: Map::new(),
        }
    }

    /// Records a trial whose parameters are given directly (e.g. when reproducing).
    pub fn add_trial(&mut self, trial: FrozenTrial) -> Result<()> {
        self.append(&JournalEntry::Trial { trial: trial.clone() })?;
        self.trials.push(trial);
        Ok(())
    }

    /// Non-dominated completed trials, in trial order.
    pub fn best_trials(&self) -> Vec<&FrozenTrial> {
        let signs: Vec<f64> = self
            .directions
            .iter()
            .map(|d| if *d == Direction::Minimize { 1.0 } else { -1.0 })
            .collect();
        let losses = |t: &FrozenTrial| -> Vec<f64> {
            t.values
                .as_deref()
                .unwrap_or_default()
                .iter()
                .zip(&signs)
                .map(|(v, s)| v * s)
                .collect()
        };

        let complete: Vec<&FrozenTrial> = self.completed_trials().collect();
        let loss_values: Vec<Vec<f64>> = complete.iter().map(|t| losses(t)).collect();
        complete
            .iter()
            .enumerate()
            .filter(|(i, _)| !loss_values.iter().any(|other| dominates(other, &loss_values[*i])))
            .map(|(_, t)| *t)
            .collect()
    }
}

/// A running trial.
pub struct Trial<'a> {
    study: &'a mut Study,
    pub number: usize,
    rng: ChaCha8Rng,
    relative: BTreeMap<String, f64>,
    params: BTreeMap<String, (Distribution, f64)>,
    user_attrs: Map<String, Value>,
}

impl Trial<'_> {
    fn suggest(&mut self, name: &str, distribution: Distribution) -> Result<f64> {
        distribution
            .validate()
            .with_context(|| format!("Invalid distribution for parameter {name}"))?;

        if let Some((existing, value)) = self.params.get(name) {
            if *existing != distribution {
                bail!("Parameter {name} was suggested twice with different distributions");
            }
            return Ok(*value);
        }

        let search_space = intersection_search_space(self.study.trials());
        let value = match self.relative.get(name) {
            Some(&value) if search_space.get(name) == Some(&distribution) && distribution.contains(value) => value,
            _ => distribution.sample_random(&mut self.rng),
        };

        self.params.insert(name.to_string(), (distribution, value));
        Ok(value)
    }

    pub fn suggest_float(&mut self, name: &str, low: f64, high: f64) -> Result<f64> {
        self.suggest(name, Distribution::float(low, high))
    }

    pub fn suggest_log_float(&mut self, name: &str, low: f64, high: f64) -> Result<f64> {
        self.suggest(name, Distribution::log_float(low, high))
    }

    pub fn suggest_int(&mut self, name: &str, low: i64, high: i64) -> Result<i64> {
        Ok(self.suggest(name, Distribution::int(low, high))?.round() as i64)
    }

    pub fn suggest_categorical(&mut self, name: &str, choices: &[&str]) -> Result<String> {
        let distribution = Distribution::categorical(choices.iter().copied());
        let index = self.suggest(name, distribution)? as usize;
        Ok(choices[index.min(choices.len() - 1)].to_string())
    }

    pub fn set_user_attr(&mut self, key: &str, value: Value) {
        self.user_attrs.insert(key.to_string(), value);
    }

    pub fn user_attr(&self, key: &str) -> Option<&Value> {
        self.user_attrs.get(key)
    }

    pub fn params(&self) -> &BTreeMap<String, (Distribution, f64)> {
        &self.params
    }

    /// Finishes the trial with objective values (one per study direction).
    pub fn complete(self, values: Vec<f64>) -> Result<FrozenTrial> {
        if values.len() != self.study.directions.len() {
            bail!(
                "Expected {} objective values, got {}",
                self.study.directions.len(),
                values.len()
            );
        }
        if values.iter().any(|v| v.is_nan()) {
            bail!("Objective values must not be NaN: {values:?}");
        }
        let trial = FrozenTrial {
            number: self.number,
            state: TrialState::Complete,
            params: self.params,
            values: Some(values),
            user_attrs: self.user_attrs,
        };
        self.study.add_trial(trial.clone())?;
        Ok(trial)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(study: &mut Study, n: usize, objective: impl Fn(&mut Trial) -> Vec<f64>) {
        for _ in 0..n {
            let mut trial = study.ask();
            let values = objective(&mut trial);
            trial.complete(values).unwrap();
        }
    }

    #[test]
    fn tpe_beats_random_on_a_quadratic() {
        let objective = |trial: &mut Trial| {
            let x = trial.suggest_float("x", -10.0, 10.0).unwrap();
            let y = trial.suggest_float("y", -10.0, 10.0).unwrap();
            let k = trial.suggest_int("k", 0, 10).unwrap() as f64;
            let c = trial.suggest_categorical("c", &["bad", "good"]).unwrap();
            vec![(x - 3.0).powi(2) + (y + 1.0).powi(2) + (k - 7.0).powi(2) + if c == "good" { 0.0 } else { 20.0 }]
        };

        let config = |startup| TpeConfig {
            n_startup_trials: startup,
            n_ei_candidates: 128,
            ..TpeConfig::default()
        };

        let mut tpe_total = 0.0;
        let mut random_total = 0.0;
        for seed in 0..5 {
            let mut tpe = Study::new(vec![Direction::Minimize], config(15), seed);
            run(&mut tpe, 80, objective);
            let mut random = Study::new(vec![Direction::Minimize], config(10_000), seed);
            run(&mut random, 80, objective);
            let best = |s: &Study| {
                s.completed_trials()
                    .map(|t| t.values.as_ref().unwrap()[0])
                    .fold(f64::INFINITY, f64::min)
            };
            tpe_total += best(&tpe);
            random_total += best(&random);
        }
        assert!(
            tpe_total < random_total * 0.5,
            "TPE {tpe_total} vs random {random_total}"
        );
    }

    #[test]
    fn multi_objective_front_improves() {
        // Classic trade-off: f1 = x^2, f2 = (x - 2)^2 plus a nuisance parameter.
        let objective = |trial: &mut Trial| {
            let x = trial.suggest_float("x", -5.0, 5.0).unwrap();
            let z = trial.suggest_float("z", 0.0, 1.0).unwrap();
            vec![x * x + z, (x - 2.0).powi(2) + z]
        };
        let mut study = Study::new(
            vec![Direction::Minimize, Direction::Minimize],
            TpeConfig {
                n_startup_trials: 10,
                n_ei_candidates: 128,
                ..TpeConfig::default()
            },
            42,
        );
        run(&mut study, 60, objective);
        let front = study.best_trials();
        assert!(front.len() >= 3);
        // Pareto-optimal points have x in [0, 2] and z near 0.
        let late_z: f64 = study.trials()[40..]
            .iter()
            .map(|t| t.param_values()["z"].as_f64().unwrap())
            .sum::<f64>()
            / 20.0;
        assert!(late_z < 0.35, "{late_z}");
    }

    #[test]
    fn maximization_direction() {
        let mut study = Study::new(vec![Direction::Maximize], TpeConfig::default(), 7);
        run(&mut study, 40, |trial| {
            vec![-(trial.suggest_float("x", 0.0, 1.0).unwrap() - 0.7).powi(2)]
        });
        let best = study.best_trials()[0].param_values()["x"].as_f64().unwrap();
        assert!((best - 0.7).abs() < 0.1, "{best}");
    }

    #[test]
    fn journal_round_trip_and_resume_determinism() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("study.jsonl");
        let objective = |trial: &mut Trial| vec![trial.suggest_float("x", 0.0, 1.0).unwrap()];

        let mut uninterrupted = Study::new(vec![Direction::Minimize], TpeConfig::default(), 5);
        run(&mut uninterrupted, 20, objective);

        let mut study = Study::create(&path, vec![Direction::Minimize], TpeConfig::default(), 5).unwrap();
        study.set_user_attr("finished", Value::Bool(false)).unwrap();
        run(&mut study, 12, objective);
        drop(study);

        let mut resumed = Study::open(&path, TpeConfig::default(), 5).unwrap();
        assert_eq!(resumed.trials().len(), 12);
        assert_eq!(resumed.user_attrs["finished"], Value::Bool(false));
        run(&mut resumed, 8, objective);

        let values = |s: &Study| s.trials().iter().map(|t| t.values.clone().unwrap()).collect::<Vec<_>>();
        assert_eq!(values(&resumed), values(&uninterrupted));

        // A truncated last line (crash during write) is tolerated.
        let mut text = std::fs::read_to_string(&path).unwrap();
        text.push_str("{\"event\":\"trial\",\"tri");
        std::fs::write(&path, text).unwrap();
        assert_eq!(Study::open(&path, TpeConfig::default(), 5).unwrap().trials().len(), 20);
    }

    #[test]
    fn pareto_front() {
        let mut study = Study::new(vec![Direction::Minimize, Direction::Maximize], TpeConfig::default(), 0);
        for (i, values) in [vec![1.0, 1.0], vec![2.0, 3.0], vec![1.5, 0.5], vec![0.5, 0.2]]
            .into_iter()
            .enumerate()
        {
            study
                .add_trial(FrozenTrial {
                    number: i,
                    state: TrialState::Complete,
                    params: BTreeMap::new(),
                    values: Some(values),
                    user_attrs: Map::new(),
                })
                .unwrap();
        }
        let front: Vec<usize> = study.best_trials().iter().map(|t| t.number).collect();
        assert_eq!(front, vec![0, 1, 3]);
    }
}
