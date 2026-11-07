// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Structured events emitted by a session (in addition to log messages), used by
//! frontends to show progress, charts and tables.

use std::sync::Arc;

use heretic_plugins::Score;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    LoadingModel,
    BatchSize,
    ResponsePrefix,
    Scorers,
    Modifier,
    Optimizing,
    Finished,
    RestoringTrial,
    Exporting,
}

/// A score together with the baseline score of the original model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreRecord {
    pub name: String,
    pub score: Score,
    pub baseline: Score,
}

/// Summary of a completed trial.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrialSummary {
    /// Study trial number (0-based).
    pub number: usize,
    /// 1-based index shown to the user.
    pub index: usize,
    /// Objective values, aligned with the objective names of the session.
    pub values: Vec<f64>,
    pub scores: Vec<ScoreRecord>,
    /// Parameters formatted for presentation.
    pub parameters: Vec<(String, String)>,
    /// Whether the trial is Pareto-optimal.
    pub pareto_optimal: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEvent {
    Phase {
        phase: Phase,
    },
    Baseline {
        scores: Vec<(String, Score)>,
        objective_names: Vec<String>,
    },
    TrialStarted {
        index: usize,
        total: usize,
        parameters: Vec<(String, String)>,
    },
    TrialFinished {
        trial: TrialSummary,
        /// Numbers of all currently Pareto-optimal trials.
        pareto: Vec<usize>,
        total: usize,
        elapsed_seconds: f64,
        remaining_seconds: Option<f64>,
    },
    /// The trial was interrupted and discarded.
    TrialCancelled {
        index: usize,
    },
}

pub type EventSink = Arc<dyn Fn(&SessionEvent) + Send + Sync>;

/// An event sink that discards everything.
pub fn no_events() -> EventSink {
    Arc::new(|_| {})
}
