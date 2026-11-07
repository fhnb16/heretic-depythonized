// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Orchestration of a Heretic run, shared by the command line and the GUI.

pub mod events;
pub mod session;

pub use events::{EventSink, Phase, ScoreRecord, SessionEvent, TrialSummary, no_events};
pub use session::{CheckpointAction, CheckpointInfo, RunOutcome, Session};
