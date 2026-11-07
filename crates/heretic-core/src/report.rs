// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Reporting of messages and progress, shared by the CLI and the GUI.
//!
//! Message texts may contain `**bold**` spans, which frontends render as emphasis.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    /// A section heading, e.g. "Loading model...".
    Heading,
    /// A regular informational line.
    Info,
    /// Secondary information (timings, memory usage).
    Detail,
    Success,
    Warning,
    Error,
    /// Only shown when debug output is enabled.
    Debug,
}

pub trait Reporter: Send + Sync {
    fn message(&self, level: Level, text: &str);

    /// Progress of a long-running task. `total` is `None` if unknown.
    fn progress(&self, _task: &str, _done: u64, _total: Option<u64>) {}
}

/// Discards everything.
pub struct NullReporter;

impl Reporter for NullReporter {
    fn message(&self, _level: Level, _text: &str) {}
}

/// Prints plain text to stderr (used in tests and as a fallback).
pub struct StderrReporter;

impl Reporter for StderrReporter {
    fn message(&self, level: Level, text: &str) {
        let text = text.replace("**", "");
        match level {
            Level::Heading => eprintln!("\n{text}"),
            Level::Warning => eprintln!("WARNING: {text}"),
            Level::Error => eprintln!("ERROR: {text}"),
            _ => eprintln!("{text}"),
        }
    }
}

pub type SharedReporter = Arc<dyn Reporter>;

/// Convenience methods on shared reporters.
pub trait ReporterExt {
    fn heading(&self, text: impl AsRef<str>);
    fn info(&self, text: impl AsRef<str>);
    fn detail(&self, text: impl AsRef<str>);
    fn success(&self, text: impl AsRef<str>);
    fn warning(&self, text: impl AsRef<str>);
    fn error(&self, text: impl AsRef<str>);
    fn debug(&self, text: impl AsRef<str>);
}

impl<R: Reporter + ?Sized> ReporterExt for R {
    fn heading(&self, text: impl AsRef<str>) {
        self.message(Level::Heading, text.as_ref());
    }
    fn info(&self, text: impl AsRef<str>) {
        self.message(Level::Info, text.as_ref());
    }
    fn detail(&self, text: impl AsRef<str>) {
        self.message(Level::Detail, text.as_ref());
    }
    fn success(&self, text: impl AsRef<str>) {
        self.message(Level::Success, text.as_ref());
    }
    fn warning(&self, text: impl AsRef<str>) {
        self.message(Level::Warning, text.as_ref());
    }
    fn error(&self, text: impl AsRef<str>) {
        self.message(Level::Error, text.as_ref());
    }
    fn debug(&self, text: impl AsRef<str>) {
        self.message(Level::Debug, text.as_ref());
    }
}
