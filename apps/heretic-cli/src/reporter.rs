// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Terminal rendering of messages and progress.

use std::sync::Mutex;

use console::style;
use heretic_core::report::{Level, Reporter};
use indicatif::{ProgressBar, ProgressStyle};

/// Renders `**bold**` spans as bold text.
pub fn render_markup(text: &str) -> String {
    let mut out = String::new();
    for (index, part) in text.split("**").enumerate() {
        if index % 2 == 1 {
            out.push_str(&style(part).bold().to_string());
        } else {
            out.push_str(part);
        }
    }
    out
}

/// Removes `**bold**` markers.
pub fn strip_markup(text: &str) -> String {
    text.replace("**", "")
}

pub struct TerminalReporter {
    pub debug: bool,
    progress: Mutex<Option<(String, ProgressBar)>>,
}

impl TerminalReporter {
    pub fn new(debug: bool) -> Self {
        Self {
            debug,
            progress: Mutex::new(None),
        }
    }

    fn clear_progress(&self) {
        if let Some((_, bar)) = self.progress.lock().unwrap().take() {
            bar.finish_and_clear();
        }
    }
}

impl Reporter for TerminalReporter {
    fn message(&self, level: Level, text: &str) {
        if level == Level::Debug && !self.debug {
            return;
        }
        self.clear_progress();
        let rendered = render_markup(text);
        match level {
            Level::Heading => {
                println!();
                println!("{rendered}");
            }
            Level::Info => println!("{rendered}"),
            Level::Detail | Level::Debug => println!("{}", style(strip_markup(text)).dim()),
            Level::Success => println!("{}", style(strip_markup(text)).green()),
            Level::Warning => println!("{}", style(strip_markup(text)).yellow()),
            Level::Error => println!("{}", style(strip_markup(text)).red()),
        }
    }

    fn progress(&self, task: &str, done: u64, total: Option<u64>) {
        let mut guard = self.progress.lock().unwrap();
        let needs_new = guard.as_ref().is_none_or(|(current, _)| current != task);
        if needs_new {
            if let Some((_, bar)) = guard.take() {
                bar.finish_and_clear();
            }
            let bar = match total {
                Some(total) => ProgressBar::new(total),
                None => ProgressBar::new_spinner(),
            };
            let template = if task.starts_with("Downloading") {
                "{msg} [{bar:30}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})"
            } else {
                "{msg} [{bar:30}] {pos}/{len}"
            };
            bar.set_style(
                ProgressStyle::with_template(template)
                    .unwrap_or_else(|_| ProgressStyle::default_bar())
                    .progress_chars("=> "),
            );
            bar.set_message(task.to_string());
            *guard = Some((task.to_string(), bar));
        }
        if let Some((_, bar)) = guard.as_ref() {
            if let Some(total) = total {
                bar.set_length(total);
            }
            bar.set_position(done);
            if total.is_some_and(|t| done >= t) {
                bar.finish_and_clear();
                *guard = None;
            }
        }
    }
}
