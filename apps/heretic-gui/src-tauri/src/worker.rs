// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! The worker thread that owns the processing session.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use heretic_core::Settings;
use heretic_core::config::ExportStrategy;
use heretic_core::prompt::Message;
use heretic_core::report::{Level, Reporter, SharedReporter};
use heretic_session::{CheckpointAction, EventSink, RunOutcome, Session};
use serde_json::{Value, json};
use tauri::AppHandle;

use crate::emit;

pub enum Command {
    Start {
        settings: Box<Settings>,
        action: Option<CheckpointAction>,
    },
    RunMore {
        count: usize,
    },
    Trials,
    Restore {
        number: usize,
    },
    Chat {
        messages: Vec<Message>,
        use_adapters: bool,
    },
    Save {
        directory: PathBuf,
        strategy: ExportStrategy,
    },
    Upload {
        token: String,
        repo_id: String,
        private: bool,
        strategy: ExportStrategy,
    },
    SaveGguf {
        output: PathBuf,
        kind: heretic_export::gguf::GgufType,
        llama_cpp_dir: Option<PathBuf>,
    },
    ModelCard,
}

struct Envelope {
    command: Command,
    reply: Option<mpsc::Sender<Result<Value, String>>>,
}

#[derive(Default)]
struct Flags {
    stop: Option<Arc<AtomicBool>>,
    cancel: Option<Arc<AtomicBool>>,
    busy: bool,
}

pub struct Worker {
    sender: mpsc::Sender<Envelope>,
    flags: Arc<Mutex<Flags>>,
}

/// Forwards log messages and progress to the frontend.
struct GuiReporter {
    app: AppHandle,
}

impl Reporter for GuiReporter {
    fn message(&self, level: Level, text: &str) {
        emit(&self.app, "heretic://log", json!({"level": level, "text": text}));
    }

    fn progress(&self, task: &str, done: u64, total: Option<u64>) {
        emit(
            &self.app,
            "heretic://progress",
            json!({"task": task, "done": done, "total": total}),
        );
    }
}

fn status(app: &AppHandle, state: &str, message: Option<String>, outcome: Option<RunOutcome>) {
    emit(
        app,
        "heretic://status",
        json!({"state": state, "message": message, "outcome": outcome}),
    );
}

impl Worker {
    pub fn spawn(app: AppHandle) -> Worker {
        let (sender, receiver) = mpsc::channel::<Envelope>();
        let flags = Arc::new(Mutex::new(Flags::default()));
        let thread_flags = flags.clone();

        std::thread::Builder::new()
            .name("heretic-session".into())
            .spawn(move || run_worker(app, receiver, thread_flags))
            .expect("failed to spawn worker thread");

        Worker { sender, flags }
    }

    pub fn stop_flag(&self) -> Option<Arc<AtomicBool>> {
        self.flags.lock().unwrap().stop.clone()
    }

    pub fn cancel_flag(&self) -> Option<Arc<AtomicBool>> {
        self.flags.lock().unwrap().cancel.clone()
    }

    /// Sends a command without waiting for its completion (progress comes through events).
    pub fn send_async(&self, command: Command) -> Result<(), String> {
        if self.flags.lock().unwrap().busy {
            return Err("Another operation is in progress".into());
        }
        self.flags.lock().unwrap().busy = true;
        self.sender
            .send(Envelope { command, reply: None })
            .map_err(|_| "The worker thread has stopped".to_string())
    }

    /// Sends a command and waits for its result.
    pub async fn request(&self, command: Command) -> Result<Value, String> {
        if self.flags.lock().unwrap().busy {
            return Err("Another operation is in progress".into());
        }
        self.flags.lock().unwrap().busy = true;
        let (reply, receiver) = mpsc::channel();
        self.sender
            .send(Envelope {
                command,
                reply: Some(reply),
            })
            .map_err(|_| "The worker thread has stopped".to_string())?;
        tauri::async_runtime::spawn_blocking(move || {
            receiver
                .recv()
                .map_err(|_| "The worker thread has stopped".to_string())?
        })
        .await
        .map_err(|e| e.to_string())?
    }
}

fn require(session: &mut Option<Session>) -> Result<&mut Session, String> {
    session
        .as_mut()
        .ok_or_else(|| "No model is loaded; start a run first".to_string())
}

fn run_trials(app: &AppHandle, session: &mut Session, target: usize) -> Result<Value, String> {
    status(app, "running", None, None);
    match session.run_trials(target) {
        Ok(outcome) => {
            status(app, "ready", None, Some(outcome));
            Ok(json!({"outcome": outcome}))
        }
        Err(error) => {
            let message = format!("{error:#}");
            status(app, "ready", Some(message.clone()), None);
            Err(message)
        }
    }
}

fn run_worker(app: AppHandle, receiver: mpsc::Receiver<Envelope>, flags: Arc<Mutex<Flags>>) {
    let reporter: SharedReporter = Arc::new(GuiReporter { app: app.clone() });
    let event_app = app.clone();
    let events: EventSink = Arc::new(move |event| {
        emit(
            &event_app,
            "heretic://session",
            serde_json::to_value(event).unwrap_or_default(),
        );
    });

    let mut session: Option<Session> = None;

    for envelope in receiver {
        let result: Result<Value, String> = match envelope.command {
            Command::Start { settings, action } => {
                // Free the previous model before loading a new one.
                session = None;
                let mut new_session = Session::new(*settings, reporter.clone(), events.clone());
                flags.lock().unwrap().stop = Some(new_session.stop_handle());
                status(&app, "preparing", None, None);

                match new_session.prepare(action) {
                    Ok(()) => {
                        flags.lock().unwrap().cancel = new_session.cancel_handle();
                        let target = new_session.settings.n_trials;
                        let result = run_trials(&app, &mut new_session, target);
                        session = Some(new_session);
                        result
                    }
                    Err(error) => {
                        let message = format!("{error:#}");
                        status(&app, "failed", Some(message.clone()), None);
                        Err(message)
                    }
                }
            }

            Command::RunMore { count } => match require(&mut session) {
                Ok(session) => {
                    let completed = session.study().map_or(0, |s| s.completed_trials().count());
                    session.settings.n_trials = completed + count;
                    run_trials(&app, session, completed + count)
                }
                Err(error) => Err(error),
            },

            Command::Trials => require(&mut session).and_then(|session| {
                let trials = session.trials().map_err(|e| format!("{e:#}"))?;
                let evaluator = session.evaluator();
                Ok(json!({
                    "trials": trials,
                    "objective_names": evaluator.map(|e| e.objective_names()).unwrap_or_default(),
                    "baseline": evaluator.map(|e| e.baseline_scores.clone()).unwrap_or_default(),
                    "applied_trial": session.applied_trial,
                    "model": session.settings.model,
                    "modified_model_type": session.settings.modified_model_type,
                    "modifier": session.modifier().map(|m| m.modifier_name()),
                }))
            }),

            Command::Restore { number } => require(&mut session).and_then(|session| {
                session.restore_trial(number).map_err(|e| format!("{e:#}"))?;
                Ok(json!({"applied_trial": number}))
            }),

            Command::Chat { messages, use_adapters } => require(&mut session).and_then(|session| {
                let chat_app = app.clone();
                let response = session
                    .chat(&messages, use_adapters, &mut |text| {
                        emit(&chat_app, "heretic://chat", json!({"text": text}));
                    })
                    .map_err(|e| format!("{e:#}"))?;
                Ok(json!({"response": response}))
            }),

            Command::Save { directory, strategy } => require(&mut session).and_then(|session| {
                let saved = session.save_model(&directory, strategy).map_err(|e| format!("{e:#}"))?;
                Ok(json!({"directory": saved.directory, "files": saved.files}))
            }),

            Command::Upload {
                token,
                repo_id,
                private,
                strategy,
            } => require(&mut session).and_then(|session| {
                let progress_app = app.clone();
                session
                    .upload_model(&token, &repo_id, private, strategy, &move |done, total, file| {
                        emit(
                            &progress_app,
                            "heretic://progress",
                            json!({"task": format!("Uploading {file}"), "done": done, "total": total}),
                        );
                    })
                    .map_err(|e| format!("{e:#}"))?;
                Ok(json!({"repo_id": repo_id}))
            }),

            Command::SaveGguf {
                output,
                kind,
                llama_cpp_dir,
            } => require(&mut session).and_then(|session| {
                let export = session
                    .save_gguf(&output, kind, llama_cpp_dir.as_deref())
                    .map_err(|e| format!("{e:#}"))?;
                Ok(json!({"path": export.path, "bytes": export.bytes, "type": export.kind.name()}))
            }),

            Command::ModelCard => require(&mut session)
                .and_then(|session| session.model_card().map(Value::String).map_err(|e| format!("{e:#}"))),
        };

        flags.lock().unwrap().busy = false;
        if let Some(reply) = envelope.reply {
            let _ = reply.send(result);
        }
    }
}
