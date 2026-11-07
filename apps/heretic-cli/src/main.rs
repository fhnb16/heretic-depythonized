// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Command line interface of Heretic.

mod reporter;

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Result, anyhow};
use console::style;
use heretic_core::Settings;
use heretic_core::config::{ExportStrategy, Sources, settings_field_docs, split_positional_args};
use heretic_core::prompt::Message;
use heretic_core::report::{ReporterExt, SharedReporter};
use heretic_session::{CheckpointAction, RunOutcome, Session, TrialSummary, no_events};
use inquire::{InquireError, Password, Select, Text};

use crate::reporter::{TerminalReporter, strip_markup};

/// Flags that are set when the user presses Ctrl+C.
fn interrupt_targets() -> &'static Mutex<Vec<Arc<AtomicBool>>> {
    static TARGETS: OnceLock<Mutex<Vec<Arc<AtomicBool>>>> = OnceLock::new();
    TARGETS.get_or_init(|| Mutex::new(Vec::new()))
}

static INTERRUPTS: AtomicUsize = AtomicUsize::new(0);

fn set_interrupt_targets(targets: Vec<Arc<AtomicBool>>) {
    INTERRUPTS.store(0, Ordering::SeqCst);
    *interrupt_targets().lock().unwrap() = targets;
}

fn install_interrupt_handler() {
    let _ = ctrlc::set_handler(|| {
        let count = INTERRUPTS.fetch_add(1, Ordering::SeqCst) + 1;
        let targets = interrupt_targets().lock().unwrap();
        if count >= 2 || targets.is_empty() {
            println!();
            println!("{}", style("Shutting down...").red());
            std::process::exit(130);
        }
        for target in targets.iter() {
            target.store(true, Ordering::SeqCst);
        }
        println!();
        println!(
            "{}",
            style("Interrupting... (press Ctrl+C again to exit immediately)").yellow()
        );
    });
}

fn print_banner() {
    let version = env!("CARGO_PKG_VERSION");
    println!("{}  v{version}", style("█░█░█▀▀░█▀▄░█▀▀░▀█▀░█░█▀▀").cyan());
    println!(
        "{}  {}",
        style("█▀█░█▀▀░█▀▄░█▀▀░░█░░█░█░░").cyan(),
        style("https://heretic-project.org").blue().underlined()
    );
    println!(
        "{}  {}",
        style("▀░▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀▀▀").cyan(),
        style("https://github.com/p-e-w/heretic").blue().underlined()
    );
    println!();
}

fn print_help() {
    println!("Usage: heretic [OPTIONS] <MODEL>");
    println!();
    println!("MODEL is a Hugging Face model ID or a path to a model on disk.");
    println!("Settings are read from config.toml in the working directory (or the file in");
    println!("HERETIC_CONFIG_FILE), HERETIC_<NAME> environment variables, and these options:");
    println!();
    for (name, description) in settings_field_docs() {
        println!("  {}", style(format!("--{}", name.replace('_', "-"))).bold());
        if !description.is_empty() {
            println!("      {description}");
        }
    }
    println!();
    println!("  {}", style("--help, -h").bold());
    println!("      Show this help.");
    println!("  {}", style("--version").bold());
    println!("      Show the version.");
}

/// Asks a question; `None` means the user cancelled (Esc).
fn ask<T>(result: Result<T, InquireError>) -> Result<Option<T>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(InquireError::OperationCanceled) => Ok(None),
        Err(InquireError::OperationInterrupted) => {
            println!();
            println!("{}", style("Shutting down...").red());
            std::process::exit(130);
        }
        Err(error) => Err(anyhow!(error)),
    }
}

/// A menu entry with a value.
struct Choice<T> {
    title: String,
    value: T,
}

impl<T> std::fmt::Display for Choice<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.title)
    }
}

fn choice<T>(title: impl Into<String>, value: T) -> Choice<T> {
    Choice {
        title: title.into(),
        value,
    }
}

fn select<T>(message: &str, choices: Vec<Choice<T>>) -> Result<Option<T>> {
    Ok(ask(Select::new(message, choices).with_page_size(15).prompt())?.map(|c| c.value))
}

fn format_trial_title(trial: &TrialSummary) -> String {
    let scores: Vec<String> = trial
        .scores
        .iter()
        .map(|s| format!("{}: {}", s.name, strip_markup(&s.score.display)))
        .collect();
    format!("[Trial {:>3}] {}", trial.index, scores.join(", "))
}

fn main() {
    install_interrupt_handler();
    print_banner();

    if let Err(error) = run() {
        println!();
        println!("{} {}", style("Error:").red().bold(), style(format!("{error:#}")).red());
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return Ok(());
    }
    if args.iter().any(|a| a == "--version") {
        println!("heretic {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    let (positional, mut options) = split_positional_args(&args);
    match positional.as_slice() {
        [] => {}
        [model] => {
            options.push("--model".into());
            options.push(model.clone());
        }
        _ => return Err(anyhow!("Unexpected arguments: {}", positional.join(" "))),
    }

    let settings = match Settings::load(&Sources {
        config_file: None,
        environment: true,
        cli_args: options,
        overrides: heretic_core::toml::Table::new(),
    }) {
        Ok(settings) => settings,
        Err(errors) => {
            println!(
                "{}",
                style(format!("Configuration contains {} errors:", errors.len())).red()
            );
            for error in errors {
                println!("{}: {}", style(&error.field).bold(), style(&error.message).yellow());
            }
            println!();
            println!(
                "Run {} or see {} for details.",
                style("heretic --help").bold(),
                style("config.default.toml").bold()
            );
            return Ok(());
        }
    };

    if settings.collect_reproducibles.is_some() || settings.reproduce.is_some() {
        return Err(anyhow!(
            "--collect-reproducibles and --reproduce are not available in this version of Heretic yet"
        ));
    }

    if settings.model.is_empty() {
        println!("{}", style("No model specified.").red());
        println!();
        print_help();
        return Ok(());
    }

    let reporter: SharedReporter = Arc::new(TerminalReporter::new(settings.print_debug_information));
    reporter.info(format!(
        "Device support: **CPU**{}",
        if heretic_engine::device::cuda_compiled() {
            ", **CUDA**"
        } else {
            ""
        }
    ));

    let mut session = Session::new(settings.clone(), reporter.clone(), no_events());

    if let Some(other) = settings.evaluate_model.clone() {
        session.prepare_evaluation()?;
        reporter.heading(format!("Loading model **{other}**..."));
        let scores = session.evaluate_other_model(&other)?;
        for (name, score) in scores {
            reporter.info(format!("  * **{name}:** {}", score.display));
        }
        return Ok(());
    }

    // Existing checkpoint?
    let mut checkpoint_action = None;
    if let Some(info) = session.existing_checkpoint() {
        let predefined = settings.checkpoint_action.as_deref().and_then(CheckpointAction::parse);
        if predefined.is_none() {
            println!();
            if info.finished {
                println!(
                    "{} You can show the results from the previous run, allowing you to export models or to run \
                     additional trials. Alternatively, you can ignore the previous run and start from scratch. \
                     This will delete the checkpoint file and all results from the previous run.",
                    style("You have already processed this model.").green()
                );
            } else {
                println!(
                    "{} You can continue the previous run from where it stopped ({} trials completed). This will \
                     override any specified settings. Alternatively, you can ignore the previous run and start \
                     from scratch. This will delete the checkpoint file and all results from the previous run.",
                    style("You have already processed this model, but the run was interrupted.").yellow(),
                    info.completed_trials
                );
            }
            println!();
        }
        checkpoint_action = match predefined {
            Some(action) => Some(action),
            None => {
                let continue_title = if info.finished {
                    "Show the results from the previous run"
                } else {
                    "Continue the previous run"
                };
                match select(
                    "How would you like to proceed?",
                    vec![
                        choice(continue_title, Some(CheckpointAction::Continue)),
                        choice(
                            "Ignore the previous run and start from scratch",
                            Some(CheckpointAction::Restart),
                        ),
                        choice("Exit program", None),
                    ],
                )? {
                    Some(Some(action)) => Some(action),
                    _ => return Ok(()),
                }
            }
        };
    }

    session.prepare(checkpoint_action)?;

    set_interrupt_targets(interrupt_flags(&session));
    let n_trials = session.settings.n_trials;
    let outcome = session.run_trials(n_trials)?;
    set_interrupt_targets(Vec::new());

    trial_menu(&mut session, &reporter, outcome)
}

fn interrupt_flags(session: &Session) -> Vec<Arc<AtomicBool>> {
    let mut flags = vec![session.stop_handle()];
    if let Some(cancel) = session.cancel_handle() {
        flags.push(cancel);
    }
    flags
}

fn trial_menu(session: &mut Session, reporter: &SharedReporter, mut outcome: RunOutcome) -> Result<()> {
    let predefined_trial = session.settings.trial_index;

    loop {
        let front = session.pareto_front()?;
        if front.is_empty() {
            if outcome == RunOutcome::Completed {
                return Err(anyhow!("No trials were completed"));
            }
            println!();
            println!("{}", style("Shutting down...").red());
            return Ok(());
        }

        println!();
        println!("{}", style("Optimization finished!").green().bold());

        if predefined_trial.is_none() {
            println!();
            println!(
                "The following trials resulted in Pareto optimal combinations of the optimization objectives. \
                 After selecting a trial, you will be able to save the model, upload it to Hugging Face, \
                 chat with it to test how well it works, or run standard benchmarks on it. \
                 You can return to this menu later to select a different trial."
            );
            println!();
        }

        enum TrialChoice {
            Trial(usize),
            More,
            Exit,
        }

        let selection = match predefined_trial {
            Some(index) => TrialChoice::Trial(
                front
                    .get(index)
                    .ok_or_else(|| {
                        anyhow!(
                            "trial_index {index} is out of range (Pareto front has {} trials)",
                            front.len()
                        )
                    })?
                    .number,
            ),
            None => {
                let mut choices: Vec<Choice<TrialChoice>> = front
                    .iter()
                    .map(|t| choice(format_trial_title(t), TrialChoice::Trial(t.number)))
                    .collect();
                choices.push(choice("Run additional trials", TrialChoice::More));
                choices.push(choice("Exit program", TrialChoice::Exit));
                select("Which trial do you want to use?", choices)?.unwrap_or(TrialChoice::Exit)
            }
        };

        match selection {
            TrialChoice::Exit => return Ok(()),
            TrialChoice::More => {
                let additional = match session.settings.n_additional_trials {
                    Some(n) => n,
                    None => loop {
                        let Some(text) = ask(Text::new("How many additional trials do you want to run?").prompt())?
                        else {
                            break 0;
                        };
                        if text.trim().is_empty() {
                            break 0;
                        }
                        match text.trim().parse::<usize>() {
                            Ok(n) if n > 0 => break n,
                            Ok(_) => println!("{}", style("Please enter a number greater than 0.").red()),
                            Err(_) => println!("{}", style("Please enter a number.").red()),
                        }
                    },
                };
                if additional == 0 {
                    continue;
                }
                let completed = session.study().map_or(0, |s| s.completed_trials().count());
                session.settings.n_trials = completed + additional;
                set_interrupt_targets(interrupt_flags(session));
                outcome = session.run_trials(completed + additional)?;
                set_interrupt_targets(Vec::new());
                continue;
            }
            TrialChoice::Trial(number) => {
                let index = front
                    .iter()
                    .find(|t| t.number == number)
                    .map_or(number + 1, |t| t.index);
                reporter.heading(format!("Restoring model from trial **{index}**..."));
                session.restore_trial(number)?;
                let exit = action_menu(session, reporter)?;
                if exit || predefined_trial.is_some() {
                    return Ok(());
                }
            }
        }
    }
}

/// Returns true if the program should exit.
fn action_menu(session: &mut Session, reporter: &SharedReporter) -> Result<bool> {
    let predefined = session.settings.model_action.clone();

    loop {
        #[derive(Clone, Copy, PartialEq)]
        enum Action {
            Save,
            Upload,
            Chat,
            Benchmark,
            Back,
        }

        let action = match predefined.as_deref() {
            Some("save") => Action::Save,
            Some("upload") => Action::Upload,
            Some(other) => {
                return Err(anyhow!(
                    "Unknown model_action \"{other}\" (expected \"save\" or \"upload\")"
                ));
            }
            None => {
                println!();
                let kind = session.settings.modified_model_type.clone();
                select(
                    &format!("What do you want to do with the {kind} model?"),
                    vec![
                        choice("Save the model to a local folder", Action::Save),
                        choice("Upload the model to Hugging Face", Action::Upload),
                        choice("Chat with the model", Action::Chat),
                        choice("Benchmark the model", Action::Benchmark),
                        choice("Return to the trial selection menu", Action::Back),
                    ],
                )?
                .unwrap_or(Action::Back)
            }
        };

        // Errors are reported without losing the optimized model, so another action can be tried.
        let result = match action {
            Action::Back => return Ok(false),
            Action::Save => save_action(session, reporter),
            Action::Upload => upload_action(session, reporter),
            Action::Chat => chat_action(session),
            Action::Benchmark => Err(anyhow!("Benchmarks are not available in this version of Heretic yet")),
        };

        if let Err(error) = result {
            println!("{} {}", style("Error:").red(), style(format!("{error:#}")).red());
        }

        if predefined.is_some() {
            return Ok(true);
        }
    }
}

fn export_strategy(session: &Session) -> Result<Option<ExportStrategy>> {
    if let Some(strategy) = session.settings.export_strategy {
        return Ok(Some(strategy));
    }
    select(
        "How do you want to export the model?",
        vec![
            choice("Merge the LoRA and export the full model", ExportStrategy::Merge),
            choice("Export the LoRA only (can be merged later)", ExportStrategy::Adapter),
        ],
    )
}

fn save_action(session: &mut Session, reporter: &SharedReporter) -> Result<()> {
    let directory = match session.settings.save_directory.clone() {
        Some(directory) => directory,
        None => match ask(Text::new("Path to the folder:").prompt())? {
            Some(directory) if !directory.trim().is_empty() => directory.trim().to_string(),
            _ => return Ok(()),
        },
    };
    let Some(strategy) = export_strategy(session)? else {
        return Ok(());
    };
    session.save_model(&PathBuf::from(&directory), strategy)?;
    reporter.success(format!("Model saved to {directory}."));
    Ok(())
}

fn upload_action(session: &mut Session, reporter: &SharedReporter) -> Result<()> {
    // The token is never stored on disk; for scripting, use HF_TOKEN.
    let token = match heretic_core::hub::token() {
        Some(token) => token,
        None => match ask(Password::new("Hugging Face access token:")
            .without_confirmation()
            .prompt())?
        {
            Some(token) if !token.trim().is_empty() => token.trim().to_string(),
            _ => return Ok(()),
        },
    };

    let user = heretic_export::hub_upload::whoami(&token)?;
    reporter.info(format!(
        "Logged in as **{} ({})**",
        user.fullname.clone().unwrap_or_else(|| user.name.clone()),
        user.email.clone().unwrap_or_else(|| "no email found".into())
    ));

    let model_name = std::path::Path::new(&session.settings.model)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".into());
    let repo_id = match session.settings.upload_repo_id.clone() {
        Some(repo_id) => repo_id,
        None => match ask(Text::new("Name of repository:")
            .with_default(&format!("{}/{model_name}-heretic", user.name))
            .prompt())?
        {
            Some(repo_id) if !repo_id.trim().is_empty() => repo_id.trim().to_string(),
            _ => return Ok(()),
        },
    };

    let private = match session.settings.upload_repo_private {
        Some(private) => private,
        None => match select(
            "Should the repository be public or private?",
            vec![choice("Public", false), choice("Private", true)],
        )? {
            Some(private) => private,
            None => return Ok(()),
        },
    };

    let Some(strategy) = export_strategy(session)? else {
        return Ok(());
    };

    let progress_reporter = reporter.clone();
    session.upload_model(&token, &repo_id, private, strategy, &move |done, total, file| {
        progress_reporter.progress(&format!("Uploading {file}"), done as u64, Some(total as u64));
    })?;
    reporter.success(format!("Model uploaded to {repo_id}."));
    Ok(())
}

fn chat_action(session: &mut Session) -> Result<()> {
    println!();
    println!(
        "{}",
        style("Press Ctrl+C to stop a response, or submit an empty message to return to the menu.").cyan()
    );

    let mut chat = vec![Message::system(session.settings.system_prompt.clone())];
    loop {
        let Some(message) = ask(Text::new("User:").prompt())? else {
            return Ok(());
        };
        if message.trim().is_empty() {
            return Ok(());
        }
        chat.push(Message::user(message));

        print!("{} ", style("Assistant:").bold());
        std::io::stdout().flush()?;

        set_interrupt_targets(session.cancel_handle().into_iter().collect());
        let response = session.chat(&chat, true, &mut |text| {
            print!("{text}");
            let _ = std::io::stdout().flush();
        });
        set_interrupt_targets(Vec::new());
        println!();

        match response {
            Ok(response) => chat.push(Message::assistant(response)),
            Err(error) => {
                println!("{}", style(format!("{error:#}")).red());
                chat.pop();
            }
        }
    }
}
