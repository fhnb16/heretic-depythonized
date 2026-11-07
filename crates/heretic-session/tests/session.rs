// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! End-to-end test of a session with a tiny random model.
//! Run with `cargo test -p heretic-session --release --test session -- --ignored --nocapture`.

use std::sync::{Arc, Mutex};

use heretic_core::Settings;
use heretic_core::config::ExportStrategy;
use heretic_core::report::StderrReporter;
use heretic_session::{CheckpointAction, RunOutcome, Session, SessionEvent};

const CONFIG: &str = r#"
model = "tiny-random/qwen2.5"
model_commit = "7a6a3128ee4137a248d6d1582824592b87a81647"
seed = 12345
batch_size = 2
max_response_length = 10
n_trials = 3
n_startup_trials = 2
modifiers = [{ plugin = "heretic.modifiers.abliteration.Abliteration" }]

[[response_prefix_test_prompts]]
dataset = "mlabonne/harmless_alpaca"
commit = "02c6a92cfcf11bb0c387334f8146d149d65b587f"
split = "train[:4]"
column = "text"

[scorer.KLDivergence.prompts]
dataset = "mlabonne/harmless_alpaca"
commit = "02c6a92cfcf11bb0c387334f8146d149d65b587f"
split = "test[:4]"
column = "text"

[scorer.KeywordRate.prompts]
dataset = "mlabonne/harmful_behaviors"
commit = "01cead01398926d81f7c52bdb790ee8cf77ebba7"
split = "test[:4]"
column = "text"

[modifier.Abliteration.good_prompts]
dataset = "mlabonne/harmless_alpaca"
commit = "02c6a92cfcf11bb0c387334f8146d149d65b587f"
split = "train[:4]"
column = "text"

[modifier.Abliteration.bad_prompts]
dataset = "mlabonne/harmful_behaviors"
commit = "01cead01398926d81f7c52bdb790ee8cf77ebba7"
split = "train[:4]"
column = "text"
"#;

fn settings(dir: &std::path::Path) -> Settings {
    let mut table: heretic_core::toml::Table = heretic_core::toml::from_str(CONFIG).unwrap();
    table.insert(
        "study_checkpoint_dir".into(),
        heretic_core::toml::Value::String(dir.join("checkpoints").display().to_string()),
    );
    Settings::from_table(table).unwrap()
}

#[test]
#[ignore = "downloads a model and datasets"]
fn full_session_with_resume() {
    let dir = tempfile::tempdir_in(env!("CARGO_TARGET_TMPDIR")).unwrap();
    let events: Arc<Mutex<Vec<SessionEvent>>> = Arc::default();
    let sink_events = events.clone();
    let sink = Arc::new(move |event: &SessionEvent| sink_events.lock().unwrap().push(event.clone()));

    let mut session = Session::new(settings(dir.path()), Arc::new(StderrReporter), sink);
    assert!(session.existing_checkpoint().is_none());
    session.prepare(None).unwrap();
    assert_eq!(session.run_trials(2).unwrap(), RunOutcome::Completed);

    let finished: Vec<usize> = events
        .lock()
        .unwrap()
        .iter()
        .filter_map(|e| match e {
            SessionEvent::TrialFinished { trial, .. } => Some(trial.index),
            _ => None,
        })
        .collect();
    assert_eq!(finished, vec![1, 2]);

    let front = session.pareto_front().unwrap();
    assert!(!front.is_empty());
    assert!(front.iter().all(|t| t.pareto_optimal));
    assert_eq!(session.trials().unwrap().len(), 2);

    session.restore_trial(front[0].number).unwrap();
    let card = session.model_card().unwrap();
    assert!(card.contains("tiny-random/qwen2.5"));
    assert!(card.contains("- heretic"));

    let saved = session
        .save_model(&dir.path().join("adapter"), ExportStrategy::Adapter)
        .unwrap();
    assert!(saved.files.contains(&"adapter_config.json".to_string()));

    // The run is not finished (3 trials configured), so it can be continued.
    let info = session.existing_checkpoint().unwrap();
    assert!(!info.finished);
    assert_eq!(info.completed_trials, 2);
    drop(session);

    let mut resumed = Session::new(
        settings(dir.path()),
        Arc::new(StderrReporter),
        heretic_session::no_events(),
    );
    resumed.prepare(Some(CheckpointAction::Continue)).unwrap();
    assert_eq!(resumed.study().unwrap().completed_trials().count(), 2);
    let target = resumed.settings.n_trials;
    assert_eq!(resumed.run_trials(target).unwrap(), RunOutcome::Completed);
    assert_eq!(resumed.trials().unwrap().len(), 3);
    assert!(resumed.existing_checkpoint().unwrap().finished);
}
