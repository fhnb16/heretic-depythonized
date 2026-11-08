// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Quantizing GGUF files with `llama-quantize` from a llama.cpp build the user downloaded.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;

use anyhow::{Context as _, Result, anyhow, bail};
use heretic_core::report::{ReporterExt, SharedReporter};
use serde::Serialize;

/// Where to get llama.cpp.
pub const LLAMA_CPP_RELEASES: &str = "https://github.com/ggml-org/llama.cpp/releases";

/// The GGUF file types Heretic offers (`llama-quantize` type names).
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum GgufType {
    F16,
    BF16,
    Q8_0,
    Q6_K,
    Q5_K_M,
    Q5_K_S,
    Q4_K_M,
    Q4_K_S,
    Q4_0,
    Q3_K_M,
    Q2_K,
}

impl GgufType {
    pub const ALL: [GgufType; 11] = [
        GgufType::F16,
        GgufType::BF16,
        GgufType::Q8_0,
        GgufType::Q6_K,
        GgufType::Q5_K_M,
        GgufType::Q5_K_S,
        GgufType::Q4_K_M,
        GgufType::Q4_K_S,
        GgufType::Q4_0,
        GgufType::Q3_K_M,
        GgufType::Q2_K,
    ];

    pub fn name(self) -> &'static str {
        match self {
            GgufType::F16 => "F16",
            GgufType::BF16 => "BF16",
            GgufType::Q8_0 => "Q8_0",
            GgufType::Q6_K => "Q6_K",
            GgufType::Q5_K_M => "Q5_K_M",
            GgufType::Q5_K_S => "Q5_K_S",
            GgufType::Q4_K_M => "Q4_K_M",
            GgufType::Q4_K_S => "Q4_K_S",
            GgufType::Q4_0 => "Q4_0",
            GgufType::Q3_K_M => "Q3_K_M",
            GgufType::Q2_K => "Q2_K",
        }
    }

    pub fn parse(text: &str) -> Result<GgufType> {
        let normalized = text.trim().to_ascii_uppercase().replace('-', "_");
        GgufType::ALL
            .into_iter()
            .find(|t| t.name() == normalized)
            .ok_or_else(|| {
                anyhow!(
                    "Unknown GGUF type \"{text}\". Choose one of: {}",
                    GgufType::ALL.map(GgufType::name).join(", ")
                )
            })
    }

    /// Approximate bits per weight (from `llama-quantize --help`), for size estimates.
    pub fn bits_per_weight(self) -> f64 {
        match self {
            GgufType::F16 | GgufType::BF16 => 16.0,
            GgufType::Q8_0 => 8.5,
            GgufType::Q6_K => 6.56,
            GgufType::Q5_K_M => 5.67,
            GgufType::Q5_K_S => 5.52,
            GgufType::Q4_K_M => 4.89,
            GgufType::Q4_K_S => 4.58,
            GgufType::Q4_0 => 4.34,
            GgufType::Q3_K_M => 3.91,
            GgufType::Q2_K => 2.96,
        }
    }

    /// One line for menus.
    pub fn description(self) -> &'static str {
        match self {
            GgufType::F16 => "16-bit, no quantization (largest, reference quality)",
            GgufType::BF16 => "bfloat16, no quantization (largest, reference quality)",
            GgufType::Q8_0 => "8-bit, practically lossless",
            GgufType::Q6_K => "6-bit, very close to the original",
            GgufType::Q5_K_M => "5-bit, high quality",
            GgufType::Q5_K_S => "5-bit, slightly smaller",
            GgufType::Q4_K_M => "4-bit, the usual balance of size and quality",
            GgufType::Q4_K_S => "4-bit, smaller",
            GgufType::Q4_0 => "4-bit legacy format, fast on some hardware",
            GgufType::Q3_K_M => "3-bit, noticeable quality loss",
            GgufType::Q2_K => "2-bit, large quality loss",
        }
    }

    /// Whether `llama-quantize` is needed (F16/BF16 are written directly).
    pub fn needs_quantize(self) -> bool {
        !matches!(self, GgufType::F16 | GgufType::BF16)
    }
}

/// A llama.cpp installation.
#[derive(Debug, Clone, Serialize)]
pub struct LlamaCpp {
    pub directory: PathBuf,
    pub quantize: PathBuf,
    /// Build number and commit, if a llama.cpp tool in the folder reports them.
    pub version: Option<String>,
}

fn executable(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

fn command(program: &Path) -> Command {
    let mut command = Command::new(program);
    // The shared libraries of llama.cpp releases live next to the executables.
    if let Some(dir) = program.parent() {
        command.current_dir(dir);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW: no console window flashing up from the GUI.
        command.creation_flags(0x0800_0000);
    }
    command
}

impl LlamaCpp {
    /// Finds `llama-quantize` in `dir` (or its `bin` / `build/bin` subfolders).
    pub fn locate(dir: &Path) -> Result<LlamaCpp> {
        if dir.as_os_str().is_empty() {
            bail!(
                "No llama.cpp folder is configured. Download a llama.cpp release from {LLAMA_CPP_RELEASES}, \
                 unpack it and select the folder that contains llama-quantize (setting llama_cpp_dir)."
            );
        }
        if !dir.is_dir() {
            bail!("The llama.cpp folder {} does not exist", dir.display());
        }
        let name = executable("llama-quantize");
        let candidates = [
            dir.to_path_buf(),
            dir.join("bin"),
            dir.join("build").join("bin"),
            dir.join("build").join("bin").join("Release"),
        ];
        let directory = candidates
            .iter()
            .find(|c| c.join(&name).is_file())
            .cloned()
            .ok_or_else(|| {
                anyhow!(
                    "{name} not found in {}; download a llama.cpp release from {LLAMA_CPP_RELEASES} and select the \
                 folder that contains {name}",
                    dir.display()
                )
            })?;
        let quantize = directory.join(&name);
        let version = Self::detect_version(&directory);
        Ok(LlamaCpp {
            directory,
            quantize,
            version,
        })
    }

    /// `--version` of a llama.cpp tool next to llama-quantize ("version: 1234 (abcdef0)").
    fn detect_version(directory: &Path) -> Option<String> {
        for tool in ["llama-cli", "llama-server", "llama-run", "llama-bench"] {
            let path = directory.join(executable(tool));
            if !path.is_file() {
                continue;
            }
            let Ok(output) = command(&path)
                .arg("--version")
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
            else {
                continue;
            };
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            if let Some(line) = text.lines().find(|l| l.trim_start().starts_with("version:")) {
                return Some(line.trim().trim_start_matches("version:").trim().to_string());
            }
        }
        None
    }

    /// Checks that `llama-quantize` runs (it prints its usage for `--help`).
    pub fn check(&self) -> Result<()> {
        let output = command(&self.quantize)
            .arg("--help")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| format!("Failed to run {}", self.quantize.display()))?;
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        if !text.contains("usage") {
            bail!(
                "{} did not print its usage; it may be damaged or built for another system (exit status {}): {}",
                self.quantize.display(),
                output.status,
                text.trim()
            );
        }
        Ok(())
    }

    /// Runs `llama-quantize <input> <output> <type> [threads]`, forwarding its output.
    pub fn quantize(&self, input: &Path, output: &Path, kind: GgufType, reporter: &SharedReporter) -> Result<()> {
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        reporter.info(format!(
            "* Quantizing to **{}** with {} ({threads} threads)...",
            kind.name(),
            self.quantize.display()
        ));
        let mut child = command(&self.quantize)
            .arg(input)
            .arg(output)
            .arg(kind.name())
            .arg(threads.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("Failed to start {}", self.quantize.display()))?;

        // Both streams are read on threads so neither pipe can fill up and block the child.
        let (sender, receiver) = mpsc::channel::<String>();
        let mut readers = Vec::new();
        let streams: Vec<Box<dyn Read + Send>> = vec![
            Box::new(child.stdout.take().unwrap()),
            Box::new(child.stderr.take().unwrap()),
        ];
        for stream in streams {
            let sender = sender.clone();
            readers.push(std::thread::spawn(move || {
                for line in BufReader::new(stream).lines().map_while(Result::ok) {
                    let _ = sender.send(line);
                }
            }));
        }
        drop(sender);

        let mut tail: Vec<String> = Vec::new();
        for line in receiver {
            if let Some((done, total)) = parse_progress(&line) {
                reporter.progress("Quantizing", done, Some(total));
            }
            reporter.debug(line.clone());
            tail.push(line);
            if tail.len() > 20 {
                tail.remove(0);
            }
        }
        for reader in readers {
            let _ = reader.join();
        }
        let status = child.wait()?;
        if !status.success() {
            bail!("llama-quantize failed ({status}). Last output:\n{}", tail.join("\n"));
        }
        if !output.is_file() {
            bail!("llama-quantize finished but did not create {}", output.display());
        }
        Ok(())
    }
}

/// Progress from llama-quantize lines like "[  12/ 291]  blk.1.attn_k.weight - ...".
fn parse_progress(line: &str) -> Option<(u64, u64)> {
    let inner = line.trim_start().strip_prefix('[')?;
    let (counts, _) = inner.split_once(']')?;
    let (done, total) = counts.split_once('/')?;
    Some((done.trim().parse().ok()?, total.trim().parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_types_case_insensitively() {
        assert_eq!(GgufType::parse("q4_k_m").unwrap(), GgufType::Q4_K_M);
        assert_eq!(GgufType::parse("Q8-0").unwrap(), GgufType::Q8_0);
        assert!(GgufType::parse("q9").unwrap_err().to_string().contains("Q4_K_M"));
        assert!(!GgufType::BF16.needs_quantize());
        assert!(GgufType::Q2_K.needs_quantize());
    }

    #[test]
    fn parses_quantize_progress() {
        assert_eq!(
            parse_progress("[  12/ 291]                  blk.1.attn_k.weight - [ 4096,  1024], type =   bf16"),
            Some((12, 291))
        );
        assert_eq!(parse_progress("main: quantize time = 1234 ms"), None);
    }

    #[test]
    fn missing_quantize_has_an_actionable_message() {
        let dir = tempfile::tempdir().unwrap();
        let error = LlamaCpp::locate(dir.path()).unwrap_err().to_string();
        assert!(error.contains("llama-quantize"), "{error}");
        assert!(error.contains(LLAMA_CPP_RELEASES), "{error}");
        let error = LlamaCpp::locate(Path::new("")).unwrap_err().to_string();
        assert!(error.contains("llama_cpp_dir"), "{error}");
    }
}
