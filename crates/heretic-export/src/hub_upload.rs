// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Uploading to the Hugging Face Hub: repository creation, LFS uploads and commits.
//!
//! The access token is only used for requests to the Hub and is never written to disk.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context as _, Result, anyhow, bail};
use base64::Engine;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::file_sha256;

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .http_status_as_error(false)
        .user_agent(concat!("heretic/", env!("CARGO_PKG_VERSION")))
        .build()
        .into()
}

fn endpoint() -> String {
    heretic_core::hub::endpoint()
}

fn check(response: &mut ureq::http::Response<ureq::Body>, what: &str) -> Result<()> {
    let status = response.status().as_u16();
    if (200..300).contains(&status) {
        return Ok(());
    }
    let body = response.body_mut().read_to_string().unwrap_or_default();
    let message = serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(String::from))
        .unwrap_or(body);
    bail!("{what} failed (HTTP {status}): {message}")
}

#[derive(Debug, Clone, Deserialize)]
pub struct User {
    pub name: String,
    #[serde(default)]
    pub fullname: Option<String>,
    #[serde(default)]
    pub email: Option<String>,
}

/// The account the token belongs to.
pub fn whoami(token: &str) -> Result<User> {
    let mut response = agent()
        .get(&format!("{}/api/whoami-v2", endpoint()))
        .header("Authorization", &format!("Bearer {token}"))
        .call()
        .context("Failed to contact the Hugging Face Hub")?;
    check(&mut response, "Authentication")?;
    Ok(response.body_mut().read_json()?)
}

/// Creates a model repository (succeeds if it already exists).
pub fn create_repo(token: &str, repo_id: &str, private: bool) -> Result<()> {
    let (organization, name) = match repo_id.split_once('/') {
        Some((org, name)) => (Some(org), name),
        None => (None, repo_id),
    };
    let mut body = json!({"type": "model", "name": name, "private": private});
    if let Some(org) = organization {
        body["organization"] = json!(org);
    }
    let mut response = agent()
        .post(&format!("{}/api/repos/create", endpoint()))
        .header("Authorization", &format!("Bearer {token}"))
        .send_json(&body)
        .context("Failed to create the repository")?;
    if response.status().as_u16() == 409 {
        // Already exists.
        return Ok(());
    }
    check(&mut response, "Repository creation")
}

/// A file to upload.
#[derive(Debug, Clone)]
pub struct UploadFile {
    /// Path in the repository.
    pub path_in_repo: String,
    pub local_path: PathBuf,
}

struct Prepared {
    file: UploadFile,
    size: u64,
    sha256: String,
    lfs: bool,
}

fn sample_base64(path: &Path) -> Result<String> {
    let mut buffer = vec![0u8; 512];
    let mut file = std::fs::File::open(path)?;
    let count = file.read(&mut buffer)?;
    Ok(base64::engine::general_purpose::STANDARD.encode(&buffer[..count]))
}

/// Uploads files and creates a single commit containing all of them.
/// `progress` receives (files done, total files, current file).
pub fn upload_files(
    token: &str,
    repo_id: &str,
    files: &[UploadFile],
    commit_message: &str,
    progress: &dyn Fn(usize, usize, &str),
) -> Result<()> {
    let agent = agent();
    let auth = format!("Bearer {token}");

    // Hash all files and ask the Hub which ones must go through LFS.
    let mut prepared = Vec::new();
    for file in files {
        let size = std::fs::metadata(&file.local_path)
            .with_context(|| format!("Failed to read {}", file.local_path.display()))?
            .len();
        prepared.push(Prepared {
            file: file.clone(),
            size,
            sha256: file_sha256(&file.local_path)?,
            lfs: false,
        });
    }

    for chunk in prepared.chunks_mut(100) {
        let request: Vec<Value> = chunk
            .iter()
            .map(|p| {
                Ok(json!({
                    "path": p.file.path_in_repo,
                    "size": p.size,
                    "sample": sample_base64(&p.file.local_path)?,
                }))
            })
            .collect::<Result<_>>()?;
        let mut response = agent
            .post(&format!("{}/api/models/{repo_id}/preupload/main", endpoint()))
            .header("Authorization", &auth)
            .send_json(json!({ "files": request }))
            .context("Preupload request failed")?;
        check(&mut response, "Preupload")?;
        let answer: Value = response.body_mut().read_json()?;
        for entry in answer
            .get("files")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
        {
            let path = entry.get("path").and_then(Value::as_str).unwrap_or_default();
            let lfs = entry.get("uploadMode").and_then(Value::as_str) == Some("lfs");
            if let Some(p) = chunk.iter_mut().find(|p| p.file.path_in_repo == path) {
                p.lfs = lfs;
            }
        }
    }

    // Upload the LFS objects.
    let total = prepared.len();
    for (index, p) in prepared.iter().enumerate() {
        progress(index, total, &p.file.path_in_repo);
        if p.lfs {
            upload_lfs(&agent, &auth, repo_id, p)?;
        }
    }

    // Create the commit.
    let mut body = String::new();
    body.push_str(&serde_json::to_string(&json!({
        "key": "header",
        "value": {"summary": commit_message, "description": ""},
    }))?);
    body.push('\n');
    for p in &prepared {
        let line = if p.lfs {
            json!({
                "key": "lfsFile",
                "value": {"path": p.file.path_in_repo, "algo": "sha256", "oid": p.sha256, "size": p.size},
            })
        } else {
            let content = std::fs::read(&p.file.local_path)?;
            json!({
                "key": "file",
                "value": {
                    "path": p.file.path_in_repo,
                    "encoding": "base64",
                    "content": base64::engine::general_purpose::STANDARD.encode(content),
                },
            })
        };
        body.push_str(&serde_json::to_string(&line)?);
        body.push('\n');
    }

    let mut response = agent
        .post(&format!("{}/api/models/{repo_id}/commit/main", endpoint()))
        .header("Authorization", &auth)
        .header("Content-Type", "application/x-ndjson")
        .send(body.as_bytes())
        .context("Commit request failed")?;
    check(&mut response, "Commit")?;
    progress(total, total, "");
    Ok(())
}

fn upload_lfs(agent: &ureq::Agent, auth: &str, repo_id: &str, p: &Prepared) -> Result<()> {
    let batch = json!({
        "operation": "upload",
        "transfers": ["basic", "multipart"],
        "hash_algo": "sha256",
        "ref": {"name": "main"},
        "objects": [{"oid": p.sha256, "size": p.size}],
    });
    let mut response = agent
        .post(&format!("{}/{repo_id}.git/info/lfs/objects/batch", endpoint()))
        .header("Authorization", auth)
        .header("Accept", "application/vnd.git-lfs+json")
        .header("Content-Type", "application/vnd.git-lfs+json")
        .send(serde_json::to_vec(&batch)?.as_slice())
        .context("LFS batch request failed")?;
    check(&mut response, "LFS batch request")?;
    let answer: Value = response.body_mut().read_json()?;

    let object = answer
        .get("objects")
        .and_then(Value::as_array)
        .and_then(|o| o.first())
        .ok_or_else(|| anyhow!("Invalid LFS batch response"))?;
    if let Some(error) = object.get("error") {
        bail!("LFS upload of {} rejected: {error}", p.file.path_in_repo);
    }
    let Some(actions) = object.get("actions") else {
        // The object already exists on the server.
        return Ok(());
    };

    if let Some(upload) = actions.get("upload") {
        let href = upload
            .get("href")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("LFS upload without href"))?;
        let header = upload.get("header").cloned().unwrap_or_default();

        match header
            .get("chunk_size")
            .and_then(|c| c.as_str().and_then(|s| s.parse::<u64>().ok()).or(c.as_u64()))
        {
            Some(chunk_size) => {
                // Multipart upload: one presigned URL per part.
                let mut parts: Vec<(usize, String)> = header
                    .as_object()
                    .into_iter()
                    .flatten()
                    .filter_map(|(k, v)| Some((k.parse::<usize>().ok()?, v.as_str()?.to_string())))
                    .collect();
                parts.sort();

                let mut file = std::fs::File::open(&p.file.local_path)?;
                let mut completed = Vec::new();
                for (number, url) in parts {
                    let offset = (number as u64 - 1) * chunk_size;
                    let length = chunk_size.min(p.size.saturating_sub(offset));
                    let mut buffer = vec![0u8; length as usize];
                    file.seek(SeekFrom::Start(offset))?;
                    file.read_exact(&mut buffer)?;
                    let mut response = agent
                        .put(&url)
                        .config()
                        .timeout_global(None)
                        .build()
                        .send(buffer.as_slice())
                        .with_context(|| format!("Upload of part {number} failed"))?;
                    check(&mut response, "Part upload")?;
                    let etag = response
                        .headers()
                        .get("etag")
                        .and_then(|e| e.to_str().ok())
                        .unwrap_or_default()
                        .to_string();
                    completed.push(json!({"partNumber": number, "etag": etag}));
                }

                let mut response = agent
                    .post(href)
                    .header("Accept", "application/vnd.git-lfs+json")
                    .header("Content-Type", "application/vnd.git-lfs+json")
                    .send(serde_json::to_vec(&json!({"oid": p.sha256, "parts": completed}))?.as_slice())
                    .context("Multipart completion failed")?;
                check(&mut response, "Multipart completion")?;
            }
            None => {
                let mut request = agent.put(href);
                if let Some(map) = header.as_object() {
                    for (key, value) in map {
                        if let Some(value) = value.as_str() {
                            request = request.header(key.as_str(), value);
                        }
                    }
                }
                let content = std::fs::read(&p.file.local_path)?;
                let mut response = request
                    .config()
                    .timeout_global(None)
                    .build()
                    .send(content.as_slice())
                    .with_context(|| format!("Upload of {} failed", p.file.path_in_repo))?;
                check(&mut response, "LFS upload")?;
            }
        }
    }

    if let Some(verify) = actions.get("verify") {
        let href = verify
            .get("href")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("LFS verify without href"))?;
        let mut request = agent
            .post(href)
            .header("Accept", "application/vnd.git-lfs+json")
            .header("Content-Type", "application/vnd.git-lfs+json");
        if let Some(map) = verify.get("header").and_then(Value::as_object) {
            for (key, value) in map {
                if let Some(value) = value.as_str() {
                    request = request.header(key.as_str(), value);
                }
            }
        } else {
            request = request.header("Authorization", auth);
        }
        let mut response = request
            .send(serde_json::to_vec(&json!({"oid": p.sha256, "size": p.size}))?.as_slice())
            .context("LFS verification failed")?;
        check(&mut response, "LFS verification")?;
    }

    Ok(())
}

/// All files of a directory (recursively), as upload entries.
pub fn directory_files(dir: &Path) -> Result<Vec<UploadFile>> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<UploadFile>) -> Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                walk(root, &path, out)?;
            } else {
                let relative = path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/");
                out.push(UploadFile {
                    path_in_repo: relative,
                    local_path: path,
                });
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(dir, dir, &mut files)?;
    files.sort_by(|a, b| a.path_in_repo.cmp(&b.path_in_repo));
    Ok(files)
}
