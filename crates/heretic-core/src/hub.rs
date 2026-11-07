// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Minimal Hugging Face Hub client.
//!
//! Files are cached in the standard Hugging Face cache layout
//! (`$HF_HUB_CACHE` / `$HF_HOME/hub` / `~/.cache/huggingface/hub`), so models and
//! datasets downloaded by other tools are reused and vice versa.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{Context as _, Result, anyhow, bail};
use regex::Regex;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepoType {
    Model,
    Dataset,
    Space,
}

impl RepoType {
    fn api_segment(self) -> &'static str {
        match self {
            RepoType::Model => "models",
            RepoType::Dataset => "datasets",
            RepoType::Space => "spaces",
        }
    }

    fn url_prefix(self) -> &'static str {
        match self {
            RepoType::Model => "",
            RepoType::Dataset => "datasets/",
            RepoType::Space => "spaces/",
        }
    }

    fn cache_prefix(self) -> &'static str {
        match self {
            RepoType::Model => "models",
            RepoType::Dataset => "datasets",
            RepoType::Space => "spaces",
        }
    }
}

/// A file in a repository, as listed by the Hub API.
#[derive(Debug, Clone, Deserialize)]
pub struct Sibling {
    pub rfilename: String,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub lfs: Option<LfsInfo>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LfsInfo {
    pub sha256: String,
    #[serde(default)]
    pub size: Option<u64>,
}

/// Repository metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct RepoInfo {
    pub id: String,
    pub sha: String,
    #[serde(default)]
    pub siblings: Vec<Sibling>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub private: Option<bool>,
}

/// Progress callback for downloads: (bytes downloaded, total bytes if known).
pub type ProgressFn<'a> = &'a (dyn Fn(u64, Option<u64>) + Send + Sync);

pub fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|e| !e.is_empty())
        .unwrap_or_else(|| "https://huggingface.co".into())
        .trim_end_matches('/')
        .to_string()
}

fn hf_home() -> PathBuf {
    if let Ok(home) = std::env::var("HF_HOME") {
        return PathBuf::from(home);
    }
    let base = std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .ok()
        .or_else(|| dirs::home_dir().map(|home| home.join(".cache")))
        .unwrap_or_else(|| PathBuf::from(".cache"));
    base.join("huggingface")
}

/// The directory where repositories are cached.
pub fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("HF_HUB_CACHE").or_else(|_| std::env::var("HUGGINGFACE_HUB_CACHE")) {
        return PathBuf::from(dir);
    }
    hf_home().join("hub")
}

/// Whether the Hub must not be contacted (`HF_HUB_OFFLINE=1`).
pub fn is_offline() -> bool {
    std::env::var("HF_HUB_OFFLINE")
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

/// The access token from `HF_TOKEN` or the token file written by `hf auth login`.
/// Heretic never writes this file itself.
pub fn token() -> Option<String> {
    if let Ok(token) = std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN")) {
        let token = token.trim().to_string();
        if !token.is_empty() {
            return Some(token);
        }
    }
    let path = std::env::var("HF_TOKEN_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| hf_home().join("token"));
    std::fs::read_to_string(path)
        .ok()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
}

fn agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_recv_response(Some(Duration::from_secs(60)))
            .http_status_as_error(false)
            .user_agent(concat!("heretic/", env!("CARGO_PKG_VERSION")))
            .build()
            .into()
    })
}

/// Validates a repository ID like `huggingface_hub.utils.validate_repo_id`.
pub fn validate_repo_id(repo_id: &str) -> Result<()> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"^(\b[\w\-.]+\b/)?\b[\w\-.]{1,96}\b$").unwrap());
    if repo_id.matches('/').count() > 1 {
        bail!("Repo id must be in the form 'repo_name' or 'namespace/repo_name': '{repo_id}'");
    }
    if !re.is_match(repo_id) || repo_id.contains("--") || repo_id.contains("..") || repo_id.ends_with(".git") {
        bail!("Invalid repo id: '{repo_id}'");
    }
    Ok(())
}

/// Whether a path likely refers to a Hugging Face repository.
/// Existing local paths take precedence, even if they are also valid repository IDs.
pub fn is_hf_path(path: &str) -> bool {
    !Path::new(path).exists() && validate_repo_id(path).is_ok()
}

fn repo_cache_dir(repo_type: RepoType, repo_id: &str) -> PathBuf {
    cache_dir().join(format!("{}--{}", repo_type.cache_prefix(), repo_id.replace('/', "--")))
}

fn with_auth(
    request: ureq::RequestBuilder<ureq::typestate::WithoutBody>,
    token: Option<&str>,
) -> ureq::RequestBuilder<ureq::typestate::WithoutBody> {
    match token {
        Some(token) => request.header("Authorization", &format!("Bearer {token}")),
        None => request,
    }
}

fn check_status(response: &ureq::http::Response<ureq::Body>, what: &str) -> Result<()> {
    let status = response.status().as_u16();
    match status {
        200..=299 => Ok(()),
        401 => bail!("{what}: unauthorized (401). The repository may be private or gated; set HF_TOKEN."),
        403 => bail!("{what}: access denied (403). The repository may be gated; accept its terms on huggingface.co."),
        404 => bail!("{what}: not found (404)"),
        _ => bail!("{what}: HTTP {status}"),
    }
}

/// Fetches repository metadata (including the file list) at a revision.
pub fn repo_info(repo_type: RepoType, repo_id: &str, revision: Option<&str>) -> Result<RepoInfo> {
    if is_offline() {
        bail!("Cannot fetch info for {repo_id}: HF_HUB_OFFLINE is set");
    }
    let mut url = format!("{}/api/{}/{}", endpoint(), repo_type.api_segment(), repo_id);
    if let Some(revision) = revision {
        url.push_str(&format!("/revision/{}", urlencode(revision)));
    }
    url.push_str("?blobs=true");

    let token = token();
    let mut response = with_auth(agent().get(&url), token.as_deref())
        .call()
        .with_context(|| format!("Failed to contact the Hugging Face Hub for {repo_id}"))?;
    check_status(&response, &format!("Repository {repo_id}"))?;
    let info: RepoInfo = response
        .body_mut()
        .read_json()
        .with_context(|| format!("Invalid repository info for {repo_id}"))?;
    Ok(info)
}

fn urlencode(text: &str) -> String {
    let mut out = String::new();
    for byte in text.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(byte as char),
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

fn is_commit_hash(revision: &str) -> bool {
    revision.len() == 40 && revision.chars().all(|c| c.is_ascii_hexdigit())
}

/// Resolves a revision (branch, tag, or commit) to a commit hash, using the cache when offline.
pub fn resolve_revision(repo_type: RepoType, repo_id: &str, revision: Option<&str>) -> Result<String> {
    let revision = revision.unwrap_or("main");
    if is_commit_hash(revision) {
        return Ok(revision.to_string());
    }

    let ref_file = repo_cache_dir(repo_type, repo_id).join("refs").join(revision);

    if !is_offline() {
        match repo_info(repo_type, repo_id, Some(revision)) {
            Ok(info) => {
                write_ref(&ref_file, &info.sha);
                return Ok(info.sha);
            }
            Err(error) => {
                if let Ok(sha) = std::fs::read_to_string(&ref_file) {
                    return Ok(sha.trim().to_string());
                }
                return Err(error);
            }
        }
    }

    std::fs::read_to_string(&ref_file)
        .map(|sha| sha.trim().to_string())
        .map_err(|_| anyhow!("{repo_id}@{revision} is not in the local cache and HF_HUB_OFFLINE is set"))
}

fn write_ref(ref_file: &Path, sha: &str) {
    if let Some(parent) = ref_file.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(ref_file, sha);
}

/// Returns the cached path of a file if it exists.
pub fn cached_file(repo_type: RepoType, repo_id: &str, commit: &str, filename: &str) -> Option<PathBuf> {
    let path = repo_cache_dir(repo_type, repo_id)
        .join("snapshots")
        .join(commit)
        .join(filename);
    path.is_file().then_some(path)
}

/// Downloads a file (or returns it from the cache) and returns its local path.
pub fn download(
    repo_type: RepoType,
    repo_id: &str,
    revision: Option<&str>,
    filename: &str,
    progress: Option<ProgressFn>,
) -> Result<PathBuf> {
    let commit = resolve_revision(repo_type, repo_id, revision)?;

    if let Some(path) = cached_file(repo_type, repo_id, &commit, filename) {
        return Ok(path);
    }

    if is_offline() {
        bail!("{filename} from {repo_id} is not in the local cache and HF_HUB_OFFLINE is set");
    }

    let target = repo_cache_dir(repo_type, repo_id)
        .join("snapshots")
        .join(&commit)
        .join(filename);
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("Failed to create {}", parent.display()))?;
    }

    let url = format!(
        "{}/{}{}/resolve/{}/{}",
        endpoint(),
        repo_type.url_prefix(),
        repo_id,
        commit,
        filename.split('/').map(urlencode).collect::<Vec<_>>().join("/")
    );

    let partial = target.with_extension(format!(
        "{}incomplete",
        target
            .extension()
            .map(|e| format!("{}.", e.to_string_lossy()))
            .unwrap_or_default()
    ));
    let mut downloaded = std::fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);

    let token = token();
    let mut request = with_auth(agent().get(&url), token.as_deref());
    if downloaded > 0 {
        request = request.header("Range", &format!("bytes={downloaded}-"));
    }

    let mut response = request
        .config()
        .timeout_recv_body(None)
        .build()
        .call()
        .with_context(|| format!("Failed to download {filename} from {repo_id}"))?;
    check_status(&response, &format!("{repo_id}/{filename}"))?;

    let resumed = response.status().as_u16() == 206;
    if !resumed {
        downloaded = 0;
    }

    let total = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .map(|length| length + downloaded);

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(resumed)
        .truncate(!resumed)
        .open(&partial)
        .with_context(|| format!("Failed to create {}", partial.display()))?;

    let mut reader = response.body_mut().as_reader();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let count = reader
            .read(&mut buffer)
            .with_context(|| format!("Download of {filename} was interrupted"))?;
        if count == 0 {
            break;
        }
        file.write_all(&buffer[..count])?;
        downloaded += count as u64;
        if let Some(progress) = progress {
            progress(downloaded, total);
        }
    }
    file.flush()?;
    drop(file);

    if let Some(total) = total
        && downloaded != total
    {
        bail!("Download of {filename} is incomplete ({downloaded} of {total} bytes)");
    }

    std::fs::rename(&partial, &target).with_context(|| format!("Failed to move {} into place", target.display()))?;
    Ok(target)
}

/// A repository resolved to a local directory, either a cached snapshot or a local path.
#[derive(Debug, Clone)]
pub struct ResolvedRepo {
    pub id: String,
    /// Commit hash for Hub repositories, `None` for local directories.
    pub commit: Option<String>,
    /// Files available in the repository (relative paths).
    pub files: Vec<String>,
    local_dir: Option<PathBuf>,
    repo_type: RepoType,
}

impl ResolvedRepo {
    /// Resolves a model/dataset given as a Hub ID or a local directory.
    pub fn open(repo_type: RepoType, path_or_id: &str, revision: Option<&str>) -> Result<ResolvedRepo> {
        let local = Path::new(path_or_id);
        if local.is_dir() {
            let mut files = Vec::new();
            collect_files(local, local, &mut files)?;
            files.sort();
            return Ok(ResolvedRepo {
                id: path_or_id.to_string(),
                commit: None,
                files,
                local_dir: Some(local.to_path_buf()),
                repo_type,
            });
        }

        validate_repo_id(path_or_id)
            .with_context(|| format!("{path_or_id} is neither a local directory nor a valid repository ID"))?;

        if !is_offline() {
            match repo_info(repo_type, path_or_id, revision) {
                Ok(info) => {
                    if let Some(revision) = revision.filter(|r| !is_commit_hash(r)) {
                        write_ref(
                            &repo_cache_dir(repo_type, path_or_id).join("refs").join(revision),
                            &info.sha,
                        );
                    } else if revision.is_none() {
                        write_ref(
                            &repo_cache_dir(repo_type, path_or_id).join("refs").join("main"),
                            &info.sha,
                        );
                    }
                    let mut files: Vec<String> = info.siblings.into_iter().map(|s| s.rfilename).collect();
                    files.sort();
                    return Ok(ResolvedRepo {
                        id: path_or_id.to_string(),
                        commit: Some(info.sha),
                        files,
                        local_dir: None,
                        repo_type,
                    });
                }
                Err(error) if resolve_cached_snapshot(repo_type, path_or_id, revision).is_none() => return Err(error),
                Err(_) => {}
            }
        }

        // Offline (or the Hub is unreachable): fall back to the cached snapshot.
        let (commit, snapshot) = resolve_cached_snapshot(repo_type, path_or_id, revision)
            .ok_or_else(|| anyhow!("{path_or_id} is not in the local cache and the Hub is unavailable"))?;
        let mut files = Vec::new();
        collect_files(&snapshot, &snapshot, &mut files)?;
        files.sort();
        Ok(ResolvedRepo {
            id: path_or_id.to_string(),
            commit: Some(commit),
            files,
            local_dir: None,
            repo_type,
        })
    }

    pub fn is_local(&self) -> bool {
        self.local_dir.is_some()
    }

    pub fn has_file(&self, filename: &str) -> bool {
        self.files.iter().any(|f| f == filename)
    }

    /// Returns the local path of a file, downloading it if necessary.
    pub fn get(&self, filename: &str, progress: Option<ProgressFn>) -> Result<PathBuf> {
        if let Some(dir) = &self.local_dir {
            let path = dir.join(filename);
            if !path.is_file() {
                bail!("{} does not exist", path.display());
            }
            return Ok(path);
        }
        download(self.repo_type, &self.id, self.commit.as_deref(), filename, progress)
    }

    /// Like [`get`](Self::get), but returns `None` if the repository doesn't contain the file.
    pub fn get_optional(&self, filename: &str) -> Result<Option<PathBuf>> {
        if !self.has_file(filename) {
            return Ok(None);
        }
        self.get(filename, None).map(Some)
    }
}

fn resolve_cached_snapshot(repo_type: RepoType, repo_id: &str, revision: Option<&str>) -> Option<(String, PathBuf)> {
    let base = repo_cache_dir(repo_type, repo_id);
    let revision = revision.unwrap_or("main");
    let commit = if is_commit_hash(revision) {
        revision.to_string()
    } else {
        std::fs::read_to_string(base.join("refs").join(revision))
            .ok()?
            .trim()
            .to_string()
    };
    let snapshot = base.join("snapshots").join(&commit);
    snapshot.is_dir().then_some((commit, snapshot))
}

fn collect_files(root: &Path, dir: &Path, files: &mut Vec<String>) -> Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("Failed to read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        if name.to_string_lossy().starts_with('.') {
            continue;
        }
        if path.is_dir() {
            collect_files(root, &path, files)?;
        } else {
            let relative = path.strip_prefix(root).unwrap_or(&path);
            files.push(relative.to_string_lossy().replace('\\', "/"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_ids() {
        assert!(validate_repo_id("Qwen/Qwen3-4B-Instruct-2507").is_ok());
        assert!(validate_repo_id("gpt2").is_ok());
        assert!(validate_repo_id("a/b/c").is_err());
        assert!(validate_repo_id("a--b/c").is_err());
        assert!(validate_repo_id("bad id").is_err());
        assert!(validate_repo_id("").is_err());
    }

    #[test]
    fn local_paths_are_not_hf_paths() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!is_hf_path(dir.path().to_str().unwrap()));
        assert!(is_hf_path("org/model"));
    }

    #[test]
    fn local_directory_repo() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        std::fs::write(dir.path().join("sub/x.txt"), "x").unwrap();
        let repo = ResolvedRepo::open(RepoType::Model, dir.path().to_str().unwrap(), None).unwrap();
        assert!(repo.is_local());
        assert_eq!(repo.files, ["config.json", "sub/x.txt"]);
        assert!(repo.get("config.json", None).unwrap().is_file());
        assert!(repo.get_optional("missing.json").unwrap().is_none());
    }
}
