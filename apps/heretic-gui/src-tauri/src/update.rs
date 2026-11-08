// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Update check and installation from the GitHub releases of the project.
//!
//! Releases are published by `.github/workflows/release.yml`: the first commit of a version
//! gets the tag `v<version>` (a regular release), later commits `v<version>.<n>` (pre-releases).
//! Assets are named `heretic-<tag without v>-<os>-x64-<cpu|cuda12>[-setup.exe|.zip|.AppImage|...]`
//! and listed in `SHA256SUMS.txt`. There are no update signatures, so a downloaded file is only
//! checked against that list, and installing always goes through the regular installer.

use std::cmp::Ordering;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use tauri::AppHandle;

const REPOSITORY: &str = "fhnb16/heretic-depythonized";
const CHECKSUMS_ASSET: &str = "SHA256SUMS.txt";
const EVENT: &str = "heretic://update";

/// The version of this build, including the build number of pre-release builds
/// (`HERETIC_BUILD_TAG` is set by the release workflow, e.g. `v2.1.4.3`).
pub fn current_version() -> &'static str {
    match option_env!("HERETIC_BUILD_TAG") {
        Some(tag) if !tag.is_empty() => tag.trim_start_matches('v'),
        _ => env!("CARGO_PKG_VERSION"),
    }
}

/// A version like `2.1.4`, `2.1.4.3` (build 3 after the 2.1.4 tag) or `2.2.0-dev.1`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Version {
    numbers: [u64; 4],
    /// Text after `-`; a pre-release sorts before the same numbers without one.
    pre: Option<String>,
}

impl Version {
    fn parse(text: &str) -> Option<Version> {
        let text = text.trim().trim_start_matches('v');
        let (core, pre) = match text.split_once('-') {
            Some((core, pre)) => (core, Some(pre.to_string())),
            None => (text, None),
        };
        let parts: Vec<&str> = core.split('.').collect();
        if parts.is_empty() || parts.len() > 4 {
            return None;
        }
        let mut numbers = [0u64; 4];
        for (slot, part) in numbers.iter_mut().zip(&parts) {
            *slot = part.parse().ok()?;
        }
        Some(Version { numbers, pre })
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> Ordering {
        self.numbers
            .cmp(&other.numbers)
            .then_with(|| match (&self.pre, &other.pre) {
                (None, None) => Ordering::Equal,
                (None, Some(_)) => Ordering::Greater,
                (Some(_), None) => Ordering::Less,
                (Some(a), Some(b)) => compare_pre(a, b),
            })
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Compares pre-release identifiers like semver: numeric parts numerically, others as text.
fn compare_pre(a: &str, b: &str) -> Ordering {
    for (x, y) in a.split('.').zip(b.split('.')) {
        let ordering = match (x.parse::<u64>(), y.parse::<u64>()) {
            (Ok(x), Ok(y)) => x.cmp(&y),
            (Ok(_), Err(_)) => Ordering::Less,
            (Err(_), Ok(_)) => Ordering::Greater,
            (Err(_), Err(_)) => x.cmp(y),
        };
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    a.split('.').count().cmp(&b.split('.').count())
}

#[derive(Deserialize)]
struct GithubRelease {
    tag_name: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    body: Option<String>,
    html_url: String,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
    #[serde(default)]
    published_at: Option<String>,
    #[serde(default)]
    assets: Vec<GithubAsset>,
}

#[derive(Deserialize, Clone)]
struct GithubAsset {
    name: String,
    size: u64,
    browser_download_url: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct UpdateAsset {
    pub name: String,
    pub size: u64,
    pub url: String,
}

/// How an update can be installed on this system.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum InstallKind {
    /// Download and run the Windows installer.
    Installer,
    /// Download and replace the running AppImage.
    Appimage,
    /// Open the release page; the user installs by hand (portable builds, .deb, other systems).
    Browser,
}

#[derive(Serialize, Debug)]
pub struct UpdateCheck {
    /// Version of this build.
    pub current: String,
    pub available: bool,
    pub latest: Option<String>,
    pub tag: Option<String>,
    pub name: Option<String>,
    pub prerelease: bool,
    /// Release notes (Markdown).
    pub notes: Option<String>,
    /// Release page.
    pub url: Option<String>,
    pub published_at: Option<String>,
    /// The file matching this build, if the release has one.
    pub asset: Option<UpdateAsset>,
    pub checksums_url: Option<String>,
    pub install: InstallKind,
}

fn agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(20)))
            .timeout_recv_response(Some(Duration::from_secs(60)))
            .http_status_as_error(false)
            .user_agent(concat!("heretic-gui/", env!("CARGO_PKG_VERSION")))
            .build()
            .into()
    })
}

/// Turns a non-success response into a message that says what happened and what to do.
fn status_error(status: u16, remaining: Option<&str>, what: &str) -> String {
    match status {
        403 | 429 if remaining == Some("0") || status == 429 => {
            "GitHub limits update checks to 60 per hour from one address; try again later".to_string()
        }
        404 => format!("{what} was not found on GitHub (repository {REPOSITORY})"),
        _ => format!("GitHub answered HTTP {status} for {what}"),
    }
}

fn get(url: &str, what: &str) -> Result<ureq::http::Response<ureq::Body>, String> {
    let response = agent()
        .get(url)
        .header("Accept", "application/vnd.github+json")
        .call()
        .map_err(|e| format!("Could not reach GitHub to {what}: {e}. Check the internet connection or a proxy"))?;
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        let remaining = response
            .headers()
            .get("x-ratelimit-remaining")
            .and_then(|v| v.to_str().ok());
        return Err(status_error(status, remaining, what));
    }
    Ok(response)
}

/// Suffix of the asset built with the same features as this binary.
fn variant() -> &'static str {
    if cfg!(feature = "cuda") { "cuda12" } else { "cpu" }
}

/// Whether this Windows executable was installed by the NSIS installer (and not unpacked from
/// the portable zip): the installer puts its uninstaller next to the executable.
fn installed_with_nsis() -> bool {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("uninstall.exe").is_file()))
        .unwrap_or(false)
}

/// Picks the asset for this build and how to install it.
fn choose_asset(assets: &[GithubAsset]) -> (Option<GithubAsset>, InstallKind) {
    let find = |suffix: String| assets.iter().find(|a| a.name.ends_with(&suffix)).cloned();
    let variant = variant();
    if cfg!(target_os = "windows") {
        if installed_with_nsis()
            && let Some(asset) = find(format!("-windows-x64-{variant}-setup.exe"))
        {
            return (Some(asset), InstallKind::Installer);
        }
        return (find(format!("-windows-x64-{variant}.zip")), InstallKind::Browser);
    }
    if cfg!(target_os = "linux") {
        if std::env::var_os("APPIMAGE").is_some()
            && let Some(asset) = find(format!("-linux-x64-{variant}.AppImage"))
        {
            return (Some(asset), InstallKind::Appimage);
        }
        return (find(format!("-linux-x64-{variant}.tar.gz")), InstallKind::Browser);
    }
    (None, InstallKind::Browser)
}

/// Finds the newest release (optionally including pre-releases) and compares it with this build.
pub fn check(include_prereleases: bool) -> Result<UpdateCheck, String> {
    let current = current_version().to_string();
    let url = format!("https://api.github.com/repos/{REPOSITORY}/releases?per_page=40");
    let mut response = get(&url, "list releases")?;
    let releases: Vec<GithubRelease> = response
        .body_mut()
        .read_json()
        .map_err(|e| format!("GitHub returned an unexpected release list: {e}"))?;

    let newest = releases
        .into_iter()
        .filter(|r| !r.draft && (include_prereleases || !r.prerelease))
        .filter_map(|r| Version::parse(&r.tag_name).map(|v| (v, r)))
        .max_by(|(a, _), (b, _)| a.cmp(b));

    let Some((version, release)) = newest else {
        return Ok(UpdateCheck {
            current,
            available: false,
            latest: None,
            tag: None,
            name: None,
            prerelease: false,
            notes: None,
            url: None,
            published_at: None,
            asset: None,
            checksums_url: None,
            install: InstallKind::Browser,
        });
    };

    let available = Version::parse(&current).is_none_or(|current| version > current);
    let (asset, install) = choose_asset(&release.assets);
    let checksums_url = release
        .assets
        .iter()
        .find(|a| a.name == CHECKSUMS_ASSET)
        .map(|a| a.browser_download_url.clone());
    Ok(UpdateCheck {
        current,
        available,
        latest: Some(release.tag_name.trim_start_matches('v').to_string()),
        tag: Some(release.tag_name.clone()),
        name: release.name,
        prerelease: release.prerelease,
        notes: release.body,
        url: Some(release.html_url),
        published_at: release.published_at,
        // Without an asset for this build the user can only pick a file on the release page.
        install: if asset.is_some() { install } else { InstallKind::Browser },
        asset: asset.map(|a| UpdateAsset {
            name: a.name,
            size: a.size,
            url: a.browser_download_url,
        }),
        checksums_url,
    })
}

/// Opens a web page in the default browser.
pub fn open_url(url: &str) -> Result<(), String> {
    if !(url.starts_with("https://") || url.starts_with("http://")) {
        return Err(format!("Refusing to open {url}"));
    }
    let result = if cfg!(target_os = "windows") {
        std::process::Command::new("rundll32")
            .args(["url.dll,FileProtocolHandler", url])
            .spawn()
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(url).spawn()
    } else {
        std::process::Command::new("xdg-open").arg(url).spawn()
    };
    result
        .map(|_| ())
        .map_err(|e| format!("Could not open the browser ({e}); open {url} by hand"))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Looks up a file in a `sha256sum` listing (`<hash>  <name>` or `<hash> *<name>`).
fn expected_hash(listing: &str, name: &str) -> Option<String> {
    listing.lines().find_map(|line| {
        let (hash, file) = line.trim().split_once(char::is_whitespace)?;
        let file = file.trim_start().trim_start_matches('*');
        (file == name).then(|| hash.to_ascii_lowercase())
    })
}

fn emit_progress(app: &AppHandle, stage: &str, done: u64, total: Option<u64>) {
    crate::emit(app, EVENT, json!({ "stage": stage, "done": done, "total": total }));
}

/// Downloads a file with progress events and returns its SHA-256.
fn download(app: &AppHandle, asset: &UpdateAsset, path: &Path) -> Result<String, String> {
    let mut response = get(&asset.url, &format!("download {}", asset.name))?;
    let total = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .or(Some(asset.size));
    let mut file = std::fs::File::create(path).map_err(|e| format!("Could not create {}: {e}", path.display()))?;
    let mut reader = response.body_mut().as_reader();
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    let mut done = 0u64;
    let mut last_emit = std::time::Instant::now();
    emit_progress(app, "downloading", 0, total);
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|e| format!("The download of {} was interrupted: {e}", asset.name))?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])
            .map_err(|e| format!("Could not write {}: {e}", path.display()))?;
        hasher.update(&buffer[..read]);
        done += read as u64;
        if last_emit.elapsed() > Duration::from_millis(150) {
            emit_progress(app, "downloading", done, total);
            last_emit = std::time::Instant::now();
        }
    }
    file.flush()
        .map_err(|e| format!("Could not write {}: {e}", path.display()))?;
    emit_progress(app, "downloading", done, total);
    if let Some(total) = total
        && done != total
    {
        return Err(format!(
            "The download of {} is incomplete ({done} of {total} bytes); try again",
            asset.name
        ));
    }
    Ok(hex(&hasher.finalize()))
}

/// Request from the frontend: the parts of an [`UpdateCheck`] needed to install.
#[derive(Deserialize)]
pub struct InstallRequest {
    pub asset: Option<UpdateAsset>,
    pub checksums_url: Option<String>,
    pub install: InstallKind,
    pub url: Option<String>,
}

/// What happened, so the frontend can tell the user.
#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
pub enum InstallOutcome {
    /// The release page was opened in the browser.
    Opened,
    /// The installer or the new AppImage was started; the app is about to exit.
    Restarting,
}

fn update_dir() -> Result<PathBuf, String> {
    let dir = std::env::temp_dir().join("heretic-update");
    std::fs::create_dir_all(&dir).map_err(|e| format!("Could not create {}: {e}", dir.display()))?;
    Ok(dir)
}

/// Downloads, verifies and starts the update. Exits the app when the installer has started.
pub fn install(app: &AppHandle, request: InstallRequest) -> Result<InstallOutcome, String> {
    let (asset, kind) = match (&request.asset, request.install) {
        (Some(asset), InstallKind::Installer | InstallKind::Appimage) => (asset.clone(), request.install),
        _ => {
            let url = request.url.ok_or("The release has no page to open")?;
            open_url(&url)?;
            return Ok(InstallOutcome::Opened);
        }
    };
    if asset.name.contains(['/', '\\']) || asset.name.starts_with('.') {
        return Err(format!("Unexpected file name {}", asset.name));
    }

    let path = update_dir()?.join(&asset.name);
    let hash = download(app, &asset, &path).inspect_err(|_| {
        let _ = std::fs::remove_file(&path);
    })?;

    emit_progress(app, "verifying", 0, None);
    if let Some(url) = &request.checksums_url {
        let mut response = get(url, "download the checksums")?;
        let listing = response
            .body_mut()
            .read_to_string()
            .map_err(|e| format!("Could not read the checksums: {e}"))?;
        match expected_hash(&listing, &asset.name) {
            Some(expected) if expected == hash => {}
            Some(expected) => {
                let _ = std::fs::remove_file(&path);
                return Err(format!(
                    "The downloaded {} is damaged: SHA-256 {hash}, expected {expected}. Nothing was installed",
                    asset.name
                ));
            }
            None => {
                return Err(format!(
                    "{} is not listed in {CHECKSUMS_ASSET}; nothing was installed",
                    asset.name
                ));
            }
        }
    }

    emit_progress(app, "installing", 0, None);
    match kind {
        InstallKind::Installer => start_installer(&path)?,
        InstallKind::Appimage => replace_appimage(&path)?,
        InstallKind::Browser => unreachable!(),
    }

    // Give the frontend a moment to show the outcome, then quit so the installer can
    // replace the executable.
    let handle = app.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(800));
        handle.exit(0);
    });
    Ok(InstallOutcome::Restarting)
}

#[cfg(target_os = "windows")]
fn start_installer(path: &Path) -> Result<(), String> {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    // `start` goes through ShellExecute, which shows the UAC prompt if the installer asks
    // for elevation (a plain CreateProcess would fail with ERROR_ELEVATION_REQUIRED).
    std::process::Command::new("cmd")
        .args(["/C", "start", ""])
        .arg(path)
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("Could not start the installer {}: {e}", path.display()))
}

#[cfg(not(target_os = "windows"))]
fn start_installer(path: &Path) -> Result<(), String> {
    Err(format!(
        "Installers only run on Windows; open {} by hand",
        path.display()
    ))
}

#[cfg(unix)]
fn replace_appimage(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let target = std::env::var_os("APPIMAGE")
        .map(PathBuf::from)
        .ok_or("Not running as an AppImage")?;
    let staged = target.with_extension("AppImage.new");
    std::fs::copy(path, &staged).map_err(|e| {
        format!(
            "Could not write next to {} ({e}); is the folder writable?",
            target.display()
        )
    })?;
    std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o755))
        .map_err(|e| format!("Could not make {} executable: {e}", staged.display()))?;
    // Renaming over the running file is safe on Linux: the old inode stays open until exit.
    std::fs::rename(&staged, &target).map_err(|e| format!("Could not replace {}: {e}", target.display()))?;
    let _ = std::fs::remove_file(path);
    std::process::Command::new(&target)
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("Updated {}, but could not restart it: {e}", target.display()))
}

#[cfg(not(unix))]
fn replace_appimage(_path: &Path) -> Result<(), String> {
    Err("AppImages only run on Linux".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(text: &str) -> Version {
        Version::parse(text).unwrap()
    }

    #[test]
    fn orders_versions_and_build_numbers() {
        assert!(v("v2.1.4.3") > v("2.1.4"));
        assert!(v("v2.1.4.10") > v("v2.1.4.9"));
        assert!(v("v2.2.0") > v("v2.1.4.99"));
        assert!(v("2.1.0") > v("2.1.0-dev.0"));
        assert!(v("2.1.0-dev.10") > v("2.1.0-dev.9"));
        assert_eq!(v("v2.1.4"), v("2.1.4.0"));
        assert!(Version::parse("latest").is_none());
        assert!(Version::parse("1.2.3.4.5").is_none());
    }

    #[test]
    fn reads_sha256sum_listings() {
        let listing = "abc123  heretic-2.1.4-windows-x64-cpu-setup.exe\nDEF456 *heretic-2.1.4-linux-x64-cpu.AppImage\n";
        assert_eq!(
            expected_hash(listing, "heretic-2.1.4-windows-x64-cpu-setup.exe").as_deref(),
            Some("abc123")
        );
        assert_eq!(
            expected_hash(listing, "heretic-2.1.4-linux-x64-cpu.AppImage").as_deref(),
            Some("def456")
        );
        assert_eq!(expected_hash(listing, "missing.zip"), None);
    }

    #[test]
    fn explains_rate_limits() {
        assert!(status_error(403, Some("0"), "list releases").contains("60 per hour"));
        assert!(status_error(404, None, "list releases").contains(REPOSITORY));
        assert!(status_error(500, None, "list releases").contains("500"));
    }
}
