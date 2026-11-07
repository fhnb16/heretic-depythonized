# Building Heretic (Rust)

Heretic is a Cargo workspace. No Python is required at any point.

| Crate | Purpose |
| :---- | :------ |
| `crates/heretic-core` | Settings (same TOML schema as Heretic 2.x), datasets, Hugging Face Hub client, tokenizer, chat templates |
| `crates/heretic-engine` | Model runtime on [candle](https://github.com/huggingface/candle): safetensors loading, LoRA adapters, generation, residuals, module I/O |
| `crates/heretic-optim` | Port of Optuna's multivariate multi-objective TPE sampler, study journal, L-BFGS, randomized low-rank SVD |
| `crates/heretic-plugins` | Modifiers (ARA, Abliteration) and scorers (KeywordRate, KLDivergence, ResponseLength) |
| `crates/heretic-export` | Merged/adapter export (streaming safetensors writer), model cards, Hugging Face upload |
| `crates/heretic-session` | Orchestration of a run, shared by the CLI and the GUI |
| `apps/heretic-cli` | The `heretic` command line program |
| `apps/heretic-gui` | Desktop application (Tauri 2, frontend in plain HTML/CSS/JS without a build step) |

## Prerequisites

* [Rust](https://rustup.rs) (stable, 1.88 or newer)
* For NVIDIA GPUs: the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 12.x (`nvcc`
  must be on `PATH` at build time; on Windows also the MSVC build tools)
* For the GUI on Linux: the WebKitGTK development packages (`libwebkit2gtk-4.1-dev librsvg2-dev`)

Neither Node.js nor Python is needed. You don't need any of this to *use* Heretic: the
[releases](../../../releases) contain ready builds, including CUDA builds with the CUDA runtime
libraries bundled (only an NVIDIA driver is required).

## Command line

```sh
# CPU only
cargo build --release -p heretic

# With CUDA
cargo build --release -p heretic --features cuda

./target/release/heretic Qwen/Qwen3-4B-Instruct-2507
```

All settings from `config.default.toml` work as before, either in `config.toml`, as
`HERETIC_<NAME>` environment variables, or as `--option-name value` arguments. The presets
live in [`presets/`](../presets). Plugin names from Heretic 2.x such as
`heretic.modifiers.ara.ARA` are accepted; Python plugin files are not supported.

## Desktop application

The frontend (`apps/heretic-gui/ui/`) is embedded into the executable, so a plain Cargo build
produces a working application:

```sh
cargo run --release -p heretic-gui                    # run (CPU)
cargo run --release -p heretic-gui --features cuda    # run with CUDA
```

Installers are built with the Tauri CLI, which is installed through Cargo:

```sh
cargo install tauri-cli --version "^2" --locked
cd apps/heretic-gui/src-tauri
cargo tauri build --bundles nsis                      # Windows installer (CPU)
cargo tauri build --bundles nsis --features cuda      # Windows installer with CUDA
cargo tauri build --bundles deb,appimage              # Linux packages
```

Editing the UI needs no tooling: change files in `ui/` and restart the app. To check every
screen without the backend, render them with a mocked API in headless Edge (Windows):

```powershell
pwsh apps/heretic-gui/ui-test/screenshots.ps1 [-Theme dark] [-Skin fluent] [-Lang en]
```

Screenshots land in `target/ui-shots/`; the script fails on any JavaScript error.

## Releases

[`.github/workflows/release.yml`](../.github/workflows/release.yml) tags and publishes every
commit on `master`. The version (`[workspace.package] version` in `Cargo.toml`) follows semver
and is bumped together with the change (patch: fixes, minor: features, major: breaking).

* The first commit of a version gets the tag `v<version>` and a regular release.
* Later commits without a version bump get `v<version>.<n>` (n = commits since `v<version>`),
  published as pre-releases that are never marked "latest".

Every release contains:

* Windows: CPU and CUDA installers (NSIS) and portable zips with `heretic.exe` and `heretic-gui.exe`
* Linux: CPU AppImage and .deb, CPU and CUDA tarballs

Versions with a semver suffix (`2.2.0-rc.1`) are published as pre-releases. Pushing a `v*`
tag by hand or running the workflow manually with "force" rebuilds an existing tag.

### CUDA builds

candle compiles its CUDA kernels with `nvcc` at build time, so there are no prebuilt
binaries to download; the release workflow installs the CUDA Toolkit on the runner and
compiles them there. Details:

* Kernels are compiled for compute capability 8.0 (`CUDA_COMPUTE_CAP=80`), which runs on
  RTX 30xx and newer (sm_80+). Older GPUs need a local build with a matching
  `CUDA_COMPUTE_CAP` (below 80, bf16 kernels are disabled).
* The CUDA libraries are loaded at runtime, so the redistributable runtime libraries
  (cudart, cuBLAS, cuBLASLt, cuRAND, NVRTC) ship next to the executables: as DLLs on Windows,
  as `.so` files with an `$ORIGIN` rpath on Linux. Users only need the NVIDIA driver.

## Tests

```sh
cargo test --workspace                     # unit tests
cargo test --release -p heretic-session --test session -- --ignored   # end-to-end, tiny model
cargo test --release -p heretic-engine --test reference -- --ignored  # parity with candle-transformers
cargo test --release -p heretic-engine --test real_model -- --ignored # real models on CPU
cargo test --release -p heretic-export --test roundtrip -- --ignored  # export round trip
```

```powershell
pwsh apps/heretic-gui/ui-test/screenshots.ps1   # GUI screens (Windows, headless Edge)
```

The ignored tests download models and datasets from the Hugging Face Hub.

## Supported architectures

Currently `llama`, `mistral`, `qwen2` and `qwen3` (including multimodal checkpoints whose
text model uses one of these architectures). More families (Gemma, MoE models, hybrids) are
planned.

## Differences from Heretic 2.x (Python)

* Reproducibility manifests (`reproduce.json` v4) of the Python version cannot be reproduced
  bit-for-bit, because random number generators and numerical kernels differ.
* `quantization = "bnb_4bit"` is not implemented yet.
* Explicit device maps (multi-GPU) are not implemented yet; `device_map` accepts `auto`,
  `cpu`, `cuda` and `cuda:<index>`.
* The BERTClassifier and BenchmarkScore scorers and the benchmark menu are not implemented yet.
