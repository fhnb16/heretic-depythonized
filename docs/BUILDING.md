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

### Update check in the desktop application

The GUI checks the GitHub releases of `fhnb16/heretic-depythonized` for a newer build
(`apps/heretic-gui/src-tauri/src/update.rs`, frontend `ui/js/update.js`):

* On start, at most once every 6 hours (Settings → About: "Check for updates on start"), and
  on demand with "Check for updates". Pre-releases (`v<version>.<n>`) count only when
  "Include builds between versions" is on. The workflow compiles the release tag into the
  binary (`HERETIC_BUILD_TAG`), so a `v2.1.4.3` build knows it is newer than `v2.1.4`; a local
  build compares its `Cargo.toml` version.
* A new release shows a quiet notice above the run status. "Update" downloads the file for this
  build (same OS and CPU/CUDA variant), verifies it against `SHA256SUMS.txt`, then:
  * Windows, installed with the NSIS installer: starts the new installer and closes the app;
  * Linux AppImage: replaces the AppImage file and restarts it;
  * portable zips, tarballs and .deb: opens the release page in the browser.
* There is no signing key, so the checksum list from the same release is the only integrity
  check. Unauthenticated GitHub API calls are limited to 60 per hour per address.

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
cargo test --release -p heretic-export --test gguf -- --ignored       # GGUF export (+ quantization
                                                                      # if HERETIC_TEST_LLAMA_CPP is set)
```

```powershell
pwsh apps/heretic-gui/ui-test/screenshots.ps1   # GUI screens (Windows, headless Edge)
```

The ignored tests download models and datasets from the Hugging Face Hub.

## Supported architectures

| Family | `model_type` (text model) | Notes |
| :--- | :--- | :--- |
| Llama 2/3, Mistral | `llama`, `mistral` | |
| Qwen2/2.5, Qwen3 | `qwen2`, `qwen3` | |
| Qwen3.5 (dense) | `qwen3_5_text` | Hybrid Gated DeltaNet + gated attention; `linear_attn.out_proj` counts as `attn.o_proj` |
| Gemma, Gemma 2, Gemma 3 | `gemma`, `gemma2`, `gemma3_text` | Softcapping, sliding windows, local/global RoPE |
| Gemma 4 (dense) | `gemma4_text` | Per-layer embeddings, shared KV layers, proportional RoPE, `attention_k_eq_v` |

Multimodal checkpoints work when their text model is one of these; only the text model is
loaded and modified, the other weights are copied unchanged on export. Mixture-of-experts
models (Qwen3-MoE, Qwen3.5-MoE, Mixtral, gpt-oss, Gemma 4 MoE) and Gemma 3n are not supported
yet; loading them fails with an explanation.

Models must be in Transformers format (`config.json` + `*.safetensors`); GGUF files can't be
used as input. Smoke tests for every family (loading, padding invariance, cached generation):

```sh
cargo test --release -p heretic-engine --test architectures -- --ignored --nocapture
HERETIC_ARCH_MODELS=Qwen/Qwen3.5-0.8B,unsloth/gemma-3-1b-it cargo test --release -p heretic-engine --test architectures -- --ignored --nocapture
```

## GGUF export and llama.cpp

Modified models can be saved as a single GGUF file for llama.cpp, LM Studio or Ollama
(CLI: "Save the model as GGUF", `model_action = "gguf"`; GUI: Save, "GGUF for llama.cpp").

* The conversion to GGUF is done by Heretic itself, in Rust (`crates/heretic-export/src/gguf`):
  a port of llama.cpp's `convert_hf_to_gguf.py` for the supported architectures — the same
  tensor names, transformations (Llama Q/K permutation, Gemma/Qwen3.5 norm offsets, Qwen3.5
  `A_log`, conv and V-head reordering), metadata keys and tokenizer metadata. The BPE
  pre-tokenizer is detected with the same hash table as upstream
  (`gguf/pre_tokenizer.rs`, generated from `get_vocab_base_pre`; regenerate it when llama.cpp
  adds tokenizers). F16 and BF16 files need nothing else.
* Quantized types (Q8_0, Q6_K, Q5_K_M, Q5_K_S, Q4_K_M, Q4_K_S, Q4_0, Q3_K_M, Q2_K) are made by
  `llama-quantize` from a llama.cpp build you download from
  <https://github.com/ggml-org/llama.cpp/releases> (any CPU build is enough). Point Heretic at
  the unpacked folder with `llama_cpp_dir` (`--llama-cpp-dir`, `HERETIC_LLAMA_CPP_DIR`, or
  Settings → llama.cpp in the GUI). Heretic writes an unquantized BF16/F16 GGUF next to the
  output, runs `llama-quantize <in> <out> <TYPE> <threads>` and deletes the intermediate file.
* `gguf_type` selects the type non-interactively; the output is a folder (the file is named
  `<model>-heretic-<TYPE>.gguf`) or a path ending in `.gguf`.
* Supported for GGUF export: Llama 2/3 and Mistral (`llama`), Qwen2/2.5 (`qwen2`), Qwen3
  (`qwen3`), Qwen3.5 dense (`qwen35`, without the MTP head), Gemma 2 and Gemma 3 (text model).
  Tokenizers: BPE from `tokenizer.json` known to llama.cpp, SentencePiece `tokenizer.model`, and
  Llama-style byte-fallback BPE. Gemma 4, mixture-of-experts models and vision towers are not
  exported to GGUF yet.

## Differences from Heretic 2.x (Python)

* Reproducibility manifests (`reproduce.json` v4) of the Python version cannot be reproduced
  bit-for-bit, because random number generators and numerical kernels differ.
* `quantization = "bnb_4bit"` is not implemented yet.
* Explicit device maps (multi-GPU) are not implemented yet; `device_map` accepts `auto`,
  `cpu`, `cuda` and `cuda:<index>`.
* The BERTClassifier and BenchmarkScore scorers and the benchmark menu are not implemented yet.
