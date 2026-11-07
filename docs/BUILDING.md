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
| `apps/heretic-gui` | Desktop application (Tauri 2 + Svelte 5) |

## Prerequisites

* [Rust](https://rustup.rs) (stable, 1.88 or newer)
* For NVIDIA GPUs: the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (`nvcc` must be on `PATH` at build time)
* For the GUI: [Node.js](https://nodejs.org) 20+ and, on Linux, the WebKitGTK development packages
  (`libwebkit2gtk-4.1-dev librsvg2-dev`)

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

```sh
cd apps/heretic-gui
npm install
npm run tauri dev                         # development
npm run tauri build                       # installer (CPU)
npm run tauri build -- --features cuda    # installer with CUDA support
```

## Tests

```sh
cargo test --workspace                     # unit tests
cargo test --release -p heretic-session --test session -- --ignored   # end-to-end, tiny model
cargo test --release -p heretic-engine --test reference -- --ignored  # parity with candle-transformers
cargo test --release -p heretic-engine --test real_model -- --ignored # real models on CPU
cargo test --release -p heretic-export --test roundtrip -- --ignored  # export round trip

cd apps/heretic-gui && npm test            # frontend unit tests
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
