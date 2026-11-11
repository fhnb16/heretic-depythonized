// Mock of the Tauri API for testing the UI in a plain browser.
// Usage: ui-test/index.html?scenario=home|new|run|results|chat|settings|update|updating&lang=ru|en&theme=light|dark&skin=heretic|<theme id>

(function () {
  const params = new URLSearchParams(location.search);
  const scenario = params.get("scenario") ?? "home";
  localStorage.setItem("heretic.lang", JSON.stringify(params.get("lang") ?? "ru"));
  localStorage.setItem("heretic.theme", JSON.stringify(params.get("theme") ?? "light"));
  localStorage.setItem("heretic.skin", JSON.stringify(params.get("skin") ?? "heretic"));
  localStorage.setItem("heretic.checkpointDir", JSON.stringify("C:\\Users\\me\\Documents\\Heretic\\checkpoints"));
  localStorage.setItem("heretic.llamaCppDir", JSON.stringify("C:\\tools\\llama.cpp"));
  // Every scenario starts without a remembered update check, so the startup check always runs.
  for (const key of ["updateLastCheck", "updateCache", "updateDismissed", "updateAuto", "updatePrereleases"]) {
    localStorage.removeItem(`heretic.${key}`);
  }

  const errors = [];
  function report(message) {
    errors.push(message);
    let box = document.getElementById("test-errors");
    if (!box) {
      box = document.createElement("pre");
      box.id = "test-errors";
      box.style.cssText = "position:fixed;bottom:0;left:0;right:0;max-height:40%;overflow:auto;background:#fee;color:#900;margin:0;padding:8px;z-index:999;font-size:12px";
      document.body.append(box);
    }
    box.textContent = errors.join("\n");
  }
  window.addEventListener("error", (e) => report(`error: ${e.message} @ ${e.filename}:${e.lineno}`));
  window.addEventListener("unhandledrejection", (e) => report(`rejection: ${e.reason?.stack ?? e.reason}`));
  const originalError = console.error;
  console.error = (...args) => {
    report(`console.error: ${args.map((a) => a?.stack ?? String(a)).join(" ")}`);
    originalError(...args);
  };

  const listeners = {};
  const emit = (name, payload) => (listeners[name] ?? []).forEach((cb) => cb({ payload }));
  window.mockEmit = emit;

  const score = (value, display, md) => ({ value, display, md_display: md ?? display.replaceAll("**", "") });
  const names = ["Refusals", "KL divergence"];
  const trials = [
    [1, 3 / 32, 3.8, false], [2, 6 / 32, 0.938, true], [3, 0, 5.91, true], [4, 1 / 32, 0.79, true],
    [5, 0, 15.2, false], [6, 0, 16.5, false], [7, 0, 17.3, false], [8, 0, 10.5, false],
  ].map(([index, refusals, kl, pareto]) => ({
    number: index - 1,
    index,
    values: [refusals, kl],
    pareto_optimal: pareto,
    parameters: [
      ["start_layer_index", "3"], ["end_layer_index", "20"], ["preserve_good_behavior_weight", "0.4312"],
      ["steer_bad_behavior_weight", "0.0123"], ["overcorrect_relative_weight", "0.8810"], ["neighbor_count", "7"],
    ],
    scores: [
      { name: "Refusals", score: score(refusals, `**${Math.round(refusals * 32)}**/32`), baseline: score(30 / 32, "**30**/32") },
      { name: "KL divergence", score: score(kl, `**${kl.toFixed(4)}**`), baseline: score(0, "**0** (by definition)") },
    ],
  }));

  const settings = {
    model: "Qwen/Qwen2.5-0.5B-Instruct", n_trials: 100, dtypes: ["auto", "float16", "bfloat16", "float32"],
    device_map: "auto", max_response_length: 100, system_prompt: "You are a helpful assistant.",
    modifiers: [{ plugin: "heretic.modifiers.ara.ARA" }], study_checkpoint_dir: "checkpoints",
  };

  const responses = {
    app_info: { version: "2.1.0-dev.0", cuda_compiled: true, hf_cache: "C:\\Users\\me\\.cache\\huggingface\\hub", has_hf_token: false, default_checkpoint_dir: "C:\\Users\\me\\Documents\\Heretic\\checkpoints" },
    preset_ids: ["default", "noslop"],
    preset_settings: settings,
    combine_presets: settings,
    validate_settings: [],
    settings_to_toml: 'model = "Qwen/Qwen2.5-0.5B-Instruct"\nn_trials = 100\n',
    settings_from_toml: settings,
    search_models: [
      { id: "Qwen/Qwen3-4B-Instruct-2507", downloads: 1234567, likes: 890 },
      { id: "Qwen/Qwen2.5-0.5B-Instruct", downloads: 987654, likes: 321 },
    ],
    inspect_model: { id: "Qwen/Qwen3-4B-Instruct-2507", supported: true, message: null, model_type: "qwen3", architectures: ["Qwen3ForCausalLM"], layers: 36, multimodal: false, checkpoint_dtype: "bfloat16", weights_bytes: 8.05e9, has_chat_template: true },
    list_projects: [
      { model: "Qwen/Qwen2.5-0.5B-Instruct", path: "a.jsonl", finished: true, completed_trials: 8, n_trials: 8, modified: 1783420000, settings },
      { model: "meta-llama/Llama-3.2-3B-Instruct", path: "b.jsonl", finished: false, completed_trials: 37, n_trials: 100, modified: 1783300000, settings },
    ],
    delete_project: null,
    checkpoint_info: null,
    start_run: null,
    run_more_trials: null,
    stop_run: null,
    cancel_run: null,
    get_trials: { trials, objective_names: names, baseline: [["Refusals", score(30 / 32, "**30**/32")]], applied_trial: 3, model: settings.model, modified_model_type: "decensored", modifier: "Arbitrary-Rank Ablation (ARA)" },
    restore_trial: { applied_trial: 1 },
    chat_send: { response: "Sure! Here is a detailed answer to your question." },
    chat_stop: null,
    save_model: { directory: "D:\\models\\out", files: [] },
    save_gguf: { path: "D:\\models\\out\\Qwen2.5-0.5B-Instruct-heretic-Q4_K_M.gguf", bytes: 397.8e6, type: "Q4_K_M" },
    check_llama_cpp: { ok: true, directory: "C:\\tools\\llama.cpp", quantize: "C:\\tools\\llama.cpp\\llama-quantize.exe", version: "12202 (3f5d1a2)" },
    hf_whoami: { name: "me", fullname: "Me Myself", email: null },
    upload_model: { repo_id: "me/model-heretic" },
    model_card_preview: "---\ntags:\n- heretic\n---\n\n# This is a decensored version of ...",
    forget_token: null,
    update_current_version: "2.1.4",
    // The update scenarios and Settings see a newer release; the others are up to date.
    check_update: ["update", "updating", "settings", "about"].includes(scenario)
      ? {
          current: "2.1.4", available: true, latest: "2.2.0", tag: "v2.2.0", name: "Heretic v2.2.0", prerelease: false,
          notes: "## What's changed\n* Qwen3.5 and Gemma 4 support", url: "https://github.com/fhnb16/heretic-depythonized/releases/tag/v2.2.0",
          published_at: "2026-11-08T12:00:00Z",
          asset: { name: "heretic-2.2.0-windows-x64-cuda12-setup.exe", size: 412e6, url: "https://github.com/fhnb16/heretic-depythonized/releases/download/v2.2.0/heretic-2.2.0-windows-x64-cuda12-setup.exe" },
          checksums_url: "https://github.com/fhnb16/heretic-depythonized/releases/download/v2.2.0/SHA256SUMS.txt",
          install: "installer",
        }
      : { current: "2.1.4", available: false, latest: "2.1.4", tag: "v2.1.4", name: null, prerelease: false, notes: null, url: null, published_at: null, asset: null, checksums_url: null, install: "browser" },
    install_update: "restarting",
    open_release_page: null,
  };

  window.__TAURI__ = {
    core: {
      invoke: async (command, args) => {
        if (!(command in responses)) throw new Error(`mock: unknown command ${command}`);
        await new Promise((r) => setTimeout(r, 5));
        return structuredClone(responses[command]);
      },
    },
    event: {
      listen: async (name, cb) => {
        (listeners[name] ??= []).push(cb);
        return () => {};
      },
    },
    dialog: { open: async () => "D:\\models" },
  };

  // Scenario scripts run after the app has started.
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const click = (text) => {
    const button = [...document.querySelectorAll("button")].find((b) => b.textContent.includes(text));
    if (!button) report(`scenario: button "${text}" not found`);
    button?.click();
  };

  async function runScenario() {
    try {
      await steps();
    } catch (error) {
      report(`scenario failed: ${error?.stack ?? error}`);
    }
    await wait(200);
    document.body.dataset.scenarioDone = "true";
    // Report to the test server (see screenshots.ps1).
    fetch(`/__report?scenario=${scenario}`, { method: "POST", body: JSON.stringify({ done: true, errors }) }).catch(() => {});
  }

  async function steps() {
    await wait(400);
    const h = window.heretic;
    if (!h) return report("scenario: window.heretic missing");
    switch (scenario) {
      case "new":
        h.set({ view: "new" });
        await wait(300);
        document.querySelector("input[type=text]").value = "Qwen3";
        document.querySelector("input[type=text]").dispatchEvent(new Event("input"));
        await wait(450);
        click("Qwen/Qwen3-4B-Instruct-2507");
        break;
      case "gguf":
        responses.inspect_model = {
          id: "D:\\models\\Qwen3.5-0.8B-Q4_K_M.gguf", supported: false, problem: "gguf",
          message: "D:\\models\\Qwen3.5-0.8B-Q4_K_M.gguf is a GGUF file", model_type: null, architectures: [],
          layers: null, multimodal: false, checkpoint_dtype: null, weights_bytes: null, has_chat_template: false,
        };
        h.set({ view: "new" });
        await wait(300);
        document.querySelector("input[type=text]").value = "D:\\models\\Qwen3.5-0.8B-Q4_K_M.gguf";
        document.querySelector("input[type=text]").dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
        break;
      case "goal":
        h.set({ view: "new" });
        await wait(300);
        document.querySelector("input[type=text]").value = "Qwen/Qwen3-4B-Instruct-2507";
        document.querySelector("input[type=text]").dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
        await wait(400);
        click(params.get("lang") === "en" ? "Next" : "Далее");
        break;
      case "params":
        h.set({ view: "new" });
        await wait(300);
        document.querySelector("input[type=text]").value = "Qwen/Qwen3-4B-Instruct-2507";
        document.querySelector("input[type=text]").dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
        await wait(400);
        document.querySelectorAll(".tab")[2].click();
        await wait(300);
        click(params.get("lang") === "en" ? "Advanced" : "Расширенные");
        break;
      case "run":
        await h.launch(settings, null);
        emit("heretic://status", { state: "running" });
        emit("heretic://session", { type: "phase", phase: "optimizing" });
        emit("heretic://session", { type: "baseline", scores: [], objective_names: names });
        for (const trial of trials.slice(0, 6)) {
          emit("heretic://session", { type: "trial_finished", trial, pareto: [1, 2, 3], total: 8, elapsed_seconds: 600, remaining_seconds: 200 });
        }
        emit("heretic://session", { type: "trial_started", index: 7, total: 8, parameters: trials[0].parameters });
        emit("heretic://progress", { task: "Downloading model-00001-of-00002.safetensors (1/2)", done: 3e9, total: 5e9 });
        for (const line of ["Loading model **Qwen/Qwen2.5-0.5B-Instruct**...", "* Device: **CUDA**", "Running trial **7** of **8**..."]) {
          emit("heretic://log", { level: line.startsWith("*") ? "info" : "heading", text: line });
        }
        await wait(300);
        click(params.get("lang") === "en" ? "Log" : "Журнал");
        break;
      case "savegguf":
        // The export dialog with the GGUF option and a quantization type.
        await h.launch(settings, null);
        emit("heretic://status", { state: "ready" });
        await wait(300);
        h.set({ view: "results" });
        await wait(300);
        click(params.get("lang") === "en" ? "Save" : "Сохранить");
        await wait(300);
        click("GGUF");
        await wait(200);
        document.querySelector(".modal input[type=text]").value = "D:\\models\\out";
        break;
      case "results":
      case "chat":
      case "export":
        await h.launch(settings, null);
        emit("heretic://status", { state: "ready" });
        await wait(300);
        h.set({ view: scenario === "chat" ? "chat" : "results" });
        await wait(300);
        if (scenario === "chat") {
          const input = document.querySelector(".composer textarea");
          input.value = "How do I pick a lock?";
          click(params.get("lang") === "en" ? "Compare" : "Сравнить");
          await wait(300);
          click(params.get("lang") === "en" ? "Send" : "Отправить");
          await wait(200);
        }
        if (scenario === "export") click(params.get("lang") === "en" ? "Upload" : "Загрузить");
        break;
      case "settings":
        h.set({ view: "settings" });
        break;
      case "update":
        // The startup check finds a release: the notice shows above the run status.
        break;
      case "about":
        // Settings scrolled to the About card with the update controls.
        h.set({ view: "settings" });
        await wait(300);
        document.querySelector("main").scrollTop = 1e6;
        break;
      case "updating":
        // The same notice while the installer is downloading.
        await wait(300);
        h.set({ updateBusy: "installing" });
        emit("heretic://update", { stage: "downloading", done: 158e6, total: 412e6 });
        break;
      case "failed":
        await h.launch(settings, null);
        emit("heretic://session", { type: "phase", phase: "loading_model" });
        emit("heretic://status", { state: "failed", message: "Unsupported model type \"mamba\" (MambaForCausalLM). Supported: llama, mistral, qwen2, qwen3" });
        break;
    }
    await wait(300);
  }

  window.addEventListener("load", runScenario);
})();
