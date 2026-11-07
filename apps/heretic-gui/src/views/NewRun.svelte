<script lang="ts">
  // New run wizard: model → goal → parameters. "Start" is available as soon as a
  // supported model is selected; everything else has sensible defaults.
  import { open } from "@tauri-apps/plugin-dialog";
  import { api, estimateVram, formatBytes, type CheckpointInfo, type ModelInspection, type ModelSearchResult, type Settings } from "../lib/api";
  import { launch } from "../lib/actions";
  import { t } from "../lib/store.svelte";
  import type { Key } from "../lib/i18n";
  import Modal from "../lib/components/Modal.svelte";

  let step = $state(0);

  // Model
  let query = $state("");
  let results = $state<ModelSearchResult[]>([]);
  let searching = $state(false);
  let modelId = $state("");
  let inspection = $state<ModelInspection | null>(null);
  let inspecting = $state(false);
  let inspectError = $state<string | null>(null);

  // Goal
  const presets: { id: string; icon: string; title: Key; description: Key }[] = [
    { id: "default", icon: "🔓", title: "presetDefault", description: "presetDefaultDesc" },
    { id: "noslop", icon: "✂️", title: "presetNoslop", description: "presetNoslopDesc" },
    { id: "nosyco", icon: "🙅", title: "presetNosyco", description: "presetNosycoDesc" },
    { id: "nohumor", icon: "🧐", title: "presetNohumor", description: "presetNohumorDesc" },
    { id: "terse", icon: "⚡", title: "presetTerse", description: "presetTerseDesc" },
    { id: "emotions", icon: "🎭", title: "presetEmotions", description: "presetEmotionsDesc" },
  ];
  let presetId = $state<string | null>("default");
  let settings = $state<Settings | null>(null);

  // Parameters
  let showAdvanced = $state(false);
  let tomlText = $state("");
  let errors = $state<string[]>([]);
  let checkpoint = $state<CheckpointInfo | null>(null);
  let starting = $state(false);

  $effect(() => {
    if (settings === null && presetId) loadPreset(presetId);
  });

  async function loadPreset(id: string) {
    presetId = id;
    const loaded = await api.presetSettings(id);
    loaded.model = modelId;
    settings = loaded;
  }

  let searchTimer: ReturnType<typeof setTimeout> | undefined;
  function onQuery() {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(async () => {
      if (query.trim().length < 2) {
        results = [];
        return;
      }
      searching = true;
      try {
        results = await api.searchModels(query);
      } catch {
        results = [];
      } finally {
        searching = false;
      }
    }, 300);
  }

  async function selectModel(id: string) {
    modelId = id;
    query = id;
    results = [];
    inspection = null;
    inspectError = null;
    inspecting = true;
    if (settings) settings.model = id;
    try {
      inspection = await api.inspectModel(id);
    } catch (error) {
      inspectError = String(error);
    } finally {
      inspecting = false;
    }
  }

  async function chooseFolder() {
    const folder = await open({ directory: true });
    if (typeof folder === "string") await selectModel(folder);
  }

  function onQueryKey(event: KeyboardEvent) {
    if (event.key === "Enter" && query.trim()) selectModel(query.trim());
  }

  async function importConfig(event: Event) {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (!file) return;
    try {
      const imported = await api.settingsFromToml(await file.text());
      imported.model = modelId || imported.model;
      settings = imported;
      presetId = null;
      errors = [];
    } catch (error) {
      errors = [String(error)];
    }
  }

  // Convenience accessors for common settings.
  const method = $derived(settings?.modifiers?.[0]?.plugin?.split(".").pop() ?? "ARA");
  const device = $derived(typeof settings?.device_map === "string" ? settings.device_map : "auto");
  const dtype = $derived(settings?.dtypes?.length === 1 ? settings.dtypes[0] : "auto");

  function setMethod(name: string) {
    if (settings) settings.modifiers = [{ plugin: name }];
  }

  function setDtype(value: string) {
    if (!settings) return;
    settings.dtypes = value === "auto" ? ["auto", "float16", "bfloat16", "float32"] : [value];
  }

  const vram = $derived(estimateVram(inspection?.weights_bytes ?? null, inspection?.checkpoint_dtype ?? null, dtype));

  async function openAdvanced() {
    showAdvanced = !showAdvanced;
    if (showAdvanced && settings) tomlText = await api.settingsToToml(settings);
  }

  async function applyToml() {
    try {
      const parsed = await api.settingsFromToml(tomlText);
      parsed.model = modelId || parsed.model;
      settings = parsed;
      errors = [];
    } catch (error) {
      errors = String(error).split("\n");
    }
  }

  const canStart = $derived(!!modelId && !!settings && (inspection?.supported ?? false) && !starting);

  async function start(action: "continue" | "restart" | null = null) {
    if (!settings) return;
    settings.model = modelId;
    errors = await api.validateSettings(settings);
    if (errors.length > 0) {
      step = 2;
      return;
    }
    if (action === null) {
      const info = await api.checkpointInfo({ ...settings });
      if (info) {
        checkpoint = info;
        return;
      }
    }
    checkpoint = null;
    starting = true;
    try {
      await launch($state.snapshot(settings), action);
    } finally {
      starting = false;
    }
  }

  const steps: Key[] = ["stepModel", "stepGoal", "stepParams"];
</script>

<div class="wizard">
  <div class="tabs row">
    {#each steps as label, i}
      <button class="tab" class:active={step === i} disabled={i > 0 && !modelId} onclick={() => (step = i)}>
        <span class="num">{i + 1}</span>{t(label)}
      </button>
    {/each}
    <span class="grow"></span>
    <button class="primary big" disabled={!canStart} onclick={() => start()}>▶ {t("start")}</button>
  </div>

  {#if step === 0}
    <section class="card stack">
      <div class="row">
        <input
          type="text"
          placeholder={t("modelSearchPlaceholder")}
          bind:value={query}
          oninput={onQuery}
          onkeydown={onQueryKey}
        />
        <button onclick={chooseFolder}>📁 {t("modelLocal")}</button>
      </div>

      {#if searching}
        <div class="progress indeterminate"><div></div></div>
      {/if}

      {#if results.length > 0}
        <ul class="results">
          {#each results as result (result.id)}
            <li>
              <button class="result" onclick={() => selectModel(result.id)}>
                <span class="grow">{result.id}</span>
                <span class="muted small">⬇ {result.downloads.toLocaleString()} · ♥ {result.likes}</span>
              </button>
            </li>
          {/each}
        </ul>
      {/if}

      {#if inspecting}
        <p class="muted">{t("modelChecking")}</p>
      {/if}
      {#if inspectError}
        <div class="error-box">{inspectError}</div>
      {/if}

      {#if inspection}
        <div class="model card">
          <div class="row">
            <h3 class="grow">{inspection.id}</h3>
            {#if inspection.supported}
              <span class="badge good">✓ {t("modelSupported")}</span>
            {:else}
              <span class="badge bad">✕ {t("modelUnsupported")}</span>
            {/if}
          </div>
          <div class="facts">
            <div><span class="muted">Architecture</span><b>{inspection.architectures.join(", ") || inspection.model_type || "?"}</b></div>
            <div><span class="muted">{t("modelLayers")}</span><b>{inspection.layers ?? "?"}</b></div>
            <div><span class="muted">{t("modelWeights")}</span><b>{formatBytes(inspection.weights_bytes)}</b></div>
            <div><span class="muted">{t("modelVram")}</span><b>{formatBytes(vram)}</b></div>
          </div>
          {#if inspection.multimodal}
            <p class="muted small">{t("modelMultimodal")}</p>
          {/if}
          {#if inspection.message}
            <div class="error-box">{inspection.message}</div>
          {/if}
        </div>
        <div class="row end">
          <button class="primary" disabled={!inspection.supported} onclick={() => (step = 1)}>{t("next")} →</button>
        </div>
      {/if}
    </section>
  {:else if step === 1}
    <section class="stack">
      <h2>{t("goalTitle")}</h2>
      <div class="presets">
        {#each presets as preset (preset.id)}
          <button class="preset card" class:active={presetId === preset.id} onclick={() => loadPreset(preset.id)}>
            <span class="icon">{preset.icon}</span>
            <b>{t(preset.title)}</b>
            <span class="muted small">{t(preset.description)}</span>
          </button>
        {/each}
        <label class="preset card" class:active={presetId === null}>
          <span class="icon">⚙️</span>
          <b>{t("goalCustom")}</b>
          <span class="muted small">{t("goalCustomHint")}</span>
          <input type="file" accept=".toml" hidden onchange={importConfig} />
        </label>
      </div>
      <div class="row end">
        <button onclick={() => (step = 0)}>← {t("back")}</button>
        <button class="primary" onclick={() => (step = 2)}>{t("next")} →</button>
      </div>
    </section>
  {:else if settings}
    <section class="card stack params">
      <label class="field">
        <span>{t("paramsTrials")}: <b>{settings.n_trials}</b></span>
        <input type="range" min="10" max="400" step="10" bind:value={settings.n_trials} />
        <span class="hint">{t("paramsTrialsHint")}</span>
      </label>

      <div class="grid2">
        <div class="field-group">
          <span class="label">{t("paramsMethod")}</span>
          <div class="segmented">
            <button class:active={method === "ARA"} onclick={() => setMethod("ARA")}>ARA</button>
            <button class:active={method === "Abliteration"} onclick={() => setMethod("Abliteration")}>Abliteration</button>
          </div>
        </div>
        <label class="field">
          {t("paramsDevice")}
          <select value={device} onchange={(e) => settings && (settings.device_map = (e.target as HTMLSelectElement).value)}>
            <option value="auto">Auto</option>
            <option value="cuda">CUDA (GPU)</option>
            <option value="cpu">CPU</option>
          </select>
        </label>
        <label class="field">
          {t("paramsDtype")}
          <select value={dtype} onchange={(e) => setDtype((e.target as HTMLSelectElement).value)}>
            <option value="auto">Auto</option>
            <option value="bfloat16">bfloat16</option>
            <option value="float16">float16</option>
            <option value="float32">float32</option>
          </select>
        </label>
        <label class="field">
          {t("paramsResponseLength")}
          <input type="number" min="1" max="2048" bind:value={settings.max_response_length} />
        </label>
      </div>

      <label class="field">
        {t("paramsSystemPrompt")}
        <input type="text" bind:value={settings.system_prompt} />
      </label>

      <button class="ghost advanced" onclick={openAdvanced}>{showAdvanced ? "▾" : "▸"} {t("paramsAdvanced")}</button>
      {#if showAdvanced}
        <label class="field">
          {t("paramsToml")}
          <textarea class="mono" rows="16" spellcheck="false" bind:value={tomlText}></textarea>
        </label>
        <div class="row end"><button onclick={applyToml}>{t("paramsApplyToml")}</button></div>
      {/if}

      {#if errors.length > 0}
        <div class="error-box">{t("paramsInvalid")}<br />{errors.join("\n")}</div>
      {/if}

      <div class="row end">
        <button onclick={() => (step = 1)}>← {t("back")}</button>
        <button class="primary big" disabled={!canStart} onclick={() => start()}>▶ {t("start")}</button>
      </div>
    </section>
  {/if}
</div>

{#if checkpoint}
  <Modal title={t("checkpointTitle")} onclose={() => (checkpoint = null)}>
    <p>
      {checkpoint.finished ? t("checkpointFinished") : t("checkpointInterrupted", { n: checkpoint.completed_trials })}
    </p>
    {#snippet footer()}
      <button onclick={() => (checkpoint = null)}>{t("cancel")}</button>
      <button class="danger" onclick={() => start("restart")}>{t("checkpointRestart")}</button>
      <button class="primary" onclick={() => start("continue")}>{t("checkpointContinue")}</button>
    {/snippet}
  </Modal>
{/if}

<style>
  .wizard {
    display: flex;
    flex-direction: column;
    gap: 18px;
  }

  .tab {
    border: none;
    background: transparent;
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-2);
  }

  .tab.active {
    color: var(--text);
    font-weight: 650;
  }

  .num {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: var(--surface-2);
    font-size: 12px;
  }

  .tab.active .num {
    background: var(--accent);
    color: var(--on-accent);
  }

  .results {
    list-style: none;
    margin: 0;
    padding: 0;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    max-height: 300px;
    overflow: auto;
  }

  .result {
    width: 100%;
    display: flex;
    gap: 10px;
    border: none;
    border-radius: 0;
    text-align: left;
    background: transparent;
  }

  .model {
    box-shadow: none;
    background: var(--surface-2);
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .facts {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
  }

  .facts div {
    display: flex;
    flex-direction: column;
  }

  .facts span {
    font-size: 12px;
  }

  .presets {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 12px;
  }

  .preset {
    display: flex;
    flex-direction: column;
    gap: 6px;
    text-align: left;
    cursor: pointer;
    border: 2px solid var(--border);
  }

  .preset.active {
    border-color: var(--accent);
    background: var(--accent-soft);
  }

  .icon {
    font-size: 24px;
  }

  .grid2 {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 14px;
  }

  .field-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .label {
    font-weight: 550;
  }

  .advanced {
    align-self: flex-start;
    padding-left: 0;
  }

  .end {
    justify-content: flex-end;
  }

  .small {
    font-size: 13px;
  }
</style>
