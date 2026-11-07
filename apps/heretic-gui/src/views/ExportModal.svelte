<script lang="ts">
  import { open } from "@tauri-apps/plugin-dialog";
  import { api, type HubUser } from "../lib/api";
  import { app, t } from "../lib/store.svelte";
  import Modal from "../lib/components/Modal.svelte";

  interface Props {
    mode: "local" | "hub";
    onclose: () => void;
  }

  let { mode, onclose }: Props = $props();

  let tab = $state<"local" | "hub">(mode);
  let strategy = $state<"merge" | "adapter">("merge");
  let directory = $state("");
  let token = $state("");
  let user = $state<HubUser | null>(null);
  let repoId = $state("");
  let isPrivate = $state(false);
  let card = $state<string | null>(null);
  let working = $state(false);
  let message = $state<string | null>(null);
  let error = $state<string | null>(null);

  const modelName = $derived((app.settings?.model ?? "model").split(/[\\/]/).pop());

  $effect(() => {
    if (tab === "hub" && user === null) checkUser(true);
  });

  async function choose() {
    const folder = await open({ directory: true });
    if (typeof folder === "string") directory = folder;
  }

  async function checkUser(silent = false) {
    error = null;
    try {
      user = await api.hfWhoami(token || null);
      if (!repoId) repoId = `${user.name}/${modelName}-heretic`;
    } catch (e) {
      user = null;
      if (!silent) error = String(e);
    }
  }

  async function loadCard() {
    try {
      card = await api.modelCardPreview();
    } catch (e) {
      error = String(e);
    }
  }

  async function save() {
    working = true;
    error = null;
    message = null;
    try {
      const result = await api.saveModel(directory, strategy);
      message = t("exportSaved", { path: result.directory });
    } catch (e) {
      error = String(e);
    } finally {
      working = false;
    }
  }

  async function upload() {
    working = true;
    error = null;
    message = null;
    try {
      const result = await api.uploadModel(token || null, repoId, isPrivate, strategy);
      message = t("exportUploaded", { repo: result.repo_id });
    } catch (e) {
      error = String(e);
    } finally {
      working = false;
    }
  }
</script>

<Modal title={tab === "local" ? t("exportLocal") : t("actionUpload")} {onclose} width="600px">
  <div class="stack">
    <div class="segmented">
      <button class:active={tab === "local"} onclick={() => (tab = "local")}>{t("exportLocal")}</button>
      <button class:active={tab === "hub"} onclick={() => (tab = "hub")}>{t("exportHub")}</button>
    </div>

    <div class="choices">
      <label class="choice card" class:active={strategy === "merge"}>
        <input type="radio" bind:group={strategy} value="merge" />
        <b>{t("exportMerge")}</b>
        <span class="muted small">{t("exportMergeHint")}</span>
      </label>
      <label class="choice card" class:active={strategy === "adapter"}>
        <input type="radio" bind:group={strategy} value="adapter" />
        <b>{t("exportAdapter")}</b>
        <span class="muted small">{t("exportAdapterHint")}</span>
      </label>
    </div>

    {#if tab === "local"}
      <label class="field">
        {t("exportFolder")}
        <div class="row">
          <input type="text" bind:value={directory} />
          <button onclick={choose}>{t("exportChoose")}</button>
        </div>
      </label>
    {:else}
      {#if user}
        <span class="badge good">✓ {t("exportLoggedIn", { name: user.fullname ?? user.name })}</span>
      {:else}
        <label class="field">
          {t("exportToken")}
          <div class="row">
            <input type="password" bind:value={token} placeholder="hf_..." />
            <button onclick={() => checkUser()}>{t("exportCheck")}</button>
          </div>
          <span class="hint">{t("exportTokenHint")}</span>
        </label>
      {/if}
      <label class="field">
        {t("exportRepo")}
        <input type="text" bind:value={repoId} placeholder="user/model-heretic" />
      </label>
      <label class="row"><input type="checkbox" bind:checked={isPrivate} /> {t("exportPrivate")}</label>
      <details ontoggle={(e) => (e.currentTarget as HTMLDetailsElement).open && card === null && loadCard()}>
        <summary>{t("exportCard")}</summary>
        <pre class="card-preview mono">{card ?? "…"}</pre>
      </details>
    {/if}

    {#if working && app.progress}
      <div class="stack tight">
        <span class="muted small">{app.progress.task}</span>
        <div class="progress"><div style="width: {app.progress.total ? (app.progress.done / app.progress.total) * 100 : 30}%"></div></div>
      </div>
    {:else if working}
      <div class="progress indeterminate"><div></div></div>
    {/if}
    {#if message}<span class="badge good">✓ {message}</span>{/if}
    {#if error}<div class="error-box">{error}</div>{/if}
  </div>

  {#snippet footer()}
    <button onclick={onclose}>{t("close")}</button>
    {#if tab === "local"}
      <button class="primary" disabled={working || !directory.trim()} onclick={save}>{t("exportSave")}</button>
    {:else}
      <button class="primary" disabled={working || !repoId.trim() || (!user && !token)} onclick={upload}>{t("exportUpload")}</button>
    {/if}
  {/snippet}
</Modal>

<style>
  .choices {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }

  .choice {
    display: flex;
    flex-direction: column;
    gap: 4px;
    cursor: pointer;
    border: 2px solid var(--border);
    box-shadow: none;
    padding: 12px;
  }

  .choice.active {
    border-color: var(--accent);
    background: var(--accent-soft);
  }

  .choice input {
    display: none;
  }

  .card-preview {
    max-height: 260px;
    overflow: auto;
    background: var(--surface-2);
    padding: 10px;
    border-radius: var(--radius-sm);
    white-space: pre-wrap;
    font-size: 12px;
  }

  .tight {
    gap: 4px;
  }

  .small {
    font-size: 13px;
  }
</style>
