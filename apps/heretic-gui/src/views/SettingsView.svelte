<script lang="ts">
  import { open } from "@tauri-apps/plugin-dialog";
  import { api, type AppInfo } from "../lib/api";
  import { prefs, setPref, t } from "../lib/store.svelte";

  interface Props {
    info: AppInfo | null;
  }

  let { info }: Props = $props();
  let forgotten = $state(false);

  async function chooseFolder() {
    const folder = await open({ directory: true });
    if (typeof folder === "string") setPref("checkpointDir", folder);
  }
</script>

<div class="settings stack">
  <div class="card stack">
    <div class="row">
      <span class="grow">{t("settingsLanguage")}</span>
      <div class="segmented">
        <button class:active={prefs.lang === "ru"} onclick={() => setPref("lang", "ru")}>Русский</button>
        <button class:active={prefs.lang === "en"} onclick={() => setPref("lang", "en")}>English</button>
      </div>
    </div>
    <div class="row">
      <span class="grow">{t("settingsTheme")}</span>
      <div class="segmented">
        <button class:active={prefs.theme === "system"} onclick={() => setPref("theme", "system")}>{t("themeSystem")}</button>
        <button class:active={prefs.theme === "light"} onclick={() => setPref("theme", "light")}>{t("themeLight")}</button>
        <button class:active={prefs.theme === "dark"} onclick={() => setPref("theme", "dark")}>{t("themeDark")}</button>
      </div>
    </div>
  </div>

  <div class="card stack">
    <label class="field">
      {t("settingsCheckpoints")}
      <div class="row">
        <input type="text" value={prefs.checkpointDir} onchange={(e) => setPref("checkpointDir", (e.target as HTMLInputElement).value)} />
        <button onclick={chooseFolder}>{t("exportChoose")}</button>
      </div>
    </label>
    <div class="row">
      <span class="grow">{t("settingsCache")}</span>
      <span class="mono muted">{info?.hf_cache ?? ""}</span>
    </div>
    <div class="row">
      <span class="grow"></span>
      <button disabled={forgotten} onclick={async () => { await api.forgetToken(); forgotten = true; }}>{t("settingsToken")}</button>
    </div>
  </div>

  <div class="card stack">
    <h3>{t("settingsAbout")}</h3>
    <div class="row"><span class="grow">Heretic</span><span class="mono">v{info?.version ?? ""}</span></div>
    <div class="row"><span class="grow">{t("settingsCuda")}</span><span>{info?.cuda_compiled ? t("yes") : t("no")}</span></div>
    <p class="muted small">
      AGPL-3.0-or-later · <a href="https://heretic-project.org" target="_blank" rel="noreferrer">heretic-project.org</a>
    </p>
  </div>
</div>

<style>
  .settings {
    max-width: 720px;
  }

  .small {
    font-size: 13px;
  }

  a {
    color: var(--accent);
  }
</style>
