<script lang="ts">
  import { onMount } from "svelte";
  import { api, type AppInfo } from "./lib/api";
  import { app, applyTheme, initEvents, prefs, setPref, t, type View } from "./lib/store.svelte";
  import type { Key } from "./lib/i18n";
  import Home from "./views/Home.svelte";
  import NewRun from "./views/NewRun.svelte";
  import Run from "./views/Run.svelte";
  import Results from "./views/Results.svelte";
  import Chat from "./views/Chat.svelte";
  import SettingsView from "./views/SettingsView.svelte";

  let info = $state<AppInfo | null>(null);

  onMount(async () => {
    applyTheme();
    await initEvents();
    info = await api.appInfo();
    if (!prefs.checkpointDir) setPref("checkpointDir", info.default_checkpoint_dir);
  });

  const nav: { view: View; icon: string; label: Key; enabled: () => boolean }[] = [
    { view: "home", icon: "⌂", label: "navHome", enabled: () => true },
    { view: "new", icon: "＋", label: "navNewRun", enabled: () => app.status !== "running" && app.status !== "preparing" },
    { view: "run", icon: "◔", label: "navRun", enabled: () => app.status !== "idle" },
    { view: "results", icon: "★", label: "navResults", enabled: () => app.modelLoaded && app.trials.length > 0 },
    { view: "chat", icon: "💬", label: "navChat", enabled: () => app.modelLoaded && app.appliedTrial !== null },
    { view: "settings", icon: "⚙", label: "navSettings", enabled: () => true },
  ];

  const statusClass = $derived(
    app.status === "running" || app.status === "preparing" ? "accent" : app.status === "failed" ? "bad" : app.status === "ready" ? "good" : "",
  );
</script>

<div class="layout">
  <aside>
    <div class="brand">
      <span class="logo">H</span>
      <div class="stack tight">
        <b>Heretic</b>
        <span class="muted tagline">{t("appTagline")}</span>
      </div>
    </div>

    <nav>
      {#each nav as item (item.view)}
        <button class="nav" class:active={app.view === item.view} disabled={!item.enabled()} onclick={() => (app.view = item.view)}>
          <span class="icon">{item.icon}</span>{t(item.label)}
        </button>
      {/each}
    </nav>

    <span class="grow"></span>

    {#if app.status !== "idle"}
      <button class="status card" onclick={() => (app.view = "run")}>
        <span class="badge {statusClass}">
          {app.status === "preparing" ? t("runPreparing") : app.status === "running" ? t("runRunning") : app.status === "failed" ? t("runFailed") : t("runReady")}
        </span>
        {#if app.status === "running"}
          <span class="muted small">{app.trials.length} / {app.total}</span>
          <div class="progress"><div style="width: {app.total ? (app.trials.length / app.total) * 100 : 0}%"></div></div>
        {/if}
      </button>
    {/if}
  </aside>

  <main>
    {#if app.view === "home"}
      <Home />
    {:else if app.view === "new"}
      <NewRun />
    {:else if app.view === "run"}
      <Run />
    {:else if app.view === "results"}
      <Results />
    {:else if app.view === "chat"}
      <Chat />
    {:else}
      <SettingsView {info} />
    {/if}
  </main>
</div>

<style>
  .layout {
    display: grid;
    grid-template-columns: 230px 1fr;
    height: 100%;
  }

  aside {
    display: flex;
    flex-direction: column;
    gap: 18px;
    padding: 18px 12px;
    background: var(--surface);
    border-right: 1px solid var(--border);
  }

  .brand {
    display: flex;
    gap: 10px;
    align-items: center;
    padding: 0 6px;
  }

  .logo {
    width: 36px;
    height: 36px;
    border-radius: 9px;
    background: #111827;
    color: #22d3ee;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 20px;
    flex-shrink: 0;
  }

  .tagline {
    font-size: 11px;
    line-height: 1.3;
  }

  .tight {
    gap: 2px;
  }

  nav {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .nav {
    display: flex;
    align-items: center;
    gap: 10px;
    border: none;
    background: transparent;
    text-align: left;
    padding: 9px 10px;
    color: var(--text-2);
  }

  .nav.active {
    background: var(--accent-soft);
    color: var(--text);
    font-weight: 600;
  }

  .icon {
    width: 20px;
    text-align: center;
  }

  .status {
    display: flex;
    flex-direction: column;
    gap: 8px;
    align-items: stretch;
    text-align: left;
    box-shadow: none;
    padding: 12px;
  }

  main {
    overflow: auto;
    padding: 24px 28px;
  }

  .small {
    font-size: 12px;
  }
</style>
