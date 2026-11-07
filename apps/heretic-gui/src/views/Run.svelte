<script lang="ts">
  import { api, formatDuration, plain } from "../lib/api";
  import { app, t } from "../lib/store.svelte";
  import Stepper from "../lib/components/Stepper.svelte";
  import ScatterPlot from "../lib/components/ScatterPlot.svelte";
  import LogPanel from "../lib/components/LogPanel.svelte";

  let showLog = $state(false);
  let stopping = $state(false);

  const done = $derived(app.trials.length);
  const running = $derived(app.status === "running" || app.status === "preparing");
  const latest = $derived([...app.trials].sort((a, b) => b.number - a.number).slice(0, 8));

  $effect(() => {
    if (!running) stopping = false;
  });

  async function stop() {
    stopping = true;
    await api.stopRun();
  }

  const statusLabel = $derived(
    app.status === "preparing"
      ? t("runPreparing")
      : app.status === "running"
        ? t("runRunning")
        : app.status === "failed"
          ? t("runFailed")
          : t("runReady"),
  );
</script>

<div class="run">
  <header class="row">
    <div class="grow stack tight">
      <h2>{app.settings?.model ?? ""}</h2>
      <div class="row">
        <span class="badge {app.status === 'failed' ? 'bad' : app.status === 'ready' ? 'good' : 'accent'}">{statusLabel}</span>
        {#if app.status === "running" || app.status === "ready"}
          <span class="muted">{t("runTrial")} {done} {t("runOf")} {app.total}</span>
          <span class="muted">· {t("runElapsed")} {formatDuration(app.elapsed)}</span>
          {#if app.remaining != null && running}
            <span class="muted">· {t("runRemaining")} ~{formatDuration(app.remaining)}</span>
          {/if}
        {/if}
      </div>
    </div>
    {#if running}
      <button disabled={stopping} onclick={stop}>⏸ {t("runStop")}</button>
      <button class="danger" onclick={() => api.cancelRun()}>✕ {t("runCancel")}</button>
    {:else if app.status === "ready"}
      <button class="primary big" onclick={() => (app.view = "results")}>{t("runShowResults")} →</button>
    {/if}
  </header>

  <div class="card stack">
    <Stepper phase={app.phase} failed={app.status === "failed"} />
    {#if app.status === "running" && app.total > 0}
      <div class="progress"><div style="width: {(done / app.total) * 100}%"></div></div>
    {/if}
    {#if app.progress}
      <div class="stack tight">
        <span class="muted small">{app.progress.task}</span>
        <div class="progress" class:indeterminate={app.progress.total == null}>
          <div style="width: {app.progress.total ? (app.progress.done / app.progress.total) * 100 : 30}%"></div>
        </div>
      </div>
    {/if}
    {#if app.status === "failed" && app.statusMessage}
      <div class="error-box">{app.statusMessage}</div>
    {/if}
  </div>

  <div class="columns">
    <div class="card stack">
      <h3>{t("resultsAllTrials")}</h3>
      {#if app.trials.length > 0}
        <ScatterPlot trials={app.trials} names={app.objectiveNames} selected={app.appliedTrial} />
      {:else}
        <p class="muted placeholder">{t("runNoTrials")}</p>
      {/if}
    </div>

    <div class="card stack">
      {#if app.current && app.status === "running"}
        <h3>{t("runTrial")} {app.current.index} · {t("runCurrentParameters")}</h3>
        <table>
          <tbody>
            {#each app.current.parameters as [name, value]}
              <tr><td class="muted">{name}</td><td class="mono">{value}</td></tr>
            {/each}
          </tbody>
        </table>
      {:else if latest.length > 0}
        <table>
          <thead>
            <tr>
              <th>#</th>
              {#each latest[0].scores as score}<th>{score.name}</th>{/each}
            </tr>
          </thead>
          <tbody>
            {#each latest as trial (trial.number)}
              <tr class:pareto={trial.pareto_optimal}>
                <td>{trial.index}</td>
                {#each trial.scores as score}<td class="mono">{plain(score.score.display)}</td>{/each}
              </tr>
            {/each}
          </tbody>
        </table>
      {:else}
        <p class="muted placeholder">{t("runNoTrials")}</p>
      {/if}
    </div>
  </div>

  <div class="card stack">
    <button class="ghost toggle" onclick={() => (showLog = !showLog)}>{showLog ? "▾" : "▸"} {t("runLog")}</button>
    {#if showLog || app.status === "failed"}
      <LogPanel lines={app.logs} />
    {/if}
  </div>
</div>

<style>
  .run {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .tight {
    gap: 4px;
  }

  .columns {
    display: grid;
    grid-template-columns: 3fr 2fr;
    gap: 16px;
  }

  .placeholder {
    padding: 40px 0;
    text-align: center;
  }

  .toggle {
    align-self: flex-start;
    padding-left: 0;
    font-weight: 600;
  }

  tr.pareto td:first-child {
    color: var(--accent);
    font-weight: 700;
  }

  .small {
    font-size: 13px;
  }
</style>
