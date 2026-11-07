<script lang="ts">
  import { api, plain, type TrialSummary } from "../lib/api";
  import { app, paretoFront, refreshTrials, t } from "../lib/store.svelte";
  import ScatterPlot from "../lib/components/ScatterPlot.svelte";
  import ExportModal from "./ExportModal.svelte";

  let applying = $state<number | null>(null);
  let error = $state<string | null>(null);
  let exportMode = $state<"local" | "hub" | null>(null);
  let moreCount = $state(20);

  $effect(() => {
    if (app.modelLoaded && app.trials.length === 0) refreshTrials();
  });

  const front = $derived(paretoFront(app.trials));
  const selected = $derived(app.trials.find((t) => t.number === app.appliedTrial) ?? null);
  const busy = $derived(app.status === "running" || app.status === "preparing" || applying !== null);

  async function apply(trial: TrialSummary) {
    if (busy || trial.number === app.appliedTrial) return;
    applying = trial.number;
    error = null;
    try {
      const result = await api.restoreTrial(trial.number);
      app.appliedTrial = result.applied_trial;
    } catch (e) {
      error = String(e);
    } finally {
      applying = null;
    }
  }

  async function runMore() {
    error = null;
    try {
      app.total = app.trials.length + moreCount;
      app.view = "run";
      await api.runMoreTrials(moreCount);
    } catch (e) {
      error = String(e);
    }
  }

  /** Relative change compared to the baseline, for coloring. */
  function trend(score: { value: number }, baseline: { value: number }): "up" | "down" | "same" {
    if (Math.abs(score.value - baseline.value) < 1e-9) return "same";
    return score.value > baseline.value ? "up" : "down";
  }
</script>

<div class="results">
  <header class="row">
    <div class="grow stack tight">
      <h2>{t("resultsTitle")}</h2>
      <p class="muted">{t("resultsHint")}</p>
    </div>
  </header>

  {#if error}<div class="error-box">{error}</div>{/if}

  <div class="columns">
    <div class="stack">
      {#each front as trial (trial.number)}
        <button class="trial card" class:selected={trial.number === app.appliedTrial} disabled={busy && applying !== trial.number} onclick={() => apply(trial)}>
          <div class="row">
            <b class="index">#{trial.index}</b>
            <span class="grow"></span>
            {#if applying === trial.number}
              <span class="badge accent">{t("resultsApplying")}</span>
            {:else if trial.number === app.appliedTrial}
              <span class="badge good">✓ {t("resultsSelected")}</span>
            {/if}
          </div>
          <div class="scores">
            {#each trial.scores as record}
              <div class="score">
                <span class="muted small">{record.name}</span>
                <span class="value {trend(record.score, record.baseline)}">{plain(record.score.display)}</span>
                <span class="muted tiny">{t("resultsBaseline")}: {plain(record.baseline.display)}</span>
              </div>
            {/each}
          </div>
        </button>
      {/each}
    </div>

    <div class="stack">
      <div class="card stack">
        <ScatterPlot trials={app.trials} names={app.objectiveNames} selected={app.appliedTrial} onselect={apply} />
      </div>

      {#if selected}
        <div class="card stack">
          <div class="row actions">
            <button class="primary" disabled={busy} onclick={() => (app.view = "chat")}>💬 {t("actionChat")}</button>
            <button disabled={busy} onclick={() => (exportMode = "local")}>💾 {t("actionSave")}</button>
            <button disabled={busy} onclick={() => (exportMode = "hub")}>☁️ {t("actionUpload")}</button>
          </div>
          <h3>{t("resultsParameters")} · #{selected.index}</h3>
          <table>
            <tbody>
              {#each selected.parameters as [name, value]}
                <tr><td class="muted">{name}</td><td class="mono">{value}</td></tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}

      <div class="card row">
        <span class="grow">{t("resultsMore")}</span>
        <input type="number" min="1" max="1000" bind:value={moreCount} style="width: 90px" />
        <button disabled={busy || moreCount < 1} onclick={runMore}>▶</button>
      </div>
    </div>
  </div>
</div>

{#if exportMode}
  <ExportModal mode={exportMode} onclose={() => (exportMode = null)} />
{/if}

<style>
  .results {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .tight {
    gap: 4px;
  }

  .columns {
    display: grid;
    grid-template-columns: 2fr 3fr;
    gap: 16px;
    align-items: start;
  }

  .trial {
    text-align: left;
    display: flex;
    flex-direction: column;
    gap: 10px;
    border: 2px solid var(--border);
    padding: 14px;
  }

  .trial.selected {
    border-color: var(--accent);
  }

  .index {
    font-size: 15px;
  }

  .scores {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 10px;
  }

  .score {
    display: flex;
    flex-direction: column;
  }

  .value {
    font-size: 18px;
    font-weight: 650;
  }

  .actions {
    flex-wrap: wrap;
  }

  .small {
    font-size: 12px;
  }

  .tiny {
    font-size: 11px;
  }
</style>
