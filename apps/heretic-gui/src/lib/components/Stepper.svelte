<script lang="ts">
  // Progress through the phases of a run.
  import { t, type Phase } from "../store.svelte";
  import type { Key } from "../i18n";

  interface Props {
    phase: Phase | null;
    failed?: boolean;
  }

  let { phase, failed = false }: Props = $props();

  const steps: { phase: Phase; label: Key }[] = [
    { phase: "loading_model", label: "phaseLoadingModel" },
    { phase: "batch_size", label: "phaseBatchSize" },
    { phase: "response_prefix", label: "phaseResponsePrefix" },
    { phase: "scorers", label: "phaseScorers" },
    { phase: "modifier", label: "phaseModifier" },
    { phase: "optimizing", label: "phaseOptimizing" },
    { phase: "finished", label: "phaseFinished" },
  ];

  const current = $derived.by(() => {
    if (phase === null) return -1;
    const index = steps.findIndex((s) => s.phase === phase);
    // Phases after optimization (restoring, exporting) count as finished.
    return index === -1 ? steps.length - 1 : index;
  });
</script>

<ol class="stepper">
  {#each steps as step, i}
    <li class:done={i < current || (i === current && phase === "finished")} class:active={i === current && phase !== "finished"} class:failed={failed && i === current}>
      <span class="dot">{i < current || phase === "finished" ? "✓" : i + 1}</span>
      <span class="name">{t(step.label)}</span>
    </li>
  {/each}
</ol>

<style>
  .stepper {
    display: flex;
    list-style: none;
    margin: 0;
    padding: 0;
    gap: 4px;
    flex-wrap: wrap;
  }

  li {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px 4px 4px;
    border-radius: 999px;
    color: var(--text-2);
    font-size: 13px;
  }

  .dot {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: var(--surface-2);
    font-size: 11px;
    font-weight: 700;
  }

  li.done .dot {
    background: var(--good-soft);
    color: var(--good);
  }

  li.active {
    background: var(--accent-soft);
    color: var(--text);
    font-weight: 600;
  }

  li.active .dot {
    background: var(--accent);
    color: var(--on-accent);
    animation: pulse 1.6s infinite;
  }

  li.failed .dot {
    background: var(--bad);
    color: #fff;
    animation: none;
  }

  @keyframes pulse {
    50% {
      opacity: 0.6;
    }
  }
</style>
