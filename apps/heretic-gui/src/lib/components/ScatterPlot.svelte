<script lang="ts">
  // Scatter plot of trials over the first two objectives, with the Pareto front highlighted.
  import type { TrialSummary } from "../api";
  import { plain } from "../api";

  interface Props {
    trials: TrialSummary[];
    names: string[];
    selected?: number | null;
    onselect?: (trial: TrialSummary) => void;
    height?: number;
  }

  let { trials, names, selected = null, onselect, height = 260 }: Props = $props();

  const width = 560;
  const pad = { left: 52, right: 16, top: 14, bottom: 40 };

  let hovered = $state<TrialSummary | null>(null);

  const twoD = $derived(names.length >= 2);

  function xValue(t: TrialSummary) {
    return twoD ? t.values[0] : t.index;
  }

  function yValue(t: TrialSummary) {
    return twoD ? t.values[1] : t.values[0];
  }

  function extent(values: number[]): [number, number] {
    if (values.length === 0) return [0, 1];
    let min = Math.min(...values);
    let max = Math.max(...values);
    if (min === max) {
      const d = Math.abs(min) * 0.1 || 1;
      min -= d;
      max += d;
    }
    const margin = (max - min) * 0.06;
    return [min - margin, max + margin];
  }

  const xs = $derived(extent(trials.map(xValue)));
  const ys = $derived(extent(trials.map(yValue)));

  const sx = (v: number) => pad.left + ((v - xs[0]) / (xs[1] - xs[0])) * (width - pad.left - pad.right);
  const sy = (v: number) => height - pad.bottom - ((v - ys[0]) / (ys[1] - ys[0])) * (height - pad.top - pad.bottom);

  function ticks([min, max]: [number, number], count = 5): number[] {
    const step = (max - min) / count;
    return Array.from({ length: count + 1 }, (_, i) => min + i * step);
  }

  function format(v: number) {
    const a = Math.abs(v);
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    return v.toFixed(2);
  }

  const front = $derived(
    trials
      .filter((t) => t.pareto_optimal)
      .slice()
      .sort((a, b) => xValue(a) - xValue(b)),
  );

  const frontPath = $derived(
    twoD && front.length > 1 ? front.map((t, i) => `${i === 0 ? "M" : "L"}${sx(xValue(t))},${sy(yValue(t))}`).join(" ") : "",
  );

  function tooltip(t: TrialSummary) {
    return `#${t.index}  ` + t.scores.map((s) => `${s.name}: ${plain(s.score.display)}`).join(" · ");
  }
</script>

<div class="plot">
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="Trials">
    {#each ticks(ys) as y}
      <line class="grid" x1={pad.left} x2={width - pad.right} y1={sy(y)} y2={sy(y)} />
      <text class="tick" x={pad.left - 8} y={sy(y) + 4} text-anchor="end">{format(y)}</text>
    {/each}
    {#each ticks(xs) as x}
      <text class="tick" x={sx(x)} y={height - pad.bottom + 16} text-anchor="middle">{format(x)}</text>
    {/each}

    <text class="label" x={(pad.left + width - pad.right) / 2} y={height - 6} text-anchor="middle">
      {twoD ? names[0] : "#"}
    </text>
    <text class="label" transform="translate(14 {(pad.top + height - pad.bottom) / 2}) rotate(-90)" text-anchor="middle">
      {twoD ? names[1] : (names[0] ?? "")}
    </text>

    {#if frontPath}
      <path class="front" d={frontPath} />
    {/if}

    {#each trials as t (t.number)}
      <circle
        class:pareto={t.pareto_optimal}
        class:selected={t.number === selected}
        cx={sx(xValue(t))}
        cy={sy(yValue(t))}
        r={t.number === selected ? 7 : t.pareto_optimal ? 5.5 : 4}
        role="button"
        tabindex="-1"
        onmouseenter={() => (hovered = t)}
        onmouseleave={() => (hovered = null)}
        onclick={() => onselect?.(t)}
        onkeydown={(e) => e.key === "Enter" && onselect?.(t)}
      >
        <title>{tooltip(t)}</title>
      </circle>
    {/each}
  </svg>
  <div class="hover muted">
    {#if hovered}{tooltip(hovered)}{:else}&nbsp;{/if}
  </div>
</div>

<style>
  .plot {
    width: 100%;
  }

  svg {
    width: 100%;
    height: auto;
    display: block;
  }

  .grid {
    stroke: var(--border);
    stroke-width: 1;
  }

  .tick {
    fill: var(--text-2);
    font-size: 11px;
  }

  .label {
    fill: var(--text-2);
    font-size: 12px;
    font-weight: 600;
  }

  .front {
    fill: none;
    stroke: var(--accent);
    stroke-width: 1.5;
    stroke-dasharray: 4 3;
    opacity: 0.7;
  }

  circle {
    fill: var(--text-2);
    opacity: 0.45;
    cursor: pointer;
    transition: r 0.15s;
  }

  circle.pareto {
    fill: var(--accent);
    opacity: 1;
  }

  circle.selected {
    stroke: var(--text);
    stroke-width: 2;
  }

  .hover {
    font-size: 12px;
    min-height: 18px;
    padding: 0 4px;
  }
</style>
