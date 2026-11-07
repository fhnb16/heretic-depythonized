// SPDX-License-Identifier: AGPL-3.0-or-later
// Trade-off chart: trials over the first two objectives (both minimized). The Pareto front is
// drawn as a staircase, the region it reaches is shaded, and the original model is marked so
// the distance every trial moved is visible.

import { plain } from "../api.js";
import { escapeHtml, h } from "../dom.js";
import { t } from "../store.js";

const WIDTH = 560;
const PAD = { left: 48, right: 16, top: 28, bottom: 44 };

function extent(values) {
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

function format(v) {
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

function ticks([min, max], count = 4) {
  const step = (max - min) / count;
  return Array.from({ length: count + 1 }, (_, i) => min + i * step);
}

function describe(trial) {
  return `#${trial.index}  ` + trial.scores.map((s) => `${s.name} ${plain(s.score.display)}`).join("  ·  ");
}

/** Baseline values of the plotted objectives, if the trials carry them. */
function origin(trials, names) {
  const scores = trials[0]?.scores;
  if (!scores || names.length < 2) return null;
  const values = names.slice(0, 2).map((name) => scores.find((s) => s.name === name)?.baseline?.value);
  return values.every((v) => typeof v === "number" && Number.isFinite(v)) ? values : null;
}

/**
 * Renders the chart. Options: trials, names, selected (trial number), onselect(trial), height.
 */
export function scatterPlot({ trials, names, selected = null, onselect = null, height = 280 }) {
  const twoD = names.length >= 2;
  const xValue = (trial) => (twoD ? trial.values[0] : trial.index);
  const yValue = (trial) => (twoD ? trial.values[1] : trial.values[0]);
  const base = twoD ? origin(trials, names) : null;

  const xs = extent([...trials.map(xValue), ...(base ? [base[0]] : [])]);
  const ys = extent([...trials.map(yValue), ...(base ? [base[1]] : [])]);
  const left = PAD.left;
  const right = WIDTH - PAD.right;
  const top = PAD.top;
  const bottom = height - PAD.bottom;
  const sx = (v) => left + ((v - xs[0]) / (xs[1] - xs[0])) * (right - left);
  const sy = (v) => bottom - ((v - ys[0]) / (ys[1] - ys[0])) * (bottom - top);

  let svg = `<svg viewBox="0 0 ${WIDTH} ${height}" role="img" aria-label="${escapeHtml(names.join(" / "))}">`;
  for (const y of ticks(ys)) {
    svg += `<line class="gridline" x1="${left}" x2="${right}" y1="${sy(y)}" y2="${sy(y)}"/>`;
    svg += `<text class="tick" x="${left - 8}" y="${sy(y) + 4}" text-anchor="end">${format(y)}</text>`;
  }
  for (const x of ticks(xs)) {
    svg += `<text class="tick" x="${sx(x)}" y="${bottom + 18}" text-anchor="middle">${format(x)}</text>`;
  }
  svg += `<line class="axis" x1="${left}" x2="${right}" y1="${bottom}" y2="${bottom}"/>`;
  svg += `<text class="label" x="${right}" y="${height - 4}" text-anchor="end">${escapeHtml(twoD ? names[0] : "#")}</text>`;
  svg += `<text class="label" x="${left}" y="${top - 12}" text-anchor="start">${escapeHtml(twoD ? names[1] : (names[0] ?? ""))}</text>`;

  // Staircase through the front, sorted along x. Both objectives are minimized, so every step
  // goes right and then down.
  const front = trials.filter((trial) => trial.pareto_optimal).sort((a, b) => xValue(a) - xValue(b));
  if (twoD && front.length > 1) {
    let d = `M${sx(xValue(front[0]))},${sy(yValue(front[0]))}`;
    for (const trial of front.slice(1)) d += ` H${sx(xValue(trial))} V${sy(yValue(trial))}`;
    svg += `<path class="front" d="${d}"/>`;
  }

  if (base) {
    const [bx, by] = [sx(base[0]), sy(base[1])];
    svg += `<circle class="origin" cx="${bx}" cy="${by}" r="6"/>`;
    const anchor = bx > (left + right) / 2 ? "end" : "start";
    const dx = anchor === "end" ? -10 : 10;
    svg += `<text class="origin-label" x="${bx + dx}" y="${by - 8}" text-anchor="${anchor}">${escapeHtml(t("chartOriginal"))}</text>`;
  }

  // Ordinary trials first, so the front is drawn on top.
  const ordered = [...trials].sort((a, b) => Number(a.pareto_optimal) - Number(b.pareto_optimal));
  for (const trial of ordered) {
    const classes = [trial.pareto_optimal ? "pareto" : "", trial.number === selected ? "selected" : ""].join(" ");
    const r = trial.number === selected ? 7 : trial.pareto_optimal ? 5 : 4;
    svg += `<circle class="${classes}" data-number="${trial.number}" cx="${sx(xValue(trial))}" cy="${sy(yValue(trial))}" r="${r}"><title>${escapeHtml(describe(trial))}</title></circle>`;
  }
  svg += "</svg>";

  const hover = h("div", { class: "plot-hover" }, " ");
  const container = h("div", { class: "plot" });
  container.innerHTML = svg;
  container.append(hover);

  const find = (target) => trials.find((trial) => String(trial.number) === target?.dataset?.number);
  container.addEventListener("mouseover", (event) => {
    const trial = find(event.target);
    hover.textContent = trial ? describe(trial) : " ";
  });
  container.addEventListener("click", (event) => {
    const trial = find(event.target);
    if (trial && onselect) onselect(trial);
  });

  return container;
}
