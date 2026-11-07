// SPDX-License-Identifier: AGPL-3.0-or-later
// Minimal DOM helpers.

function appendChildren(element, children) {
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false || child === true) continue;
    element.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

/**
 * Creates an element. Props: `class`, `style`, `on<event>` handlers, `dataset`,
 * DOM properties (`value`, `checked`, `disabled`, ...) and attributes.
 */
export function h(tag, props, ...children) {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(props ?? {})) {
    if (value == null || value === false) continue;
    if (key === "class") element.className = value;
    else if (key === "style") element.style.cssText = value;
    else if (key === "dataset") Object.assign(element.dataset, value);
    else if (key.startsWith("on") && typeof value === "function") element.addEventListener(key.slice(2), value);
    else if (key in element && typeof value !== "string") element[key] = value;
    else if (key === "value") element.value = value;
    else element.setAttribute(key, value === true ? "" : value);
  }
  appendChildren(element, children);
  return element;
}

/** Replaces the children of an element. */
export function replace(element, ...children) {
  element.replaceChildren();
  appendChildren(element, children);
  return element;
}

/** Renders `**bold**` spans as <b> elements. */
export function rich(text) {
  const fragment = document.createDocumentFragment();
  String(text ?? "")
    .split("**")
    .forEach((part, index) => {
      if (!part) return;
      fragment.append(index % 2 === 1 ? h("b", null, part) : document.createTextNode(part));
    });
  return fragment;
}

export function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

/** A progress bar; `fraction` null means indeterminate. */
export function progressBar(fraction) {
  const bar = h("div", { class: fraction == null ? "progress indeterminate" : "progress" }, h("div"));
  bar.firstChild.style.width = `${fraction == null ? 30 : Math.max(0, Math.min(1, fraction)) * 100}%`;
  return bar;
}
