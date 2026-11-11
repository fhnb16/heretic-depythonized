// SPDX-License-Identifier: AGPL-3.0-or-later
// Plain-language explanations: a collapsible "how this works" block and short hints.
// Body texts are i18n strings; blank lines separate paragraphs, lines starting with
// "- " become list items, and **bold** spans are rendered as <b>.

import { h } from "../dom.js";
import { icon } from "../icons.js";
import { t } from "../store.js";

function inline(text) {
  return text.split(/(\*\*[^*]+\*\*)/).map((part) =>
    part.startsWith("**") && part.endsWith("**") ? h("b", null, part.slice(2, -2)) : part,
  );
}

/** Renders paragraphs and bullet lists from a translated string. */
export function richText(key, params) {
  const blocks = t(key, params).split(/\n\s*\n/);
  return blocks.map((block) => {
    const lines = block.split("\n").filter((line) => line.trim());
    if (lines.length && lines.every((line) => line.trimStart().startsWith("- "))) {
      return h("ul", null, lines.map((line) => h("li", null, inline(line.trimStart().slice(2)))));
    }
    return h("p", null, inline(lines.join(" ")));
  });
}

/** A collapsible explanation: `summaryKey` is the clickable line, `bodyKey` the text. */
export function explain(summaryKey, bodyKey, { open = false } = {}) {
  return h(
    "details",
    { class: "help", open },
    h("summary", null, icon("help"), h("span", null, t(summaryKey))),
    h("div", { class: "help-body" }, richText(bodyKey)),
  );
}

/** Clickable example values (e.g. system prompts) that fill an input when chosen. */
export function examples(labelKey, items, onPick) {
  return h(
    "div",
    { class: "examples" },
    h("span", { class: "muted small" }, t(labelKey)),
    h(
      "div",
      { class: "chips" },
      items.map(([nameKey, valueKey]) =>
        h(
          "button",
          { type: "button", class: "chip", title: t(valueKey), onclick: () => onPick(t(valueKey)) },
          t(nameKey),
        ),
      ),
    ),
  );
}

/** System prompt examples shared by the run parameters and the chat. */
export const SYSTEM_PROMPT_EXAMPLES = [
  ["spEmpty", "spEmptyValue"],
  ["spAssistant", "spAssistantValue"],
  ["spWriter", "spWriterValue"],
  ["spTerse", "spTerseValue"],
  ["spCoder", "spCoderValue"],
  ["spTeacher", "spTeacherValue"],
  ["spRussian", "spRussianValue"],
  ["spRoleplay", "spRoleplayValue"],
];
