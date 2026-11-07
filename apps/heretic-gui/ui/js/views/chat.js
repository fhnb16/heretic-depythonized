// SPDX-License-Identifier: AGPL-3.0-or-later
// Chat with the modified model, optionally side by side with the original.

import { api } from "../api.js";
import { h, replace } from "../dom.js";
import { icon } from "../icons.js";
import { state, subscribe, t } from "../store.js";

export function mount(root) {
  if (!state.modelLoaded || state.appliedTrial === null) {
    root.append(h("div", { class: "card empty muted" }, t("chatNoModel")));
    return () => {};
  }

  const local = { mode: "modified", turns: [], generating: null, error: null };

  const modeControl = h("div", { class: "segmented" });
  const clear = h("button", { class: "ghost" }, t("chatClear"));
  const system = h("textarea", { rows: 2 });
  system.value = state.settings?.system_prompt ?? "You are a helpful assistant.";
  const messages = h("div", { class: "messages" });
  const errorBox = h("div");
  const input = h("textarea", { rows: 2, placeholder: t("chatPlaceholder") });
  const sendButton = h("button", { class: "primary" });

  clear.addEventListener("click", () => {
    local.turns = [];
    render();
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  });

  sendButton.addEventListener("click", () => (local.generating ? api.chatStop() : send()));

  root.append(
    h(
      "div",
      { class: "chat" },
      h("div", { class: "row" }, modeControl, h("span", { class: "grow" }), clear),
      h("details", { class: "system" }, h("summary", { class: "muted" }, t("chatSystem")), system),
      messages,
      errorBox,
      h("div", { class: "row composer" }, input, sendButton),
      h(
        "div",
        { class: "composer-hint" },
        h("span", null, h("kbd", null, "Enter"), ` ${t("chatHintSend")}`),
        h("span", null, h("kbd", null, "Shift"), "+", h("kbd", null, "Enter"), ` ${t("chatHintNewline")}`),
      ),
    ),
  );

  const sides = () => (local.mode === "compare" ? ["modified", "original"] : [local.mode]);

  function history(side) {
    const result = [{ role: "system", content: system.value }];
    for (const turn of local.turns) {
      result.push({ role: "user", content: turn.user });
      if (turn[side] !== undefined) result.push({ role: "assistant", content: turn[side] });
    }
    return result;
  }

  async function generate(side) {
    local.generating = side;
    state.chatStream = "";
    render();
    const turn = local.turns[local.turns.length - 1];
    try {
      const result = await api.chatSend(history(side), side === "modified");
      turn[side] = result.response;
    } catch (error) {
      // A stopped generation keeps what was produced so far.
      turn[side] = state.chatStream || "";
      if (!String(error).toLowerCase().includes("cancel")) local.error = String(error);
    }
    state.chatStream = "";
    local.generating = null;
    render();
  }

  async function send() {
    const text = input.value.trim();
    if (!text || local.generating) return;
    input.value = "";
    local.error = null;
    local.turns.push({ user: text });
    for (const side of sides()) await generate(side);
    input.focus();
  }

  function render() {
    replace(
      modeControl,
      [
        ["modified", t("chatModified")],
        ["original", t("chatOriginal")],
        ["compare", t("chatCompare")],
      ].map(([mode, label]) =>
        h(
          "button",
          {
            class: local.mode === mode ? "active" : "",
            disabled: !!local.generating,
            onclick: () => {
              local.mode = mode;
              render();
            },
          },
          label,
        ),
      ),
    );
    clear.disabled = !!local.generating;
    sendButton.className = local.generating ? "danger" : "primary";
    replace(sendButton, local.generating ? [icon("stop"), t("chatStop")] : t("chatSend"));

    const label = (side) => (side === "modified" ? t("chatModified") : t("chatOriginal"));
    replace(
      messages,
      local.turns.map((turn, index) => [
        h("div", { class: "bubble user" }, turn.user),
        h(
          "div",
          { class: `answers${local.mode === "compare" ? " two" : ""}` },
          sides().map((side) => {
            let text;
            if (turn[side] !== undefined) text = h("div", { class: "text" }, turn[side]);
            else if (local.generating === side && index === local.turns.length - 1)
              text = h("div", { class: "text" }, state.chatStream, h("span", { class: "cursor" }, "▍"));
            else text = h("div", { class: "text muted" }, "…");
            return h(
              "div",
              { class: "bubble assistant" },
              local.mode === "compare" ? h("span", { class: `side ${side}` }, label(side)) : null,
              text,
            );
          }),
        ),
      ]),
    );
    messages.scrollTop = messages.scrollHeight;
    replace(errorBox, local.error ? h("div", { class: "error-box" }, local.error) : null);
  }

  render();
  // Streamed tokens update the last answer.
  return subscribe(() => local.generating && render());
}
