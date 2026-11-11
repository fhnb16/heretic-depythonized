// SPDX-License-Identifier: AGPL-3.0-or-later
// New run wizard: model → goal → parameters. "Start" is available as soon as a supported
// model is selected; everything else has sensible defaults.

import { api, estimateVram, formatBytes } from "../api.js";
import { h, progressBar, replace } from "../dom.js";
import { launch, t } from "../store.js";
import { openDialog } from "../tauri.js";
import { modal } from "../components/widgets.js";
import { SYSTEM_PROMPT_EXAMPLES, examples, explain } from "../components/help.js";
import { icon } from "../icons.js";

// [id, icon, title, short description, longer explanation]
const PRESETS = [
  ["default", "unlock", "presetDefault", "presetDefaultDesc", "presetDefaultMore"],
  ["noslop", "scissors", "presetNoslop", "presetNoslopDesc", "presetNoslopMore"],
  ["nosyco", "slash", "presetNosyco", "presetNosycoDesc", "presetNosycoMore"],
  ["nohumor", "meh", "presetNohumor", "presetNohumorDesc", "presetNohumorMore"],
  ["terse", "short", "presetTerse", "presetTerseDesc", "presetTerseMore"],
  ["emotions", "smile", "presetEmotions", "presetEmotionsDesc", "presetEmotionsMore"],
];

const STEPS = ["stepModel", "stepGoal", "stepParams"];

export function mount(root) {
  const local = {
    step: 0,
    modelId: "",
    inspection: null,
    inspecting: false,
    inspectError: null,
    // Selected goal presets (several can be combined); empty when a custom config is used.
    presetIds: ["default"],
    settings: null,
    errors: [],
    starting: false,
  };

  const tabs = h("div", { class: "row tabs" });
  const body = h("div");
  root.append(h("div", { class: "wizard stack-lg" }, tabs, body));

  const canStart = () => !!local.modelId && !!local.settings && !!local.inspection?.supported && !local.starting;

  function renderTabs() {
    replace(
      tabs,
      STEPS.map((label, i) =>
        h(
          "button",
          {
            class: `tab${local.step === i ? " active" : ""}`,
            disabled: i > 0 && !local.modelId,
            onclick: () => go(i),
          },
          h("span", { class: "num" }, i + 1),
          t(label),
        ),
      ),
      h("span", { class: "grow" }),
      h("button", { class: "primary big", disabled: !canStart(), onclick: () => start(null) }, icon("play"), t("start")),
    );
  }

  function go(step) {
    local.step = step;
    renderTabs();
    replace(body, [modelStep, goalStep, paramsStep][step]());
  }

  async function loadPresets(ids) {
    local.presetIds = ids;
    const settings = ids.length === 1 ? await api.presetSettings(ids[0]) : await api.combinePresets(ids);
    settings.model = local.modelId;
    local.settings = settings;
  }

  // ---------- Step 1: model ----------

  function modelStep() {
    const results = h("ul", { class: "results", hidden: true });
    const searching = progressBar(null);
    searching.hidden = true;
    const details = h("div", { class: "stack" });

    const input = h("input", {
      type: "text",
      placeholder: t("modelSearchPlaceholder"),
      value: local.modelId,
      autofocus: true,
    });

    let timer;
    input.addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(async () => {
        const query = input.value.trim();
        if (query.length < 2) {
          results.hidden = true;
          return;
        }
        searching.hidden = false;
        let items = [];
        try {
          items = await api.searchModels(query);
        } catch {
          items = [];
        }
        searching.hidden = true;
        results.hidden = items.length === 0;
        replace(
          results,
          items.map((item) =>
            h(
              "li",
              null,
              h(
                "button",
                {
                  class: "result",
                  onclick: () => {
                    input.value = item.id;
                    results.hidden = true;
                    select(item.id);
                  },
                },
                h("span", { class: "grow mono" }, item.id),
                h(
                  "span",
                  { class: "meta num" },
                  h("span", null, icon("download", 14), item.downloads.toLocaleString("en-US")),
                  h("span", null, icon("heart", 14), item.likes.toLocaleString("en-US")),
                ),
              ),
            ),
          ),
        );
      }, 300);
    });
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && input.value.trim()) {
        results.hidden = true;
        select(input.value.trim());
      }
    });

    async function select(id) {
      local.modelId = id;
      local.inspection = null;
      local.inspectError = null;
      local.inspecting = true;
      if (local.settings) local.settings.model = id;
      renderDetails();
      renderTabs();
      try {
        local.inspection = await api.inspectModel(id);
      } catch (error) {
        local.inspectError = String(error);
      }
      local.inspecting = false;
      renderDetails();
      renderTabs();
    }

    function renderDetails() {
      const items = [];
      if (local.inspecting) items.push(h("p", { class: "muted" }, t("modelChecking")));
      if (local.inspectError) items.push(h("div", { class: "error-box" }, local.inspectError));
      const m = local.inspection;
      if (m) {
        const dtype = local.settings?.dtypes?.length === 1 ? local.settings.dtypes[0] : "auto";
        const fact = (label, value) => h("div", null, h("span", { class: "muted" }, label), h("b", null, value));
        items.push(
          h(
            "div",
            { class: "model-card card" },
            h(
              "div",
              { class: "row" },
              h("h3", { class: "grow ellipsis" }, m.id),
              m.supported
                ? h("span", { class: "badge good" }, t("modelSupported"))
                : h("span", { class: "badge bad" }, t("modelUnsupported")),
            ),
            m.model_type || m.architectures.length
              ? h(
                  "div",
                  { class: "facts" },
                  fact(t("modelArchitecture"), m.architectures.join(", ") || m.model_type || "?"),
                  fact(t("modelLayers"), m.layers ?? "?"),
                  fact(t("modelWeights"), formatBytes(m.weights_bytes)),
                  fact(t("modelVram"), formatBytes(estimateVram(m.weights_bytes, m.checkpoint_dtype, dtype))),
                )
              : null,
            m.multimodal ? h("p", { class: "muted small" }, t("modelMultimodal")) : null,
            m.problem
              ? h(
                  "div",
                  { class: "error-box problem" },
                  h("b", null, t(`problem_${m.problem}`)),
                  h("p", null, t(`problem_${m.problem}_hint`)),
                  m.message ? h("p", { class: "mono small muted" }, `${t("modelDetails")}: ${m.message}`) : null,
                )
              : m.message
                ? h("div", { class: "error-box" }, m.message)
                : null,
          ),
          h(
            "div",
            { class: "row end" },
            h("button", { class: "primary", disabled: !m.supported, onclick: () => go(1) }, t("next"), icon("right")),
          ),
        );
      }
      replace(details, items);
    }

    renderDetails();
    return h(
      "section",
      { class: "card stack" },
      h(
        "div",
        { class: "row" },
        input,
        h(
          "button",
          {
            class: "nowrap",
            onclick: async () => {
              const folder = await openDialog({ directory: true });
              if (typeof folder === "string") {
                input.value = folder;
                select(folder);
              }
            },
          },
          icon("folder"),
          t("modelLocal"),
        ),
      ),
      searching,
      results,
      details,
      explain("helpModelTitle", "helpModelBody"),
    );
  }

  // ---------- Step 2: goal ----------

  function goalStep() {
    const cards = h("div", { class: "presets" });
    const errors = h("div");

    function render() {
      const fileInput = h("input", {
        type: "file",
        accept: ".toml",
        hidden: true,
        onchange: async (event) => {
          const file = event.target.files?.[0];
          if (!file) return;
          try {
            const imported = await api.settingsFromToml(await file.text());
            imported.model = local.modelId || imported.model;
            local.settings = imported;
            local.presetIds = [];
            replace(errors);
          } catch (error) {
            replace(errors, h("div", { class: "error-box" }, String(error)));
          }
          render();
        },
      });

      replace(
        cards,
        PRESETS.map(([id, name, title, description, more]) => {
          const selected = local.presetIds.includes(id);
          return h(
            "button",
            {
              class: `preset card${selected ? " active" : ""}`,
              "aria-pressed": String(selected),
              onclick: async () => {
                // Toggle; at least one goal stays selected.
                let ids = selected ? local.presetIds.filter((x) => x !== id) : [...local.presetIds, id];
                if (ids.length === 0) ids = [id];
                // Keep the catalog order so the combination doesn't depend on click order.
                ids = PRESETS.map(([p]) => p).filter((p) => ids.includes(p));
                try {
                  await loadPresets(ids);
                  replace(errors);
                } catch (error) {
                  replace(errors, h("div", { class: "error-box" }, String(error)));
                }
                render();
              },
            },
            h("span", { class: "preset-head" }, icon(name, 20), h("span", { class: "preset-check" }, icon("check", 14))),
            h("b", null, t(title)),
            h("span", { class: "muted small" }, t(description)),
            h("span", { class: "preset-more" }, t(more)),
          );
        }),
        h(
          "label",
          { class: `preset card${local.presetIds.length === 0 ? " active" : ""}` },
          h("span", { class: "preset-head" }, icon("file", 20), h("span", { class: "preset-check" }, icon("check", 14))),
          h("b", null, t("goalCustom")),
          h("span", { class: "muted small" }, t("goalCustomHint")),
          h("span", { class: "preset-more" }, t("goalCustomMore")),
          fileInput,
        ),
      );
      replace(
        combined,
        local.presetIds.length > 1
          ? h(
              "div",
              { class: "note" },
              t("goalCombined", {
                goals: local.presetIds
                  .map((id) => t(PRESETS.find(([p]) => p === id)[2]))
                  .join(" + "),
              }),
            )
          : null,
      );
    }

    const combined = h("div");
    render();
    return h(
      "section",
      { class: "stack" },
      h("h2", null, t("goalTitle")),
      h("p", { class: "muted" }, t("goalMultiHint")),
      explain("helpHowTitle", "helpHowBody"),
      cards,
      combined,
      explain("helpCombineTitle", "helpCombineBody"),
      errors,
      h(
        "div",
        { class: "row end" },
        h("button", { onclick: () => go(0) }, icon("left"), t("back")),
        h("button", { class: "primary", onclick: () => go(2) }, t("next"), icon("right")),
      ),
    );
  }

  // ---------- Step 3: parameters ----------

  function paramsStep() {
    const s = local.settings;
    if (!s) return h("p", { class: "muted" }, "…");

    const errorBox = h("div");
    const showErrors = () =>
      replace(
        errorBox,
        local.errors.length ? h("div", { class: "error-box" }, `${t("paramsInvalid")}\n${local.errors.join("\n")}`) : null,
      );

    const trialsLabel = h("span", { class: "num" }, s.n_trials);
    const trials = h("input", { type: "range", min: 10, max: 400, step: 10, value: String(s.n_trials) });
    trials.addEventListener("input", () => {
      s.n_trials = Number(trials.value);
      trialsLabel.textContent = trials.value;
    });

    const method = () => s.modifiers?.[0]?.plugin?.split(".").pop() ?? "ARA";
    const methodControl = h("div", { class: "segmented" });
    const renderMethod = () =>
      replace(
        methodControl,
        ["ARA", "Abliteration"].map((name) =>
          h(
            "button",
            {
              class: method() === name ? "active" : "",
              onclick: () => {
                // Goal presets configure prompt sets for one method; carry them over.
                const previous = s.modifier?.[method()];
                if (previous && name !== method()) {
                  s.modifier[name] ??= {};
                  for (const key of ["good_prompts", "bad_prompts"]) {
                    if (previous[key] !== undefined) s.modifier[name][key] = previous[key];
                  }
                }
                s.modifiers = [{ plugin: name }];
                renderMethod();
              },
            },
            name,
          ),
        ),
      );
    renderMethod();

    const select = (value, options, onchange) => {
      const element = h(
        "select",
        { onchange: (event) => onchange(event.target.value) },
        options.map(([v, label]) => h("option", { value: v, selected: v === value }, label)),
      );
      return element;
    };

    const device = typeof s.device_map === "string" ? s.device_map : "auto";
    const dtype = s.dtypes?.length === 1 ? s.dtypes[0] : "auto";

    const responseLength = h("input", { type: "number", min: 1, max: 4096, value: String(s.max_response_length) });
    responseLength.addEventListener("change", () => (s.max_response_length = Number(responseLength.value)));

    const systemPrompt = h("textarea", { rows: 3, value: s.system_prompt });
    systemPrompt.value = s.system_prompt ?? "";
    systemPrompt.addEventListener("input", () => (s.system_prompt = systemPrompt.value));
    const systemExamples = examples("spExamples", SYSTEM_PROMPT_EXAMPLES, (value) => {
      systemPrompt.value = value;
      s.system_prompt = value;
    });

    const advanced = h("div", { class: "stack", hidden: true });
    const toml = h("textarea", { class: "mono", rows: 16, spellcheck: false });
    advanced.append(
      explain("helpTomlTitle", "helpTomlBody"),
      h("label", { class: "field" }, t("paramsToml"), toml),
      h(
        "div",
        { class: "row end" },
        h(
          "button",
          {
            onclick: async () => {
              try {
                const parsed = await api.settingsFromToml(toml.value);
                parsed.model = local.modelId || parsed.model;
                local.settings = parsed;
                local.errors = [];
                go(2);
              } catch (error) {
                local.errors = String(error).split("\n");
                showErrors();
              }
            },
          },
          t("paramsApplyToml"),
        ),
      ),
    );

    const advancedToggle = h("button", { class: "ghost toggle", "aria-expanded": "false" }, icon("chevron"), t("paramsAdvanced"));
    advancedToggle.addEventListener("click", async () => {
      advanced.hidden = !advanced.hidden;
      advancedToggle.setAttribute("aria-expanded", String(!advanced.hidden));
      if (!advanced.hidden) toml.value = await api.settingsToToml(s);
    });

    showErrors();
    return h(
      "section",
      { class: "card stack" },
      h(
        "label",
        { class: "field" },
        h("span", { class: "row" }, h("span", { class: "grow" }, t("paramsTrials")), trialsLabel),
        trials,
        h("span", { class: "hint" }, t("paramsTrialsHint")),
      ),
      explain("helpTrialsTitle", "helpTrialsBody"),
      h(
        "div",
        { class: "grid2" },
        h("div", { class: "field" }, t("paramsMethod"), methodControl, h("span", { class: "hint" }, t("paramsMethodHint"))),
        h(
          "label",
          { class: "field" },
          t("paramsDevice"),
          select(
            device,
            [
              ["auto", "Auto"],
              ["cuda", "CUDA (GPU)"],
              ["cpu", "CPU"],
            ],
            (v) => (s.device_map = v),
          ),
        ),
        h(
          "label",
          { class: "field" },
          t("paramsDtype"),
          select(
            dtype,
            [
              ["auto", "Auto"],
              ["bfloat16", "bfloat16"],
              ["float16", "float16"],
              ["float32", "float32"],
            ],
            (v) => (s.dtypes = v === "auto" ? ["auto", "float16", "bfloat16", "float32"] : [v]),
          ),
        ),
        h(
          "label",
          { class: "field" },
          t("paramsResponseLength"),
          responseLength,
          h("span", { class: "hint" }, t("paramsResponseLengthHint")),
        ),
      ),
      explain("helpMethodTitle", "helpMethodBody"),
      explain("helpHardwareTitle", "helpHardwareBody"),
      h(
        "label",
        { class: "field" },
        t("paramsSystemPrompt"),
        systemPrompt,
        h("span", { class: "hint" }, t("paramsSystemPromptHint")),
      ),
      systemExamples,
      explain("helpSystemTitle", "helpSystemBody"),
      advancedToggle,
      advanced,
      errorBox,
      h(
        "div",
        { class: "row end" },
        h("button", { onclick: () => go(1) }, icon("left"), t("back")),
        h("button", { class: "primary big", disabled: !canStart(), onclick: () => start(null) }, icon("play"), t("start")),
      ),
    );
  }

  // ---------- Start ----------

  async function start(action) {
    const s = local.settings;
    if (!s) return;
    s.model = local.modelId;
    local.errors = await api.validateSettings(s);
    if (local.errors.length > 0) {
      go(2);
      return;
    }
    if (action === null) {
      const info = await api.checkpointInfo(s);
      if (info) {
        const close = modal({
          title: t("checkpointTitle"),
          body: h(
            "p",
            null,
            info.finished ? t("checkpointFinished") : t("checkpointInterrupted", { n: info.completed_trials }),
          ),
          buttons: [
            h("button", { onclick: () => close() }, t("cancel")),
            h(
              "button",
              {
                class: "danger",
                onclick: () => {
                  close();
                  start("restart");
                },
              },
              t("checkpointRestart"),
            ),
            h(
              "button",
              {
                class: "primary",
                onclick: () => {
                  close();
                  start("continue");
                },
              },
              t("checkpointContinue"),
            ),
          ],
        });
        return;
      }
    }
    local.starting = true;
    await launch(structuredClone(s), action);
  }

  loadPresets(["default"]).then(() => renderTabs());
  go(0);
  return () => {};
}
