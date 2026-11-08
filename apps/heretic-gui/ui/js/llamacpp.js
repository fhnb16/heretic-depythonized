// SPDX-License-Identifier: AGPL-3.0-or-later
// GGUF export: the llama.cpp folder preference and the GGUF types offered.

const KEY = "heretic.llamaCppDir";

export const LLAMA_CPP_RELEASES = "https://github.com/ggml-org/llama.cpp/releases";

/** The llama.cpp folder chosen in Settings ("" if none). */
export function getLlamaCppDir() {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? '""') || "";
  } catch {
    return "";
  }
}

export function setLlamaCppDir(value) {
  try {
    localStorage.setItem(KEY, JSON.stringify(value ?? ""));
  } catch {
    // Storage may be unavailable; the folder then only lasts for this session.
  }
}

/** GGUF types (llama-quantize names) with their i18n hint keys; F16/BF16 need no llama.cpp. */
export const GGUF_TYPES = [
  { id: "Q8_0", hint: "ggufQ8" },
  { id: "Q6_K", hint: "ggufQ6" },
  { id: "Q5_K_M", hint: "ggufQ5m" },
  { id: "Q5_K_S", hint: "ggufQ5s" },
  { id: "Q4_K_M", hint: "ggufQ4m" },
  { id: "Q4_K_S", hint: "ggufQ4s" },
  { id: "Q4_0", hint: "ggufQ40" },
  { id: "Q3_K_M", hint: "ggufQ3" },
  { id: "Q2_K", hint: "ggufQ2" },
  { id: "F16", hint: "ggufF16" },
  { id: "BF16", hint: "ggufBF16" },
];

export const DEFAULT_GGUF_TYPE = "Q4_K_M";

export const needsLlamaCpp = (type) => type !== "F16" && type !== "BF16";
