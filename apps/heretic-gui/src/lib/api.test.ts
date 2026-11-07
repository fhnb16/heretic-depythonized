import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));

import { estimateVram, formatBytes, formatDuration, plain, segments } from "./api";
import { translate } from "./i18n";

describe("formatting", () => {
  it("formats durations like the CLI", () => {
    expect(formatDuration(42.4)).toBe("42s");
    expect(formatDuration(200)).toBe("3m 20s");
    expect(formatDuration(3900)).toBe("1h 5m");
    expect(formatDuration(null)).toBe("—");
  });

  it("formats bytes", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(8 * 1024 ** 3)).toBe("8.0 GB");
    expect(formatBytes(null)).toBe("—");
  });

  it("handles bold markup", () => {
    expect(plain("**3**/100")).toBe("3/100");
    expect(segments("a **b** c")).toEqual([
      { text: "a ", bold: false },
      { text: "b", bold: true },
      { text: " c", bold: false },
    ]);
  });

  it("estimates memory for other dtypes", () => {
    const bf16 = estimateVram(8 * 1024 ** 3, "bfloat16", "auto")!;
    const f32 = estimateVram(8 * 1024 ** 3, "bfloat16", "float32")!;
    expect(f32).toBeGreaterThan(bf16 * 1.8);
    expect(estimateVram(null, null, "auto")).toBeNull();
  });
});

describe("i18n", () => {
  it("substitutes parameters", () => {
    expect(translate("en", "exportSaved", { path: "/x" })).toBe("Saved to /x");
    expect(translate("ru", "checkpointInterrupted", { n: 7 })).toContain("7");
  });
});
