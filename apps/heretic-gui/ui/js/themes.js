// SPDX-License-Identifier: AGPL-3.0-or-later
// Color themes ported from the theme catalog (see AGENTS.md, "Themes") at the token level:
// palette of both modes, font, radii and accent. Layout stays the app's own; decorations the catalog
// draws with extra CSS (glossy gradients, bevels, glows) are deliberately not ported.
//
// Palette order: [bg, surface, surface-press, separator, separator-strong, text, text-2,
// text-3, danger, ok]. Values may use var(--accent) (monochrome terminal themes).

const SANS_APPLE = '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif';
const ACC = (pct, base) => `color-mix(in srgb, var(--accent) ${pct}%, ${base})`;

export const THEMES = [
  {
    id: "aero",
    name: "Aero",
    accent: "#2a8dd4",
    font: '"Segoe UI", Tahoma, Verdana, sans-serif',
    r: 3,
    rc: 6,
    // The catalog paints a blue desktop behind white windows; the light page is softened so
    // headings that sit on the page stay readable.
    light: ["#cfe0f1", "#ffffff", "#e5f1fb", "#dfe8f2", "#8fb0d6", "#1e1e1e", "#4a5d73", "#7f8fa3", "#c42b1c", "#2c9e2c"],
    dark: ["#0d2440", "#1b2838", "#24405e", "#2a3b50", "#3f6a98", "#eaf2fb", "#a8bfd6", "#7089a3", "#e5544a", "#4cbf4c"],
  },
  {
    id: "alloy",
    name: "Alloy",
    accent: "#e7d700",
    font: "Arial, Tahoma, Verdana, sans-serif",
    r: 5,
    rc: 7,
    light: ["#dcdfe3", "#ffffff", "#d2d2d2", "#e4e5e7", "#b1b3b4", "#202022", "#68686a", "#9d9ea0", "#e0001b", "#44b700"],
    dark: ["#121214", "#242426", "#36363a", "#2e2e32", "#57585a", "#f0f0f2", "#a7a9ac", "#707174", "#ff453a", "#44b700"],
  },
  {
    id: "board",
    name: "Board",
    accent: "#7ef24a",
    font: '"JetBrains Mono", "SF Mono", ui-monospace, Menlo, Consolas, monospace',
    r: 2,
    rc: 3,
    dark: ["#0b1017", "#111821", "#18212c", "#1e2a38", "#2c3c4e", "#dfe7ef", "#8fa3b8", "#5f7386", "#ff6b6b", "#7ef24a"],
    light: ["#f4f7fa", "#ffffff", "#e9eef3", "#dde4ec", "#c3cfdb", "#16202b", "#52637a", "#8596a8", "#d64545", "#3f8f1f"],
  },
  {
    id: "bubble",
    name: "Bubble",
    accent: "#3390ec",
    font: '-apple-system, BlinkMacSystemFont, Roboto, "Segoe UI", sans-serif',
    r: 8,
    rc: 14,
    light: ["#dfe7ee", "#ffffff", "#eef2f5", "#e6ebef", "#c9d3dc", "#000000", "#707e8b", "#9aa7b3", "#e53935", "#4fae4e"],
    dark: ["#0e1621", "#182533", "#202b36", "#22303f", "#2b3a4a", "#f5f5f5", "#7f91a4", "#5f7082", "#e53935", "#4fae4e"],
  },
  {
    id: "ember",
    name: "Ember",
    accent: "#ff9000",
    font: "Arial, Helvetica, sans-serif",
    r: 4,
    rc: 4,
    dark: ["#000000", "#1b1b1b", "#2a2a2a", "#000000", "#333333", "#ffffff", "#c6c6c6", "#8a8a8a", "#ff4d4d", "#66c05e"],
    light: ["#ffffff", "#f2f2f2", "#e4e4e4", "#ffffff", "#d9d9d9", "#111111", "#555555", "#8a8a8a", "#e03b3b", "#3e9a36"],
  },
  {
    id: "flat",
    name: "Flat",
    accent: "#007aff",
    font: SANS_APPLE,
    r: 5,
    rc: 0,
    light: ["#efeff4", "#ffffff", "#d9d9de", "#c8c7cc", "#b2b2b7", "#000000", "#8e8e93", "#aeaeb2", "#ff3b30", "#4cd964"],
    dark: ["#000000", "#1c1c1e", "#2c2c2e", "#38383a", "#48484a", "#ffffff", "#8e8e93", "#636366", "#ff453a", "#30d158"],
  },
  {
    id: "fluent",
    name: "Fluent",
    accent: "#0078d4",
    font: '"Segoe UI Variable Text", "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, "Noto Sans", sans-serif',
    r: 4,
    rc: 8,
    light: ["#f3f3f3", "#ffffff", "#ededed", "#e5e5e5", "#d2d2d2", "#1b1b1b", "#5d5d5d", "#8a8a8a", "#c42b1c", "#0f7b0f"],
    dark: ["#202020", "#2b2b2b", "#383838", "#3a3a3a", "#4a4a4a", "#ffffff", "#c8c8c8", "#9a9a9a", "#ff99a4", "#6ccb5f"],
  },
  {
    id: "forum",
    name: "Forum",
    accent: "#ff4500",
    font: '"IBM Plex Sans", -apple-system, "Segoe UI", Arial, sans-serif',
    r: 4,
    rc: 8,
    rb: 999,
    light: ["#dae0e6", "#ffffff", "#f6f7f8", "#edeff1", "#cccccc", "#1c1c1c", "#576f76", "#878a8c", "#ea0027", "#2f9e4a"],
    dark: ["#030303", "#1a1a1b", "#272729", "#343536", "#474748", "#d7dadc", "#818384", "#6f7071", "#ff585b", "#46d160"],
  },
  {
    id: "frost",
    name: "Frost",
    accent: "#0a84ff",
    font: SANS_APPLE,
    r: 14,
    rc: 22,
    rb: 999,
    light: ["#f2f2f7", "rgba(255, 255, 255, 0.72)", "rgba(255, 255, 255, 0.92)", "rgba(0, 0, 0, 0.08)", "rgba(0, 0, 0, 0.18)", "#0b0b0d", "#6b6b75", "#9a9aa4", "#ff3b30", "#34c759"],
    dark: ["#07070a", "rgba(52, 52, 60, 0.5)", "rgba(72, 72, 82, 0.62)", "rgba(255, 255, 255, 0.1)", "rgba(255, 255, 255, 0.22)", "#f5f5f7", "#a9a9b4", "#74747e", "#ff453a", "#30d158"],
    // Translucent surfaces need an opaque layer under dialogs and the sidebar.
    solid: ["#ffffff", "#1d1d22"],
  },
  {
    id: "gloss",
    name: "Gloss",
    accent: "#0b60d8",
    font: '"Helvetica Neue", Helvetica, Arial, sans-serif',
    r: 8,
    rc: 10,
    light: ["#c5ccd3", "#ffffff", "#d9dde2", "#a9b1b9", "#8b959f", "#1b1f24", "#5c6773", "#8e99a4", "#d9291c", "#3d9c3d"],
    dark: ["#23262b", "#30343a", "#3b4047", "#1a1c20", "#121417", "#eef1f4", "#a7afb9", "#7b838d", "#e5544a", "#4cb04c"],
  },
  {
    id: "grid",
    name: "GRiD",
    accent: "#ffb000",
    font: '"IBM Plex Mono", "Roboto Mono", "SF Mono", Menlo, Consolas, monospace',
    r: 0,
    rc: 0,
    dark: [ACC(4, "#050505"), ACC(8, "#050505"), ACC(18, "#050505"), ACC(26, "#050505"), ACC(42, "#050505"), "var(--accent)", ACC(74, "#050505"), ACC(50, "#050505"), "#ff5c33", "var(--accent)"],
    light: [ACC(7, "#fbfaf6"), ACC(4, "#ffffff"), ACC(16, "#ffffff"), ACC(30, "#ffffff"), ACC(50, "#ffffff"), ACC(45, "#000000"), ACC(55, "#3a3a3a"), ACC(45, "#8a8a8a"), "#d9441f", ACC(45, "#000000")],
  },
  {
    id: "groove",
    name: "Groove",
    accent: "#1ed760",
    font: '"Helvetica Neue", Helvetica, Arial, sans-serif',
    r: 8,
    rc: 8,
    rb: 999,
    dark: ["#121212", "#181818", "#282828", "#282828", "#3e3e3e", "#ffffff", "#b3b3b3", "#6a6a6a", "#f15e6c", "#1ed760"],
    light: ["#ffffff", "#f6f6f6", "#ebebeb", "#ebebeb", "#d4d4d4", "#121212", "#6a6a6a", "#9a9a9a", "#e0485a", "#1a9e4a"],
  },
  {
    id: "honey",
    name: "Honey",
    accent: "#ffcc00",
    font: 'Inter, "Helvetica Neue", Arial, sans-serif',
    r: 12,
    rc: 16,
    light: ["#f5f4f2", "#ffffff", "#f1f0ed", "#f0efec", "#e0ded9", "#000000", "#5c5c5c", "#999999", "#ff3333", "#3bb300"],
    dark: ["#121212", "#1f1f1f", "#2a2a2a", "#2a2a2a", "#3a3a3a", "#ffffff", "#a8a8a8", "#7a7a7a", "#ff3333", "#3bb300"],
  },
  {
    id: "ink",
    name: "Ink",
    accent: "#1d9bf0",
    font: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    r: 12,
    rc: 16,
    rb: 999,
    dark: ["#000000", "#000000", "#16181c", "#2f3336", "#536471", "#e7e9ea", "#71767b", "#5b6065", "#f4212e", "#00ba7c"],
    light: ["#ffffff", "#ffffff", "#f7f9f9", "#eff3f4", "#cfd9de", "#0f1419", "#536471", "#8b98a5", "#f4212e", "#00ba7c"],
  },
  {
    id: "lounge",
    name: "Lounge",
    accent: "#5865f2",
    font: '"Noto Sans", "Helvetica Neue", Helvetica, Arial, sans-serif',
    r: 4,
    rc: 8,
    dark: ["#313338", "#2b2d31", "#35373c", "#3f4147", "#4e5058", "#dbdee1", "#b5bac1", "#949ba4", "#f23f43", "#23a55a"],
    light: ["#ffffff", "#f2f3f5", "#e3e5e8", "#e3e5e8", "#c4c9ce", "#313338", "#4e5058", "#5c5e66", "#f23f43", "#23a55a"],
  },
  {
    id: "material",
    name: "Material 3",
    accent: "#6750a4",
    font: 'Roboto, "Roboto Flex", "Google Sans Text", -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", sans-serif',
    r: 12,
    rc: 20,
    rb: 20,
    light: ["#f7f2fa", "#ffffff", "#ece6f0", "#e3dce6", "#cac4d0", "#1d1b20", "#49454f", "#79747e", "#b3261e", "#386a20"],
    dark: ["#141218", "#211f26", "#2b2930", "#322f37", "#49454f", "#e6e0e9", "#cac4d0", "#938f99", "#f2b8b5", "#9ccf7f"],
  },
  {
    id: "metro",
    name: "Metro",
    accent: "#00a4ef",
    font: '"Segoe UI", "Segoe UI Variable Text", Tahoma, sans-serif',
    r: 0,
    rc: 0,
    dark: ["#1d1d1d", "#2b2b2b", "#3a3a3a", "#1d1d1d", "#4c4c4c", "#ffffff", "#b9b9b9", "#8a8a8a", "#e51400", "#60a917"],
    light: ["#ffffff", "#f2f2f2", "#e1e1e1", "#ffffff", "#c8c8c8", "#000000", "#4c4c4c", "#767676", "#e51400", "#4f8c13"],
  },
  {
    id: "mmo",
    name: "MMO",
    accent: "#00e5ff",
    accentLight: "#008a99",
    accentDark: "#a8843a",
    font: 'Inter, Tahoma, "Segoe UI", sans-serif',
    display: '"Cinzel", "Trajan Pro", "Times New Roman", serif',
    r: 3,
    rc: 6,
    dark: ["#0f1217", "#14191f", "#262e38", "#3a4454", "#8a97a8", "#f0f4f8", "#9ba5b7", "#6b7588", "#ff2a4b", "#00e5ff"],
    light: ["#d6d1c4", "#f6f3eb", "#ded7c6", "#c2baa6", "#8f8878", "#1d1c19", "#4d4a42", "#7a7566", "#d9203f", "#00838f"],
  },
  {
    id: "neon",
    name: "Neon",
    accent: "#fe2c55",
    font: '"Proxima Nova", Inter, -apple-system, "Segoe UI", Arial, sans-serif',
    r: 4,
    rc: 8,
    dark: ["#000000", "#121212", "#1f1f1f", "#1f1f1f", "#2f2f2f", "#ffffff", "#a8a8a8", "#767676", "#ff4040", "#25f4ee"],
    light: ["#ffffff", "#f1f1f2", "#e3e3e4", "#ededee", "#d6d6d8", "#161823", "#75767d", "#a2a2a7", "#e53030", "#0f9f9a"],
  },
  {
    id: "olive",
    name: "Olive",
    accent: "#c4b550",
    accentLight: "#000080",
    font: 'Tahoma, Verdana, "Segoe UI", sans-serif',
    r: 0,
    rc: 0,
    dark: ["#3e4637", "#4c5844", "#5a6a50", "#3e4637", "#282e22", "#d8ded3", "#a0aa95", "#7f8a74", "#c2653e", "#95a86b"],
    light: ["#c0c0c0", "#c0c0c0", "#b8b8b8", "#a8a8a8", "#808080", "#000000", "#303030", "#5a5a5a", "#a8321e", "#3f6b1f"],
    // Fields are sunken: white on light, dark olive on dark.
    field: ["#ffffff", "#3e4637"],
  },
  {
    id: "page",
    name: "Page",
    accent: "#2383e2",
    font: 'ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif',
    display: 'Georgia, "Iowan Old Style", "Times New Roman", serif',
    r: 3,
    rc: 4,
    light: ["#ffffff", "#ffffff", "#f1f1ef", "#e9e9e7", "#d3d1cb", "#37352f", "#787774", "#9b9a97", "#e03e3e", "#448361"],
    dark: ["#191919", "#202020", "#2f2f2f", "#2f2f2f", "#3f3f3f", "#d4d4d4", "#9b9b9b", "#7f7f7f", "#e03e3e", "#5aa57b"],
  },
  {
    id: "patch",
    name: "Patch",
    accent: "#67c1f5",
    accentLight: "#1a75b8",
    font: '"Motiva Sans", Arial, Tahoma, Helvetica, sans-serif',
    r: 2,
    rc: 3,
    dark: ["#2e2e2e", "#252525", "#3d3d3d", "#333333", "#454545", "#e6e6e6", "#999797", "#6e6d6d", "#ff453a", "#a4d007"],
    light: ["#e9eef2", "#ffffff", "#dce2e8", "#e1e6eb", "#b8c2cc", "#1f2328", "#57606a", "#8c959f", "#d1242f", "#4b6b09"],
  },
  {
    id: "plain",
    name: "Plain",
    accent: "#ff6600",
    font: "Verdana, Geneva, Tahoma, sans-serif",
    r: 2,
    rc: 0,
    light: ["#f6f6ef", "#f6f6ef", "#ebebe0", "#e6e6dc", "#c8c8bc", "#000000", "#828282", "#a6a6a0", "#cc0000", "#3c8a2e"],
    dark: ["#1c1c19", "#1c1c19", "#2a2a25", "#2e2e29", "#45453e", "#e6e6dc", "#9a9a8f", "#75756c", "#e04a4a", "#5aa84a"],
  },
  {
    id: "thread",
    name: "Thread",
    accent: "#f07f2e",
    font: 'Roboto, "Segoe UI", Arial, sans-serif',
    r: 4,
    rc: 4,
    dark: ["#16202d", "#1d2a3a", "#253549", "#16202d", "#34485f", "#d7dde5", "#9aa7b6", "#6f7f92", "#ff4c4c", "#8bc34a"],
    light: ["#eeeeee", "#dddddd", "#d0d0d0", "#eeeeee", "#bbbbbb", "#333333", "#666666", "#888888", "#cc0000", "#4f8a10"],
  },
  {
    id: "vapor",
    name: "Vapor",
    accent: "#2d73ff",
    font: '"Motiva Sans", "Segoe UI", "Helvetica Neue", Arial, Helvetica, sans-serif',
    r: 3,
    rc: 4,
    dark: ["#25282e", "#2e323a", "#393f49", "#3a3f48", "#4a515c", "#dfe3e6", "#acb2b8", "#7e848b", "#c9433f", "#8ac329"],
    light: ["#dde1e6", "#f7f8fa", "#e1e5ea", "#dde1e6", "#b5bcc5", "#1c2026", "#565e69", "#8a929c", "#c23b37", "#4a7a16"],
  },
  {
    id: "wave",
    name: "Wave",
    accent: "#0077ff",
    font: 'Inter, -apple-system, BlinkMacSystemFont, Roboto, sans-serif',
    r: 8,
    rc: 12,
    light: ["#edeef0", "#ffffff", "#f0f2f5", "#e7e8ec", "#dce1e6", "#000000", "#818c99", "#99a2ad", "#e64646", "#4bb34b"],
    dark: ["#141414", "#222222", "#2c2d2e", "#363738", "#424242", "#e1e3e6", "#939393", "#76787a", "#e64646", "#4bb34b"],
  },
];

/** The app's own theme, defined in app.css. */
export const DEFAULT_THEME = { id: "heretic", name: "Heretic" };

// ---------- Color math (same formulas as the catalog's applyTheme) ----------

const isHex = (value) => /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.test(value);

function toRgb(hex) {
  let h = hex.slice(1);
  if (h.length === 3) h = [...h].map((c) => c + c).join("");
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function toHex(rgb) {
  return `#${rgb.map((v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0")).join("")}`;
}

export function mix(a, b, t) {
  const x = toRgb(a);
  const y = toRgb(b);
  return toHex(x.map((v, i) => v + (y[i] - v) * t));
}

function luminance(hex) {
  const c = toRgb(hex).map((v) => {
    v /= 255;
    return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}

function contrast(a, b) {
  const [l1, l2] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (l1 + 0.05) / (l2 + 0.05);
}

/** Accent shade for text and outlines on the surface: at least 4.5:1 where it can be computed. */
function accentInk(accent, surface, text) {
  if (!isHex(surface) || !isHex(text)) return "color-mix(in srgb, var(--accent) 72%, var(--text))";
  for (let t = 0; t <= 1.0001; t += 0.1) {
    const candidate = mix(accent, text, t);
    if (contrast(candidate, surface) >= 4.5) return candidate;
  }
  return text;
}

/** CSS variables for a catalog theme in one mode; an empty object means the app's own theme. */
export function themeVariables(id, dark) {
  const theme = THEMES.find((t) => t.id === id);
  if (!theme) return {};
  const [bg, surface, press, line, lineStrong, text, text2, text3, bad, good] = dark ? theme.dark : theme.light;
  const base = (dark ? theme.accentDark : theme.accentLight) ?? theme.accent;
  const accent = dark ? mix(base, "#ffffff", 0.18) : base;
  const hover = dark ? mix(accent, "#ffffff", 0.22) : mix(accent, "#000000", 0.18);
  const [r, g, b] = toRgb(accent);
  // White on mid-tone accents (blue, violet) where it holds 3:1 for 14px/500 button labels,
  // dark text on light accents (yellow, orange, lime).
  const onAccent = contrast(accent, "#ffffff") >= 3 ? "#ffffff" : "#101010";
  const pick = (pair) => (pair ? pair[dark ? 1 : 0] : null);

  return {
    "--bg": bg,
    "--surface": surface,
    "--surface-2": press,
    "--surface-elev": pick(theme.solid) ?? surface,
    "--field": pick(theme.field) ?? pick(theme.solid) ?? surface,
    "--border": line,
    "--border-strong": lineStrong,
    "--text": text,
    "--text-2": text2,
    "--text-3": text3,
    "--accent": accent,
    "--accent-2": hover,
    "--accent-soft": `rgba(${r}, ${g}, ${b}, ${dark ? 0.24 : 0.14})`,
    "--accent-ink": accentInk(accent, pick(theme.solid) ?? surface, text),
    "--on-accent": onAccent,
    "--good": good,
    // Semantic, not decorative: "interrupted" stays amber in every theme.
    "--warn": dark ? "#e0b04a" : "#9a6a00",
    "--bad": bad,
    "--font": theme.font,
    "--font-display": theme.display ?? theme.font,
    "--radius": `${theme.rc}px`,
    "--radius-sm": `${Math.min(theme.r, 10)}px`,
    "--radius-btn": `${theme.rb ?? Math.min(theme.r, 10)}px`,
  };
}

/** Colors for a small preview tile in the theme picker. */
export function themePreview(id, dark) {
  const theme = THEMES.find((t) => t.id === id);
  if (!theme) {
    return dark
      ? { bg: "#141414", surface: "#1b1b1b", line: "#2b2b2b", text: "#ececea", muted: "#74746f", accent: "#f2a93b" }
      : { bg: "#f4f4f3", surface: "#ffffff", line: "#e3e3e0", text: "#1b1b1a", muted: "#8a8a85", accent: "#f0a020" };
  }
  const vars = themeVariables(id, dark);
  const resolve = (value) => value.replaceAll("var(--accent)", vars["--accent"]);
  return {
    bg: resolve(vars["--bg"]),
    surface: resolve(vars["--surface-elev"]),
    line: resolve(vars["--border-strong"]),
    text: resolve(vars["--text"]),
    muted: resolve(vars["--text-3"]),
    accent: vars["--accent"],
    radius: vars["--radius-sm"],
    font: theme.font,
  };
}

export const THEME_VARIABLES = Object.keys(themeVariables("wave", false));
