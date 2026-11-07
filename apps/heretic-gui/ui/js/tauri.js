// SPDX-License-Identifier: AGPL-3.0-or-later
// Access to the Tauri API through the global object (app.withGlobalTauri = true),
// so the frontend needs no bundler or npm packages.

const tauri = window.__TAURI__;

if (!tauri) {
  throw new Error("The Tauri API is not available");
}

export const invoke = (command, args) => tauri.core.invoke(command, args);
export const listen = (event, handler) => tauri.event.listen(event, handler);
export const openDialog = (options) => tauri.dialog.open(options);
