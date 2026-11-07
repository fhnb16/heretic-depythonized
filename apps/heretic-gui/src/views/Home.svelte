<script lang="ts">
  import { api, type Project } from "../lib/api";
  import { launch } from "../lib/actions";
  import { app, prefs, t } from "../lib/store.svelte";

  let projects = $state<Project[]>([]);
  let loading = $state(true);

  async function refresh() {
    loading = true;
    try {
      projects = prefs.checkpointDir ? await api.listProjects(prefs.checkpointDir) : [];
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    // Reload when the checkpoint folder changes.
    prefs.checkpointDir;
    refresh();
  });

  async function open(project: Project) {
    if (!project.settings) return;
    await launch(project.settings, "continue");
  }

  async function remove(project: Project) {
    if (!confirm(t("projectDeleteConfirm"))) return;
    await api.deleteProject(project.path);
    await refresh();
  }

  function date(seconds: number | null) {
    if (seconds == null) return "";
    return new Date(seconds * 1000).toLocaleString(prefs.lang === "ru" ? "ru-RU" : "en-US", {
      dateStyle: "medium",
      timeStyle: "short",
    });
  }
</script>

<div class="home">
  <section class="hero card">
    <div class="stack">
      <h1>{t("homeTitle")}</h1>
      <p class="muted">{t("homeSubtitle")}</p>
    </div>
    <button class="primary big" onclick={() => (app.view = "new")}>＋ {t("homeStart")}</button>
  </section>

  <section class="stack">
    <h2>{t("homeProjects")}</h2>
    {#if !loading && projects.length === 0}
      <p class="muted empty card">{t("homeNoProjects")}</p>
    {/if}
    <div class="grid">
      {#each projects as project (project.path)}
        <div class="card project">
          <div class="row">
            <h3 class="grow name" title={project.model}>{project.model || "?"}</h3>
            {#if project.finished}
              <span class="badge good">{t("projectFinished")}</span>
            {:else}
              <span class="badge warn">{t("projectInterrupted")}</span>
            {/if}
          </div>
          <div class="progress">
            <div style="width: {project.n_trials ? Math.min(100, (project.completed_trials / project.n_trials) * 100) : 0}%"></div>
          </div>
          <p class="muted small">
            {project.completed_trials}{project.n_trials ? ` / ${project.n_trials}` : ""}
            {t("projectTrials")} · {date(project.modified)}
          </p>
          <div class="row">
            <button class="primary" disabled={app.status === "preparing" || app.status === "running"} onclick={() => open(project)}>
              {project.finished ? t("projectResults") : t("projectContinue")}
            </button>
            <span class="grow"></span>
            <button class="ghost danger" onclick={() => remove(project)}>{t("projectDelete")}</button>
          </div>
        </div>
      {/each}
    </div>
  </section>
</div>

<style>
  .home {
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  .hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    padding: 28px;
    background: linear-gradient(135deg, var(--accent-soft), var(--surface) 70%);
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 14px;
  }

  .project {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .small {
    font-size: 13px;
  }

  .empty {
    padding: 24px;
    text-align: center;
  }
</style>
