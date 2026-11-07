<script lang="ts">
  // Scrolling log of backend messages.
  import type { LogLine } from "../store.svelte";
  import { segments } from "../api";

  interface Props {
    lines: LogLine[];
    height?: string;
  }

  let { lines, height = "260px" }: Props = $props();

  let container: HTMLDivElement | undefined = $state();
  let follow = $state(true);

  $effect(() => {
    // Re-run when lines change.
    lines.length;
    if (follow && container) {
      queueMicrotask(() => container && (container.scrollTop = container.scrollHeight));
    }
  });

  function onscroll() {
    if (!container) return;
    follow = container.scrollHeight - container.scrollTop - container.clientHeight < 24;
  }
</script>

<div class="log mono" style="height: {height}" bind:this={container} {onscroll}>
  {#each lines as line, i (i)}
    <div class="line {line.level}">
      {#each segments(line.text) as part}
        {#if part.bold}<b>{part.text}</b>{:else}{part.text}{/if}
      {/each}
    </div>
  {/each}
</div>

<style>
  .log {
    overflow: auto;
    background: var(--surface-2);
    border-radius: var(--radius-sm);
    padding: 10px 12px;
    font-size: 12px;
    line-height: 1.55;
  }

  .line {
    white-space: pre-wrap;
    word-break: break-word;
  }

  .line.heading {
    margin-top: 8px;
    font-weight: 600;
  }

  .line.detail,
  .line.debug {
    color: var(--text-2);
  }

  .line.success {
    color: var(--good);
  }

  .line.warning {
    color: var(--warn);
  }

  .line.error {
    color: var(--bad);
  }
</style>
