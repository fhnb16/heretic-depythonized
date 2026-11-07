<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    title: string;
    onclose: () => void;
    width?: string;
    children: Snippet;
    footer?: Snippet;
  }

  let { title, onclose, width = "520px", children, footer }: Props = $props();
</script>

<svelte:window onkeydown={(e) => e.key === "Escape" && onclose()} />

<div class="backdrop" role="presentation" onclick={(e) => e.target === e.currentTarget && onclose()}>
  <div class="modal card" style="width: {width}" role="dialog" aria-modal="true" aria-label={title}>
    <div class="head row">
      <h2 class="grow">{title}</h2>
      <button class="ghost" aria-label="Close" onclick={onclose}>✕</button>
    </div>
    <div class="body">{@render children()}</div>
    {#if footer}
      <div class="foot row">{@render footer()}</div>
    {/if}
  </div>
</div>

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    background: rgba(5, 8, 15, 0.45);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 50;
    padding: 20px;
  }

  .modal {
    max-width: 100%;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .body {
    overflow: auto;
  }

  .foot {
    justify-content: flex-end;
  }
</style>
