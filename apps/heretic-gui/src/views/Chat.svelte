<script lang="ts">
  import { api, type Message } from "../lib/api";
  import { app, t } from "../lib/store.svelte";

  interface Turn {
    user: string;
    modified?: string;
    original?: string;
  }

  let system = $state(app.settings?.system_prompt ?? "You are a helpful assistant.");
  let compare = $state(false);
  let useAdapters = $state(true);
  let turns = $state<Turn[]>([]);
  let input = $state("");
  let generating = $state<"modified" | "original" | null>(null);
  let error = $state<string | null>(null);
  let list: HTMLDivElement | undefined = $state();

  $effect(() => {
    // Keep the newest text visible while streaming.
    app.chatStream;
    turns.length;
    if (list) queueMicrotask(() => list && (list.scrollTop = list.scrollHeight));
  });

  function history(side: "modified" | "original"): Message[] {
    const messages: Message[] = [{ role: "system", content: system }];
    for (const turn of turns) {
      messages.push({ role: "user", content: turn.user });
      const answer = turn[side];
      if (answer !== undefined) messages.push({ role: "assistant", content: answer });
    }
    return messages;
  }

  async function generate(side: "modified" | "original") {
    generating = side;
    app.chatStream = "";
    try {
      const result = await api.chatSend(history(side), side === "modified");
      turns[turns.length - 1][side] = result.response;
    } catch (e) {
      // A stopped generation keeps what was produced so far.
      turns[turns.length - 1][side] = app.chatStream || undefined;
      if (!String(e).toLowerCase().includes("cancel")) error = String(e);
    } finally {
      app.chatStream = "";
      generating = null;
    }
  }

  async function send() {
    const text = input.trim();
    if (!text || generating) return;
    input = "";
    error = null;
    turns.push({ user: text });
    if (compare) {
      await generate("modified");
      await generate("original");
    } else {
      await generate(useAdapters ? "modified" : "original");
    }
  }

  function onkey(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  }

  const sides = $derived<("modified" | "original")[]>(compare ? ["modified", "original"] : [useAdapters ? "modified" : "original"]);
</script>

{#if !app.modelLoaded || app.appliedTrial === null}
  <div class="card empty muted">{t("chatNoModel")}</div>
{:else}
  <div class="chat">
    <div class="toolbar row">
      <div class="segmented">
        <button class:active={!compare && useAdapters} onclick={() => ((compare = false), (useAdapters = true))}>{t("chatModified")}</button>
        <button class:active={!compare && !useAdapters} onclick={() => ((compare = false), (useAdapters = false))}>{t("chatOriginal")}</button>
        <button class:active={compare} onclick={() => (compare = true)}>⇆ {t("chatCompare")}</button>
      </div>
      <span class="grow"></span>
      <button class="ghost" disabled={!!generating} onclick={() => (turns = [])}>{t("chatClear")}</button>
    </div>

    <details class="system">
      <summary class="muted">{t("chatSystem")}</summary>
      <textarea rows="2" bind:value={system}></textarea>
    </details>

    <div class="messages" bind:this={list}>
      {#each turns as turn, i}
        <div class="user bubble">{turn.user}</div>
        <div class="answers" class:two={compare}>
          {#each sides as side}
            <div class="assistant bubble">
              {#if compare}<span class="badge {side === 'modified' ? 'accent' : ''}">{side === "modified" ? t("chatModified") : t("chatOriginal")}</span>{/if}
              {#if turn[side] !== undefined}
                <div class="text">{turn[side]}</div>
              {:else if generating === side && i === turns.length - 1}
                <div class="text">{app.chatStream}<span class="cursor">▍</span></div>
              {:else}
                <div class="text muted">…</div>
              {/if}
            </div>
          {/each}
        </div>
      {/each}
    </div>

    {#if error}<div class="error-box">{error}</div>{/if}

    <div class="composer row">
      <textarea rows="2" placeholder={t("chatPlaceholder")} bind:value={input} onkeydown={onkey}></textarea>
      {#if generating}
        <button class="danger" onclick={() => api.chatStop()}>■ {t("chatStop")}</button>
      {:else}
        <button class="primary" disabled={!input.trim()} onclick={send}>{t("chatSend")}</button>
      {/if}
    </div>
  </div>
{/if}

<style>
  .chat {
    display: flex;
    flex-direction: column;
    gap: 12px;
    height: calc(100vh - 56px);
  }

  .messages {
    flex: 1;
    overflow: auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 4px;
  }

  .bubble {
    border-radius: var(--radius);
    padding: 10px 14px;
    max-width: 85%;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .user {
    align-self: flex-end;
    background: var(--accent);
    color: var(--on-accent);
  }

  .answers {
    display: grid;
    gap: 10px;
  }

  .answers.two {
    grid-template-columns: 1fr 1fr;
  }

  .answers.two .bubble {
    max-width: 100%;
  }

  .assistant {
    background: var(--surface);
    border: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .cursor {
    animation: blink 1s infinite;
  }

  @keyframes blink {
    50% {
      opacity: 0;
    }
  }

  .composer textarea {
    flex: 1;
  }

  .system textarea {
    margin-top: 6px;
  }

  .empty {
    padding: 40px;
    text-align: center;
  }
</style>
