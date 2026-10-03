/**
 * Shared runtime for HOOK-based harnesses (Claude Code, Codex, Cursor CLI, ...).
 *
 * The ONE runtime path (see docs/superpowers/specs/2026-07-27-reflect-pages-runtime.md):
 *   - AUTO-INJECT once per session, on the first prompt, from the source `cfg.autoInject` names:
 *     "reflect" (default) is an agentic synthesis over the bank returning the root-cause decision
 *     with exact values; "pages" is a knowledge-page search and "recall" a raw memory recall, both
 *     retrieval-only and far cheaper; "none" injects nothing and leaves it to the tools. Cached
 *     per session, injected on the turn it ran.
 *   - KNOWLEDGE PAGES every turn: the page set is fetched on a cadence and matched LOCALLY against
 *     the prompt (section-level lexical scoring — no server/LLM call); the top sections are
 *     injected with provenance. Fast like recall, organized like reflect.
 *   - The tool-guide/page-roster block re-injects on the same cadence.
 *
 * Every outcome is recorded in the diagnostics file — a memory-less session can't masquerade as a
 * memory session. A failure never breaks the agent.
 *
 * A harness plugs in with a HookSpec: its name, how to read (prompt, cwd, sessionId) from its
 * stdin event, and how to wrap injected context in its native output schema. The pure logic lives
 * in `buildHookOutput` (client + cache file in, injection string out) so it's unit-testable
 * without stdin/stdout; `runHook` is thin plumbing around it, with a `makeClient` seam for tests.
 */
import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { deriveBankIdOrSkip } from "./bank";
import type { Config } from "./config";
import { applyBankConfig, loadConfig } from "./config";
import { diag, diagFilePath } from "./diag";
import { describeError, log, setLogLevel } from "./log";
import { startBackgroundSeed } from "./seed";
import type { ClientOpts } from "./hindsight";
import { HindsightClient, ReflectError } from "./hindsight";
import { brandWord } from "./brand";
import {
  buildReflectQuery,
  buildSystemInjection,
  formatPageFallback,
  formatRecallFallback,
  PAGE_INJECT_LEAD,
  RECALL_INJECT_LEAD,
} from "./inject";
import type { PageRef } from "./knowledge-injection";
import { buildRosterRefresh, parsePageList } from "./knowledge-injection";
import {
  clearReflectJob,
  readReflectJob,
  readSessionCache,
  reflectJobFile,
  sessionCacheFile,
  sessionRootDir,
  writeReflectJob,
  writeSessionCache,
  type ReflectJob,
  type SessionCache,
} from "./session-cache";
import { appendJournalTurn, journalPath } from "./turn-journal";

export interface HookEventFields {
  prompt?: string;
  cwd?: string;
  sessionId?: string;
}

export interface HookSpec {
  /** Harness name — config `harnesses.<name>` section, {harness} template field, diag records. */
  harness: string;
  /** Read the fields out of the harness's stdin event (shapes differ per harness). */
  parse(event: Record<string, unknown>): HookEventFields;
  /** Optional event gate, evaluated before config loading: false makes the hook a silent no-op. */
  accept?(event: Record<string, unknown>): boolean;
  /** Some hosts execute hook commands from their global config directory. Those hosts must provide
   * a workspace path in the event; falling back to process.cwd() would create a bank for config. */
  requireCwd?: boolean;
  /** Record this prompt in the session's own turn journal (core/turn-journal.ts). Set ONLY by the
   *  harnesses whose host keeps no durable transcript, because for them the journal IS the
   *  transcript their Stop hook retains — see the ZCode and TraeCode entries in
   *  harness/hook-lifecycle.ts. */
  journalPrompt?: boolean;
  /** Wrap injected context (and an optional user-facing notice) in the harness's native
   *  hook-output schema. Harnesses whose schema has no user-visible channel ignore `notice`. */
  emit(context: string, notice?: string, event?: Record<string, unknown>): unknown;
}

/** Minimal client shape `buildHookOutput` needs — `HindsightClient` satisfies it structurally. */
interface HookClient {
  reflect(query: string, opts: { budget?: string; timeoutMs: number }): Promise<string>;
  listPages(): Promise<unknown>;
  searchKnowledgePages(
    query: string,
    opts?: { limit?: number; timeoutMs?: number }
  ): Promise<{ id: string; name: string; snippet: string }[]>;
  recallObservations(query: string, opts: { timeoutMs: number }): Promise<string[]>;
  knowledgePagesSupported?: boolean;
  /** Recorded on reflect failures so the diag trail says which bank to look at server-side. */
  readonly bank?: string;
}

/** How many turns auto-inject may FAIL on before a session gives up on memory. The budget is
 *  turns, not time: each retry costs another full attempt (up to `reflectTimeoutMs` on the
 *  reflect path), so a dead server spends this many turns before every later turn is free. */
const HOOK_INJECT_ATTEMPTS = 2;

/** argv flag that turns a prompt-hook binary into the reflect worker (see `runReflectWorker`). */
export const REFLECT_WORKER_FLAG = "--reflect-worker";
/** How long the worker keeps one synthesis alive. Well past any hook window, short of forever. */
const LATE_REFLECT_TIMEOUT_MS = 120_000;
/** A pending job older than this has no live worker behind it; stop waiting for it. */
const LATE_REFLECT_MAX_AGE_MS = LATE_REFLECT_TIMEOUT_MS + 30_000;
/** Opens a synthesis delivered on a later prompt, so the agent reads it against the right goal. */
export const LATE_REFLECT_LEAD =
  "(This synthesis was written for the request that opened this session. It was not ready in time " +
  "for that turn and is arriving now.)";

/**
 * Runs the first-prompt synthesis in a process that outlives the hook, so a synthesis slower than
 * `reflectTimeoutMs` is delivered on a later prompt instead of being discarded. Supplied by
 * `runHook` on the hook harnesses; absent (persistent-plugin harnesses, tests) the hook reflects
 * in-process exactly as before.
 */
export interface LateReflect {
  /** Record the job and start the worker. False = could not start; reflect in-process instead. */
  start(request: { query: string; budget: string }): boolean;
  read(): ReflectJob | undefined;
  clear(): void;
  /** How often the hook looks for the worker's answer while it waits. */
  pollMs?: number;
}

/** The worker is still writing the synthesis when the hook's own deadline passes. A timeout as far
 *  as this turn is concerned (so the retrieval fallback runs), but not a lost answer. */
class ReflectPendingError extends ReflectError {
  constructor(timeoutMs: number) {
    super(
      `reflect still running after ${timeoutMs}ms; it will be delivered on a later prompt`,
      undefined,
      true
    );
    this.name = "ReflectPendingError";
  }
}

/** Start the worker and wait for its answer until the hook's deadline. */
async function reflectViaWorker(late: LateReflect, timeoutMs: number): Promise<string> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const job = late.read();
    if (job?.state === "ready") {
      late.clear();
      return job.answer ?? "";
    }
    if (!job || job.state === "failed") {
      late.clear();
      throw new ReflectError(
        job?.error ?? "reflect worker left no result",
        job?.status,
        job?.timedOut === true
      );
    }
    if (Date.now() >= deadline) throw new ReflectPendingError(timeoutMs);
    await new Promise((resolve) => setTimeout(resolve, late.pollMs ?? 200));
  }
}

/**
 * What to cache for a once-per-session auto-injection, given what the source returned.
 *
 * `null` (ran, nothing to say) resolves the session — `""` is cached and no later turn retries.
 * `undefined` (failed) stays unresolved so a later turn can try again, until the attempt budget
 * is spent. Caching a failure as an answer is what made one first-prompt timeout cost a whole
 * session's memory (#4607); applied to all three sources, not just reflect.
 */
function resolveInjection(got: string | null | undefined, attempts: number): string | undefined {
  if (got != null) return got;
  return got === null || attempts >= HOOK_INJECT_ATTEMPTS ? "" : undefined;
}

/** Knowledge-page search for the prompt, formatted for injection. `null` = it ran and nothing
 *  matched (an answer); `undefined` = it FAILED (recorded as `${event}_failed`) and is worth
 *  retrying. The two used to be one value, which cached a transient failure as "this session has
 *  no memory" for every remaining turn (#4607). Never throws. */
async function injectPages(
  harness: string,
  prompt: string,
  client: HookClient,
  timeoutMs: number,
  event: string,
  lead?: string
): Promise<string | null | undefined> {
  const t0 = Date.now();
  try {
    // The search query rides in a GET query string; the goal's opening carries its keywords.
    // How MANY pages come back is the client's `pageSearchLimit`, shared with the MCP tool.
    const hits = await client.searchKnowledgePages(prompt.slice(0, 500), { timeoutMs });
    diag(harness, event, { ms: Date.now() - t0, count: hits.length });
    return hits.length ? formatPageFallback(hits, lead) : null;
  } catch (e) {
    diag(harness, `${event}_failed`, { ms: Date.now() - t0, error: describeError(e) });
    return undefined;
  }
}

/** Raw recall over the bank (what it asks for is the client's `recallOptions`), formatted for
 *  injection; same contract as `injectPages`. */
async function injectRecall(
  harness: string,
  prompt: string,
  client: HookClient,
  timeoutMs: number,
  event: string,
  lead?: string
): Promise<string | null | undefined> {
  const t0 = Date.now();
  try {
    // What is asked for — types, token budget, everything — is the client's `recallOptions`.
    const observations = await client.recallObservations(prompt.slice(0, 2000), { timeoutMs });
    diag(harness, event, { ms: Date.now() - t0, count: observations.length });
    return observations.length ? formatRecallFallback(observations, lead) : null;
  } catch (e) {
    diag(harness, `${event}_failed`, { ms: Date.now() - t0, error: describeError(e) });
    return undefined;
  }
}

/**
 * Reflect timed out or 5xx'd: the synthesis path broke, but retrieval may still answer. Try the
 * curated knowledge pages first (search), and only when none match fall back to a raw recall
 * over the bank's memories. Returns the memory body to inject, or a nullish value when both came
 * back empty or failed — the caller only asks whether there IS a body, so the sources' tri-state
 * is passed through rather than flattened. Never throws.
 */
async function reflectFallback(
  harness: string,
  prompt: string,
  client: HookClient,
  timeoutMs: number
): Promise<string | null | undefined> {
  // Page search and recall share ONE retrieval budget after reflect fails, rather than each
  // spending a full injectTimeoutMs and doubling the time added to the host's hook window.
  const deadline = Date.now() + timeoutMs;
  const remaining = () => Math.max(deadline - Date.now(), 1);
  return (
    (await injectPages(harness, prompt, client, remaining(), "reflect_fallback_pages")) ??
    // Event name predates the `injectRecall` rename and is kept: it is a logged contract that
    // other tools read, so renaming it would silently break them.
    (await injectRecall(harness, prompt, client, remaining(), "reflect_fallback_observations"))
  );
}

export interface HookOutput {
  /** The model-facing injection block, or undefined when there's nothing to inject. */
  context?: string;
  /** User-facing line(s) — set only on the reflect turn (its goal + result preview). */
  notice?: string;
  /** This session's knowledge-page roster, from the shared per-session cache. The SINGLE source
   *  of truth for every page-derived block in a turn: the persistent-plugin runtime rebuilds its
   *  SessionStart preamble from this too, so a turn can no longer be told both "no pages yet" and
   *  a list of pages in the same context (#4607). Empty also signals a bank the engine never built. */
  pages: PageRef[];
}

/**
 * Pure hook logic: reflect once per session (cached); knowledge-page sections + roster on every
 * turn. Returns the injection plus a per-turn user-facing notice.
 */
export async function buildHookOutput(args: {
  harness: string;
  prompt: string;
  cfg: Config;
  client: HookClient;
  cacheFile: string;
  /** Present on the hook harnesses: lets a slow synthesis be delivered on a later prompt. */
  lateReflect?: LateReflect;
}): Promise<HookOutput> {
  const { harness, prompt, cfg, client, cacheFile, lateReflect } = args;

  const cached = readSessionCache(cacheFile);
  const turns = (cached.turns ?? 0) + 1;

  // ── auto-inject: once per session, on the first prompt ────────────────────────
  // cfg.autoInject picks the source: a reflect synthesis, a knowledge-page search, or a recall of
  // observations. "none" = tool-only mode: the roster's tool guide instead sends new goals through
  // knowledge pages first and reserves reflection for gaps. Whatever the source, its body is
  // cached as `reflectAnswer` (the field name predates the other sources).
  let reflectAnswer = cached.reflectAnswer;
  let reflectAttempts = cached.reflectAttempts ?? 0;
  let reflectRanThisTurn = false;
  // Set by whichever source actually FAILED — reflect's catch, or a pages/recall helper returning
  // undefined. An empty answer is not a failure: a source can legitimately have nothing to say on a
  // sparse bank (diag records reflect_empty), and reporting that would tell the user the plugin
  // broke on exactly the sessions where it did not.
  let reflectFailed = false;
  // Set when reflect timed out / 5xx'd and a retrieval-only fallback supplied the memory instead.
  let fallback: string | null | undefined;
  // ── a synthesis an earlier prompt's hook could not wait for ───────────────────
  // The worker has had the user's whole turn to finish. Delivered once, here, or dropped for good
  // if the worker failed or died.
  let reflectPending = cached.reflectPending === true;
  let lateAnswer: string | undefined;
  if (reflectPending && lateReflect) {
    const job = lateReflect.read();
    if (job?.state === "ready") {
      lateAnswer = job.answer || undefined;
      diag(harness, job.answer ? "reflect_late_ok" : "reflect_late_empty", {
        ms: job.ms,
        chars: (job.answer ?? "").length,
        answer: (job.answer ?? "").slice(0, 8000),
      });
      lateReflect.clear();
      reflectPending = false;
    } else if (
      !job ||
      job.state === "failed" ||
      Date.now() - job.startedAt > LATE_REFLECT_MAX_AGE_MS
    ) {
      diag(harness, "reflect_late_failed", {
        bank: client.bank,
        error: job?.error ?? "worker left no result",
      });
      lateReflect.clear();
      reflectPending = false;
    }
  } else if (reflectPending) {
    reflectPending = false; // nothing can deliver it in this process
  }

  const deferInitialReflect = cached.deferInitialReflect === true;
  if (deferInitialReflect) {
    // A new bank has no useful history yet. Do not burn the once-per-session synthesis on prompt
    // one; this marker is deliberately consumed below so prompt two remains eligible to reflect.
    diag(harness, "reflect_deferred_new_bank", { query: prompt.slice(0, 80) });
  } else if (cfg.autoInject === "pages" && reflectAnswer === undefined) {
    reflectRanThisTurn = true;
    reflectAttempts++;
    const got = await injectPages(
      harness,
      prompt,
      client,
      cfg.injectTimeoutMs,
      "inject_pages",
      PAGE_INJECT_LEAD
    );
    // `undefined` is the search FAILING, distinct from `null` (it ran, nothing matched). These
    // modes stayed silent on a failure while reflect announced one — the same turn looked healthy
    // whether memory was unavailable or simply had nothing to say.
    if (got === undefined) reflectFailed = true;
    reflectAnswer = resolveInjection(got, reflectAttempts);
  } else if (cfg.autoInject === "recall" && reflectAnswer === undefined) {
    reflectRanThisTurn = true;
    reflectAttempts++;
    const got = await injectRecall(
      harness,
      prompt,
      client,
      cfg.injectTimeoutMs,
      "inject_recall",
      RECALL_INJECT_LEAD
    );
    if (got === undefined) reflectFailed = true; // see the pages branch above
    reflectAnswer = resolveInjection(got, reflectAttempts);
  } else if (cfg.autoInject === "reflect" && reflectAnswer === undefined) {
    reflectRanThisTurn = true;
    reflectAttempts++;
    const t0 = Date.now();
    // Previously clamped to a hardcoded 20s, which made a raised reflectTimeoutMs dead config on
    // this path (#4398); the 20s now lives in the default instead. Not clamped: a user who raises reflectTimeoutMs past the host's prompt-hook timeout (30s on
    // the hook harnesses) must raise that too — see the README's reflectTimeoutMs row.
    const timeoutMs = cfg.reflectTimeoutMs;
    try {
      // Automatic reflection runs inside the host's hook window. Hindsight's low budget is the
      // supported default for bounded reflect calls; callers that explicitly invoke the MCP
      // tool still get the deeper high-budget path.
      const query = buildReflectQuery(prompt);
      reflectAnswer = lateReflect?.start({ query, budget: "low" })
        ? await reflectViaWorker(lateReflect, timeoutMs)
        : await client.reflect(query, { budget: "low", timeoutMs });
      diag(harness, reflectAnswer ? "reflect_ok" : "reflect_empty", {
        ms: Date.now() - t0,
        chars: reflectAnswer.length,
        query: prompt.slice(0, 80),
        // Verbatim injected synthesis — the ONLY durable record of what the agent actually saw
        // (post-mortems otherwise depend on the harness happening to persist hook context).
        answer: reflectAnswer.slice(0, 8000),
      });
    } catch (e) {
      // A failure is RETRYABLE, not an answer — `resolveInjection` leaves it unresolved until the
      // budget is spent. Caching "" here meant one timeout on the session's first prompt disabled
      // synthesis for the ENTIRE session, and against a real server a reflect near the timeout is a
      // coin flip, not an edge case (#4607).
      //
      // Still running in the worker is neither: the answer is on its way, so this turn must not
      // schedule a second reflect (resolved as "") and a later prompt delivers it.
      reflectPending = e instanceof ReflectPendingError;
      reflectAnswer = reflectPending ? "" : resolveInjection(undefined, reflectAttempts);
      reflectFailed = !reflectPending;
      if (!reflectPending) {
        log.warn(harness, "reflect failed — session runs without memory", {
          error: describeError(e),
        });
      }
      diag(harness, reflectPending ? "reflect_pending" : "reflect_failed", {
        ms: Date.now() - t0,
        bank: client.bank,
        timeoutMs,
        // Wider than describeError's default: the server's error body is the useful part.
        error: describeError(e, 1500),
        query: prompt.slice(0, 80),
      });
      if (e instanceof ReflectError && e.fallbackEligible) {
        fallback = await reflectFallback(harness, prompt, client, cfg.injectTimeoutMs);
        // The fallback body is cached exactly like a reflect answer: injected once, not retried.
        if (fallback) reflectAnswer = fallback;
      }
    }
  }

  // ── knowledge-page roster (ids + titles only): refreshed on the cadence ────────
  const cadence = cfg.pageRefreshEveryTurns;
  const stale = !cached.pages || (cadence > 0 && turns - cached.pages.atTurn >= cadence);
  let pages = cached.pages?.list ?? [];
  if (stale) {
    const t0 = Date.now();
    try {
      pages = parsePageList(await client.listPages());
      diag(harness, "pages_ok", { ms: Date.now() - t0, count: pages.length });
    } catch (e) {
      diag(
        harness,
        client.knowledgePagesSupported === false ? "knowledge_pages_unavailable" : "pages_failed",
        {
          ms: Date.now() - t0,
          error: describeError(e),
        }
      );
    }
  }

  writeSessionCache(cacheFile, {
    turns,
    reflectAnswer,
    reflectAttempts,
    ...(reflectPending ? { reflectPending: true } : {}),
    pages: { atTurn: stale ? turns : (cached.pages?.atTurn ?? turns), list: pages },
  } satisfies SessionCache);

  const blocks: string[] = [];
  // Hook-injected context lands in the USER MESSAGE and persists in the transcript — unlike a
  // system prompt, it accumulates. The reflect block is injected exactly ONCE, the turn reflect
  // ran. No cadence re-injection: replaying the turn-1 synthesis at arbitrary later turns reads
  // as random noise once the session drifts, and the agent holds hindsight_reflect for a FRESH
  // pass if compaction or drift makes it need memory again.
  if (reflectAnswer && reflectRanThisTurn) blocks.push(buildSystemInjection(reflectAnswer));
  // The late synthesis follows the same once-only rule, one prompt (or more) after it was asked for.
  if (lateAnswer) blocks.push(buildSystemInjection(`${LATE_REFLECT_LEAD}\n\n${lateAnswer}`));
  // Knowledge pages are NOT auto-injected: the agent pulls them through
  // hindsight_search_knowledge_pages when a question warrants it — an unprompted injection on
  // every turn (even a plain "yes") read as phantom research. The roster below keeps the tool
  // and the page names in front of the agent.
  if (cadence > 0 && turns % cadence === 0) {
    blocks.push(
      buildRosterRefresh(pages, {
        reflectOnNewGoals: cfg.autoInject !== "reflect",
        extra: cfg.toolGuideExtra,
      })
    );
  }
  const kept = blocks.filter(Boolean);

  // User-facing notice ONLY on a turn that actually ran a source (reflect, pages or recall),
  // showing its assigned goal and a preview of what came back. Ordinary turns stay silent — page
  // knowledge is pulled via the hindsight_search_knowledge_pages tool, a visible tool call.
  let notice: string | undefined;
  if (lateAnswer) {
    const preview = lateAnswer.replace(/\s+/g, " ").trim();
    notice =
      `${brandWord()} · memory for this session's opening goal arrived late — added now\n` +
      `↳ ${preview.length > 140 ? `${preview.slice(0, 140)}…` : preview}`;
  } else if (reflectPending && reflectRanThisTurn && !fallback) {
    notice = `${brandWord()} · memory is still being written — it will be added on your next prompt`;
  } else if (fallback) {
    // Silent: the session still got memory, just not a synthesis. The notice used to say which
    // source answered and point at the diag file, but that read as an error on a turn that
    // worked; reflect_failed + reflect_fallback_* in the diag trail carry the details.
  } else if (reflectRanThisTurn && reflectAnswer) {
    const q = prompt.replace(/\s+/g, " ").trim();
    const excerpt = q.length > 48 ? `${q.slice(0, 48)}…` : q;
    // Page/recall bodies open with a fixed lead line; preview what came back, not that.
    const body =
      cfg.autoInject === "reflect" ? reflectAnswer : reflectAnswer.split("\n").slice(1).join("\n");
    const preview = body.replace(/\s+/g, " ").trim();
    notice =
      `${brandWord()} · goal: recall this repo's past decisions about “${excerpt}”\n` +
      `↳ ${preview.length > 140 ? `${preview.slice(0, 140)}…` : preview}`;
  } else if (reflectFailed) {
    // The failure is already in the diag trail and plugin.log, but both are files nobody is
    // tailing mid-session, so a memory-less session looked exactly like a healthy one (#3443).
    // One terse line pointing at the trail — not an explanation, and not advice to the agent:
    // this fires only on a turn that actually attempted a source, so at most
    // HOOK_INJECT_ATTEMPTS times per session.
    notice = `${brandWord()} · no memory this turn — see ${diagFilePath()}`;
  }

  return { context: kept.length ? kept.join("\n\n") : undefined, notice, pages };
}

/**
 * The reflect worker: the same hook binary, re-executed detached with `REFLECT_WORKER_FLAG`. It
 * makes the one reflect call the job describes and records the outcome in the job file. Never
 * throws and never writes to stdout — nothing is listening.
 */
export async function runReflectWorker(
  jobFile: string,
  makeClient: (opts: ClientOpts) => HookClient = (o) => new HindsightClient(o)
): Promise<void> {
  const job = readReflectJob(jobFile);
  const request = job?.request;
  if (!job || job.state !== "pending" || !request) return;
  const t0 = Date.now();
  const record = (outcome: Partial<ReflectJob>) => {
    try {
      writeReflectJob(jobFile, { ...job, ...outcome });
    } catch {
      /* the hook treats a job that never resolves as failed once it is too old */
    }
  };
  try {
    const cfg = loadConfig({ harness: request.harness });
    setLogLevel(cfg.logLevel);
    const client = makeClient({
      apiUrl: request.apiUrl,
      apiToken: cfg.apiToken,
      bank: request.bank,
    });
    const answer = await client.reflect(request.query, {
      budget: request.budget,
      timeoutMs: LATE_REFLECT_TIMEOUT_MS,
    });
    record({ state: "ready", answer, ms: Date.now() - t0 });
  } catch (e) {
    record({
      state: "failed",
      ms: Date.now() - t0,
      error: describeError(e, 1500),
      status: e instanceof ReflectError ? e.status : undefined,
      timedOut: e instanceof ReflectError ? e.timedOut : undefined,
    });
  }
}

/** File-backed `LateReflect` for one session: the job file sits beside the session cache and the
 *  worker is this same binary. `spawnFn` is injectable for tests. */
export function fileLateReflect(
  cacheFile: string,
  target: { harness: string; apiUrl: string; bank: string },
  spawnFn: typeof spawn = spawn
): LateReflect | undefined {
  // Only a bundled hook binary (`claude-hook.js`, `codex-hook.js`, …) can be re-executed as the
  // worker. Anything else — a test runner, a host that imports this module — reflects in-process.
  const script = process.argv[1];
  if (!script || !/-hook\.[cm]?js$/.test(script)) return undefined;
  const jobFile = reflectJobFile(cacheFile);
  return {
    start(request) {
      try {
        writeReflectJob(jobFile, {
          state: "pending",
          startedAt: Date.now(),
          request: { ...target, ...request },
        });
        const child = spawnFn(process.execPath, [script, REFLECT_WORKER_FLAG, jobFile], {
          detached: true,
          stdio: "ignore",
          windowsHide: true,
        });
        // spawn() failures often arrive asynchronously as an 'error' event; unhandled, that would
        // crash the hook. The job then never resolves and is dropped once it is too old.
        child.on("error", () => {});
        child.unref();
        return true;
      } catch {
        clearReflectJob(jobFile);
        return false;
      }
    },
    read: () => readReflectJob(jobFile),
    clear: () => clearReflectJob(jobFile),
  };
}

/** Run one hook invocation: stdin event in, (maybe) an injection object on stdout. */
export async function runHook(
  spec: HookSpec,
  makeClient: (opts: ClientOpts) => HookClient = (o) => new HindsightClient(o)
): Promise<void> {
  // Worker mode comes first: it is this binary re-executed by `fileLateReflect`, with no stdin
  // event to read and nothing to print.
  if (process.argv[2] === REFLECT_WORKER_FLAG && process.argv[3]) {
    return runReflectWorker(process.argv[3], makeClient);
  }
  // Anti-recursion: the codebase survey's own headless session sets this so its hooks are a no-op.
  if (process.env.HINDSIGHT_DISABLE_HOOKS) return;

  let ev: Record<string, unknown> = {};
  try {
    ev = JSON.parse(readFileSync(0, "utf8")) as Record<string, unknown>;
  } catch {
    return; // no/invalid event: stay silent
  }
  if (spec.accept && !spec.accept(ev)) return;
  const { prompt: rawPrompt, cwd: rawCwd, sessionId } = spec.parse(ev);
  if (spec.requireCwd && !rawCwd) return;
  const cwd = rawCwd || process.cwd();
  const prompt = (rawPrompt || "").trim();
  if (!prompt) return;

  let cfg = loadConfig({ harness: spec.harness });
  setLogLevel(cfg.logLevel);
  if (cfg.disabled) {
    log.debug(spec.harness, "hook skipped: disabled");
    return;
  }

  // Before any network work, and deliberately before the bank is resolved: the journal is this
  // session's only record of what the user said, and a slow or unreachable server must not be able
  // to cost the turn. Journaling under `retainSessions: false` writes a temp file nothing reads,
  // which is cheaper than threading that decision through two processes to find out.
  if (spec.journalPrompt) {
    appendJournalTurn(journalPath(spec.harness, sessionId), { role: "user", content: prompt });
  }

  const out = (context: string | undefined, notice?: string) =>
    process.stdout.write(JSON.stringify(spec.emit(context ?? "", notice, ev)));

  const sessionRoot = sessionRootDir(spec.harness, sessionId, cwd);
  const derived = deriveBankIdOrSkip(cfg, cwd, spec.harness, sessionRoot);
  if (derived === null) return; // repository unidentifiable: stay silent rather than invent a bank
  const resolved = applyBankConfig(cfg, derived, cwd);
  cfg = resolved.cfg;
  const bankId = resolved.bankId;
  if (cfg.disabled) {
    log.debug(spec.harness, "hook skipped: bank disabled via banks override", { bank: bankId });
    return;
  }
  const client = makeClient({
    apiUrl: cfg.apiUrl,
    apiToken: cfg.apiToken,
    bank: bankId,
    maxParallelRetains: cfg.maxParallelRetains,
    observationScopes: cfg.observationScopes,
    pageSearchLimit: cfg.pageSearchLimit,
    recallOptions: cfg.recallOptions,
  });
  const cacheFile = sessionCacheFile(spec.harness, sessionId || "no-session");

  // Safety net on EVERY harness: on the session's FIRST prompt (no cache file yet), fire the
  // ingestion engine. Normally the shared SessionStart lifecycle already did — the engine's
  // per-bank lock makes this a no-op — but a session that predates installation never received
  // SessionStart. This limited fallback heals that case without duplicating any survey logic.
  if (cfg.autoSeed !== false && !existsSync(cacheFile)) {
    startBackgroundSeed(cwd, { limit: cfg.seedLimit, harness: spec.harness });
  }

  const output = await buildHookOutput({
    harness: spec.harness,
    prompt,
    cfg,
    client,
    cacheFile,
    lateReflect: fileLateReflect(cacheFile, {
      harness: spec.harness,
      apiUrl: cfg.apiUrl,
      bank: bankId,
    }),
  });
  // Mid-session heal: a bank with ZERO pages means the engine never built it (e.g. the session
  // predates the install, so no SessionStart and the first-prompt net already passed). Fire the
  // idempotent engine — the per-bank lock makes repeats free while it builds.
  if (
    cfg.autoSeed !== false &&
    client.knowledgePagesSupported !== false &&
    output.pages.length === 0
  ) {
    startBackgroundSeed(cwd, { limit: cfg.seedLimit, harness: spec.harness });
  }
  if (output.context || output.notice) out(output.context, output.notice);
}
