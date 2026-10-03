import { mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import type { PageRef } from "./knowledge-injection";
import type { RetainCursor, RetainCursorStore } from "./retain-cursor";
import type { UsageCursorStore } from "./usage";

/** Process-shared state for hook harnesses. SessionStart and prompt hooks are separate Node
 * processes, so this temp-file handoff carries lifecycle decisions without writing user config or
 * bank state. */
export interface SessionCache {
  turns?: number;
  reflectAnswer?: string; // present (even "") = reflect already resolved this session
  /** How many times auto-inject has been ATTEMPTED this session — every source, not just reflect.
   *  A failure leaves `reflectAnswer` unset so a later turn can retry; this bounds that retry (see
   *  HOOK_INJECT_ATTEMPTS). Named for reflect like `reflectAnswer` above, whose name also predates
   *  the other two sources. */
  reflectAttempts?: number;
  /** SessionStart saw a new/empty bank; consume this on prompt one, then allow reflect. */
  deferInitialReflect?: boolean;
  /** The first-prompt synthesis outlived its hook: a background worker is still writing it (or has
   *  finished since) and a later prompt delivers it. See `ReflectJob`. */
  reflectPending?: boolean;
  pages?: { atTurn: number; list: PageRef[] };
}

/**
 * The first-prompt synthesis, run by a worker process that outlives the hook that started it.
 *
 * A hook has to exit for the user's turn to begin, and exiting used to abort the reflect request —
 * so a synthesis one second slower than `reflectTimeoutMs` was computed by the server and thrown
 * away. The worker keeps the request alive instead and leaves the answer here; the hook reads it
 * if it lands inside the timeout, and otherwise a later prompt's hook picks it up.
 *
 * Its own file, like the retain cursor and for the same reason: the prompt hook rewrites the
 * session cache as a fresh object, and two writers (hook and worker) must not share a record.
 */
export interface ReflectJob {
  state: "pending" | "ready" | "failed";
  /** Epoch ms the hook started the job — a pending job older than the worker can live is dead. */
  startedAt: number;
  /** What the worker needs to make the call. No token: the worker reloads config for that. */
  request?: { harness: string; apiUrl: string; bank: string; query: string; budget: string };
  answer?: string;
  ms?: number;
  error?: string;
  status?: number;
  timedOut?: boolean;
}

export function reflectJobFile(cacheFile: string): string {
  return `${cacheFile}.reflect.json`;
}

export function readReflectJob(jobFile: string): ReflectJob | undefined {
  try {
    return JSON.parse(readFileSync(jobFile, "utf8")) as ReflectJob;
  } catch {
    return undefined;
  }
}

/** Throws on failure: a job that cannot be recorded must not be started (the hook falls back to
 *  reflecting in-process). */
export function writeReflectJob(jobFile: string, job: ReflectJob): void {
  writeFileAtomic(jobFile, JSON.stringify(job));
}

export function clearReflectJob(jobFile: string): void {
  try {
    rmSync(jobFile, { force: true });
  } catch {
    /* best-effort: a leftover job file is a few bytes in the temp dir */
  }
}

export function sessionCacheFile(harness: string, sessionId: string): string {
  return join(tmpdir(), `hindsight-${harness}`, `${sessionId}.json`);
}

export function readSessionCache(cacheFile: string): SessionCache {
  try {
    return JSON.parse(readFileSync(cacheFile, "utf8")) as SessionCache;
  } catch {
    return {};
  }
}

/**
 * Replace a state file atomically: write a sibling temp file, then rename over the target.
 *
 * A plain writeFileSync is not atomic — a reader can observe a half-written file, and a process
 * killed mid-write leaves one behind. rename() is atomic on POSIX and replaces the destination on
 * Windows, so a reader sees either the old contents or the new, never a fragment (#3136, which
 * measured this class on the per-agent plugin; its Python state writes already had os.replace()).
 */
function writeFileAtomic(path: string, body: string): void {
  mkdirSync(dirname(path), { recursive: true });
  // The temp name is per-process, so two writers cannot collide on it.
  const tmp = `${path}.${process.pid}.tmp`;
  try {
    writeFileSync(tmp, body);
    renameSync(tmp, path);
  } catch (e) {
    try {
      rmSync(tmp, { force: true });
    } catch {
      /* nothing further to do */
    }
    throw e;
  }
}

export function writeSessionCache(cacheFile: string, cache: SessionCache): void {
  try {
    writeFileAtomic(cacheFile, JSON.stringify(cache));
  } catch {
    /* session state is best-effort */
  }
}

/** The session root's own file, deliberately NOT the shared session cache — see sessionRootDir. */
function sessionRootFile(harness: string, sessionId: string): string {
  return join(tmpdir(), `hindsight-${harness}`, `${sessionId}.root`);
}

/**
 * The directory this session STARTED in: recorded the first time any of its hooks runs — normally
 * SessionStart — and returned unchanged for the rest of the session.
 *
 * Bank resolution needs a directory that does not move. A hook event reports the agent's LIVE
 * working directory, and an agent `cd`s during ordinary work. Inside a git repo that is harmless
 * (every subdirectory resolves back to the repo root), but a plain directory tree has no root to
 * resolve to, so the bank id followed the agent and ONE conversation was retained into a bank per
 * directory it visited — the same document, its facts split across banks (#3563).
 *
 * Deliberately keyed on the session id rather than read from a harness-exported project-root
 * variable: every hook harness reports a session id, only Claude Code exports a root. Same reason
 * `nearestExistingDir` walks the tree instead of guessing at nine env var names.
 *
 * It lives in its OWN file, like the retain cursor and for the same reason: the prompt hook writes
 * a fresh session-cache object rather than merging, so a field there would be dropped on every user
 * prompt — and the very first prompt is where it would first be needed.
 *
 * Best-effort throughout. A temp file that cannot be read or written just yields `cwd`, which is
 * exactly the behaviour before this existed — never an error, never a lost retain.
 */
export function sessionRootDir(
  harness: string,
  sessionId: string | undefined,
  cwd: string
): string {
  if (!sessionId || !cwd) return cwd;
  const file = sessionRootFile(harness, sessionId);
  try {
    const recorded = readFileSync(file, "utf8").trim();
    if (recorded) return recorded;
  } catch {
    /* not recorded yet — this hook is the session's first */
  }
  try {
    writeFileAtomic(file, cwd);
  } catch {
    /* best-effort: an unrecorded root costs stability, never data */
  }
  return cwd;
}

/**
 * How many of a session's turns core/usage.ts has recorded. Its own file for the same reason as
 * the retain cursor below; losing it re-records the session's turns, which the report dedupes.
 */
export function fileUsageCursorStore(harness: string): UsageCursorStore {
  const file = (sessionId: string) =>
    join(tmpdir(), `hindsight-${harness}`, `${sessionId}.usage.json`);
  return {
    read: (sessionId) => {
      try {
        const turns = (JSON.parse(readFileSync(file(sessionId), "utf8")) as { turns?: unknown })
          .turns;
        return typeof turns === "number" ? turns : undefined;
      } catch {
        return undefined;
      }
    },
    write: (sessionId, turns) => {
      try {
        writeFileAtomic(file(sessionId), JSON.stringify({ turns }));
      } catch {
        /* best-effort */
      }
    },
  };
}

/** The cursor's own file, deliberately NOT the shared session cache — see fileCursorStore. */
function cursorFile(harness: string, sessionId: string): string {
  return join(tmpdir(), `hindsight-${harness}`, `${sessionId}.retain.json`);
}

/**
 * Retain cursor for the hook harnesses: Stop runs in a fresh process every time, so "what have I
 * already written" cannot live in memory.
 *
 * It lives in its OWN file rather than a field of the session cache. It shared that file until now,
 * and the prompt hook — which writes a fresh `{turns, reflectAnswer, pages}` object rather than
 * merging — dropped the cursor on EVERY user prompt. The next Stop then found none and rewrote the
 * whole document, so the incremental write-back never engaged past a session's first turn on any
 * hook harness. Separate files make that structural: the two writers have different lifecycles and
 * no longer share a record, so neither can clobber the other, and the concurrent read-modify-write
 * that #3136 measured has nothing left to lose here.
 *
 * Losing the file (temp cleanup, reboot) is still not a correctness problem — a missing cursor
 * means the next retain replaces the whole document, exactly as it did before appends existed.
 */
export function fileCursorStore(harness: string): RetainCursorStore {
  return {
    read: (sessionId) => {
      try {
        return JSON.parse(readFileSync(cursorFile(harness, sessionId), "utf8")) as RetainCursor;
      } catch {
        return undefined;
      }
    },
    write: (sessionId, cursor) => {
      try {
        writeFileAtomic(cursorFile(harness, sessionId), JSON.stringify(cursor));
      } catch {
        /* best-effort: a cursor that cannot be written costs a replace, never data */
      }
    },
  };
}
