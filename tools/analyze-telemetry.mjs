#!/usr/bin/env node
/**
 * MLAF Reliability Program — Iteration 2 telemetry analyzer.
 *
 * Input: a Replay-log JSON exported from the in-app "⬇ Replay log" button
 *        (an array of per-frame decision traces).
 * Output: the full measured report — frame-share, time-in-state percentiles,
 *         lock statistics, reset rate, sentence funnel, and loop forensics.
 *
 * Usage:  node tools/analyze-telemetry.mjs <path-to-mlaf-telemetry.json>
 *
 * EVERY number printed is computed from the frames. Nothing is assumed.
 */
import { readFileSync } from 'node:fs';

const path = process.argv[2];
if (!path) { console.error('usage: node analyze-telemetry.mjs <log.json>'); process.exit(1); }

const frames = JSON.parse(readFileSync(path, 'utf8'));
if (!Array.isArray(frames) || frames.length === 0) { console.error('empty / not an array'); process.exit(1); }

const N = frames.length;
const pct = (n) => `${((n / N) * 100).toFixed(1)}%`;
const quant = (arr, q) => {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))];
};
const stats = (arr) => arr.length ? {
  avg: +(arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(1),
  median: quant(arr, 0.5), p95: quant(arr, 0.95), max: Math.max(...arr), n: arr.length,
} : { avg: 0, median: 0, p95: 0, max: 0, n: 0 };

// ── frame deltas (ms) for time accounting ──
const dt = [];
for (let i = 1; i < N; i++) dt.push(Math.max(0, frames[i].ts - frames[i - 1].ts));
const medDt = quant(dt, 0.5) || 33;
const fps = medDt ? +(1000 / medDt).toFixed(1) : 0;

// ── TASK 1: frame-share by decision ──
const byDecision = {};
for (const f of frames) byDecision[f.decision || '∅'] = (byDecision[f.decision || '∅'] || 0) + 1;
const ranked = Object.entries(byDecision).sort((a, b) => b[1] - a[1]);

// ── TASK 3: time-in-state via consecutive-decision episodes ──
const episodes = []; // {decision, frames, ms}
let cur = null;
for (let i = 0; i < N; i++) {
  const d = frames[i].decision;
  const step = i > 0 ? Math.max(0, frames[i].ts - frames[i - 1].ts) : medDt;
  if (!cur || cur.decision !== d) { cur = { decision: d, frames: 0, ms: 0 }; episodes.push(cur); }
  cur.frames++; cur.ms += step;
}
const epByDecision = {};
for (const e of episodes) (epByDecision[e.decision] ||= []).push(e.ms);

// ── TASK 4/5: lock + reset analysis ──
// A "confirming run" = contiguous CONFIRMING/DETECTING frames. Success if the
// next frame is LOCKED; reset if it ends in suppression/no-progress/idle.
let lockEvents = 0, confirmRuns = 0, lockedRuns = 0, resetRuns = 0;
const timeToLock = [];
let runStart = -1, runMs = 0;
const isConfirm = (d) => d === 'CONFIRMING';
for (let i = 0; i < N; i++) {
  const d = frames[i].decision;
  const step = i > 0 ? Math.max(0, frames[i].ts - frames[i - 1].ts) : medDt;
  if (isConfirm(d)) {
    if (runStart < 0) { runStart = i; runMs = 0; confirmRuns++; }
    runMs += step;
  } else {
    if (runStart >= 0) {
      if (d === 'LOCKED') { lockedRuns++; timeToLock.push(runMs); }
      else resetRuns++;
      runStart = -1;
    }
    if (d === 'LOCKED') lockEvents++;
  }
}
if (runStart >= 0) resetRuns++; // ended mid-confirm
const suppressedFrames = byDecision['DETECTION_SUPPRESSED'] || 0;
const noProgressFrames = byDecision['NO_PROGRESS'] || 0;

// ── TASK 6: sentence funnel from sentenceBuffer length ──
let maxBuf = 0; for (const f of frames) if (Array.isArray(f.sentenceBuffer)) maxBuf = Math.max(maxBuf, f.sentenceBuffer.length);

// ── TASK 7: loop forensics — most common 3-decision n-grams ──
const trigram = {};
for (let i = 2; i < N; i++) {
  const k = `${frames[i - 2].decision} → ${frames[i - 1].decision} → ${frames[i].decision}`;
  trigram[k] = (trigram[k] || 0) + 1;
}
const topLoops = Object.entries(trigram).sort((a, b) => b[1] - a[1]).slice(0, 6);

// ════════════════════════ REPORT ════════════════════════
console.log(`\n══ MLAF TELEMETRY ANALYSIS ══  ${path.split('/').pop()}`);
console.log(`frames=${N}  duration=${((frames[N-1].ts-frames[0].ts)/1000).toFixed(1)}s  ~${fps}fps (median Δ ${medDt}ms)\n`);

console.log('TASK 1 — FRAME-SHARE BY DECISION (ranked):');
for (const [d, c] of ranked) console.log(`  ${String(d).padEnd(22)} ${pct(c).padStart(7)}  (${c})`);

console.log('\nTASK 3 — TIME-IN-STATE per episode (ms): avg / median / p95 / max  (#episodes)');
for (const [d, arr] of Object.entries(epByDecision).sort((a,b)=>b[1].reduce((x,y)=>x+y,0)-a[1].reduce((x,y)=>x+y,0))) {
  const s = stats(arr); console.log(`  ${d.padEnd(22)} ${s.avg} / ${s.median} / ${s.p95} / ${s.max}  (${s.n})`);
}

console.log('\nTASK 4 — LOCK PIPELINE:');
console.log(`  confirming runs       : ${confirmRuns}`);
console.log(`  → locked              : ${lockedRuns}  (success ${confirmRuns ? ((lockedRuns/confirmRuns)*100).toFixed(1) : 0}%)`);
console.log(`  → reset (no lock)     : ${resetRuns}  (failure ${confirmRuns ? ((resetRuns/confirmRuns)*100).toFixed(1) : 0}%)`);
console.log(`  lock events (frames)  : ${lockEvents}`);
console.log(`  time-to-lock ms       : avg ${stats(timeToLock).avg} / median ${stats(timeToLock).median} / p95 ${stats(timeToLock).p95} / max ${stats(timeToLock).max}`);
console.log(`  attempts per success  : ${lockedRuns ? (confirmRuns/lockedRuns).toFixed(1) : '∞ (zero locks)'}`);

console.log('\nTASK 5 — RESET / REPEAT EVIDENCE:');
console.log(`  confirm-run reset rate     : ${confirmRuns ? ((resetRuns/confirmRuns)*100).toFixed(1) : 0}%`);
console.log(`  DETECTION_SUPPRESSED frames : ${suppressedFrames}  (${pct(suppressedFrames)})`);
console.log(`  NO_PROGRESS frames          : ${noProgressFrames}  (${pct(noProgressFrames)})`);

console.log('\nTASK 6 — SENTENCE FUNNEL:');
console.log(`  frames reaching CONFIRMING : ${pct(byDecision['CONFIRMING']||0)}`);
console.log(`  lock events                : ${lockEvents}`);
console.log(`  max words committed        : ${maxBuf}`);

console.log('\nTASK 7 — TOP DECISION SEQUENCES (loop forensics):');
for (const [k, c] of topLoops) console.log(`  ${c.toString().padStart(4)}×  ${k}`);

console.log('\n── DOMINANT BOTTLENECK (by frame-share) ──');
console.log(`  ${ranked[0][0]} = ${pct(ranked[0][1])} of all frames\n`);
