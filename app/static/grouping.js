/* Auto-grouping for the upload queue: files that look like consecutive parts
 * of one recording become one group (= one transcript). Everything else stays
 * a group of its own. The user can merge/split afterwards, so this only has
 * to guess the obvious cases right:
 *   - OBS-style timestamp names ("2026-10-20 10-00-01.mkv"): the next part
 *     starts shortly after the previous file was last written (= its end).
 *   - explicit part markers ("meeting_teil1.m4a", "call part 2.mp3").
 * Plain numbering ("notiz_01", "notiz_02") is NOT grouped — that is the
 * typical "20 separate recordings" case.
 */
"use strict";

const TS_RE = /(\d{4})-(\d{2})-(\d{2})[ _T](\d{2})[-_.:](\d{2})[-_.:](\d{2})/;
const PART_RE = /^(.*?)[\s_.-]*(?:part|teil|pt|abschnitt|segment)[\s_.-]*(\d{1,3})$/i;
const CONT_MAX_GAP_MS = 15 * 60 * 1000; // pause between stop and restart
const CONT_SLACK_MS = 2 * 60 * 1000;    // clock jitter: next start slightly before prev mtime
const PART_MAX_GAP_MS = 12 * 60 * 60 * 1000; // parts of one recording land within a session, not days apart

function nameStartMs(name) {
  const m = name.match(TS_RE);
  if (!m) return null;
  return new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]).getTime();
}

function partKey(name) {
  const dot = name.lastIndexOf(".");
  const base = dot > 0 ? name.slice(0, dot) : name;
  const ext = dot > 0 ? name.slice(dot).toLowerCase() : "";
  const m = base.match(PART_RE);
  return m ? { key: m[1].toLowerCase() + "|" + ext, n: +m[2] } : null;
}

function isContinuation(prev, next) {
  const a = nameStartMs(prev.name);
  const b = nameStartMs(next.name);
  if (a !== null && b !== null) {
    if (!prev.lastModified || b <= a) return false;
    const gap = b - prev.lastModified;
    return gap >= -CONT_SLACK_MS && gap <= CONT_MAX_GAP_MS;
  }
  const pa = partKey(prev.name);
  const pb = partKey(next.name);
  if (!pa || !pb || pa.key !== pb.key || pb.n !== pa.n + 1) return false;
  // Plausibility guard: unlike the timestamp branch there is no name-derived
  // start time, so consecutive part numbers alone would merge files that merely
  // share a base name (e.g. call_part1.mp3 on Monday, call_part2.mp3 on
  // Wednesday). Bound continuation by file mtime: real parts are written within
  // one session. When mtime is unavailable, do not auto-group (user can merge).
  if (prev.lastModified == null || next.lastModified == null) return false;
  const gap = next.lastModified - prev.lastModified;
  return gap >= -CONT_SLACK_MS && gap <= PART_MAX_GAP_MS;
}

/* files: [{name, lastModified}] → array of groups (arrays), natural name order. */
function autoGroup(files) {
  const sorted = [...files].sort((x, y) =>
    x.name.localeCompare(y.name, undefined, { numeric: true, sensitivity: "base" }));
  const groups = [];
  for (const f of sorted) {
    const last = groups[groups.length - 1];
    if (last && isContinuation(last[last.length - 1], f)) last.push(f);
    else groups.push([f]);
  }
  return groups;
}

if (typeof module !== "undefined") module.exports = { autoGroup };
