"""Benchmark diarization pipelines against the API.

For every audio file in --data and every config, calls /v1/audio/transcriptions
(verbose_json, speaker_attribution) and reports wall time, detected speaker
count and — where a <name>.rttm reference exists — DER of the final
speaker-labelled words (collar 0.25 s, overlap scored).

    uv run python bench/run_bench.py --api http://192.168.107.103:8013 \
        --data bench/data --out bench/results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import httpx
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

COHERE = {"model": "cohere-transcribe"}
PLUS = {"model": "granite-speech-4.1-2b-plus"}
CONFIGS = {
    "cohere+pyannote": {**COHERE, "diarization_engine": "pyannote"},
    "cohere+nemotron": {**COHERE, "diarization_engine": "nemotron"},
    "plus+granite-saa": {**PLUS, "diarization_engine": "granite"},
    "plus+nemotron": {**PLUS, "diarization_engine": "nemotron"},
}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".opus", ".ogg", ".flac"}


def load_rttm(path: Path) -> Annotation:
    ann = Annotation()
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            start, dur = float(p[3]), float(p[4])
            ann[Segment(start, start + dur)] = p[7]
    return ann


def hypothesis(resp: dict) -> Annotation:
    """Speaker annotation from word spans (tighter than segment spans)."""
    ann = Annotation()
    for seg in resp.get("segments", []):
        spk = seg.get("speaker") or "?"
        words = seg.get("words") or []
        spans = [(w["start"], w["end"]) for w in words] or [(seg["start"], seg["end"])]
        for s, e in spans:
            if e > s:
                ann[Segment(s, e)] = spk
    return ann.support(collar=0.3)  # merge word gaps within a speaker run


def run(api: str, audio: Path, fields: dict, num_speakers: int | None) -> tuple[dict, float]:
    data = {
        "language": "de",
        "response_format": "verbose_json",
        "speaker_attribution": "true",
        "timestamp_granularities[]": "word",
        **fields,
    }
    if num_speakers:
        data["num_speakers"] = str(num_speakers)
    t0 = time.perf_counter()
    r = httpx.post(
        f"{api}/v1/audio/transcriptions",
        files={"file": (audio.name, audio.read_bytes())},
        data=data,
        timeout=3600,
    )
    elapsed = time.perf_counter() - t0
    try:
        body = r.json()
    except json.JSONDecodeError:
        body = {"detail": r.text[:200]}
    if r.status_code != 200:
        body = {"error": body.get("detail", r.status_code)}
    return body, elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://192.168.107.103:8013")
    ap.add_argument("--data", type=Path, default=Path("bench/data"))
    ap.add_argument("--out", type=Path, default=Path("bench/results.csv"))
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--known-speakers", action="store_true",
                    help="also run each config with num_speakers from the reference")
    args = ap.parse_args()

    files = sorted(p for p in args.data.iterdir() if p.suffix.lower() in AUDIO_EXT)
    metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    rows = []
    for name in args.configs.split(","):
        fields = CONFIGS[name]
        # Warm-up on the shortest file so load time doesn't skew the first result.
        run(args.api, min(files, key=lambda p: p.stat().st_size), fields, None)
        for audio in files:
            rttm = audio.with_suffix(".rttm")
            ref = load_rttm(rttm) if rttm.exists() else None
            ref_n = len(ref.labels()) if ref else None
            modes = [None, ref_n] if (args.known_speakers and ref_n) else [None]
            for n in modes:
                resp, secs = run(args.api, audio, fields, n)
                row = {
                    "file": audio.stem, "config": name, "num_speakers_given": n or "",
                    "seconds": round(secs, 1), "ref_speakers": ref_n or "",
                }
                if "error" in resp:
                    row["error"] = str(resp["error"])[:120]
                else:
                    hyp = hypothesis(resp)
                    row["duration"] = round(resp.get("duration") or 0, 1)
                    row["rtfx"] = round(row["duration"] / secs, 1) if secs else ""
                    row["hyp_speakers"] = len(hyp.labels())
                    if ref is not None:
                        c = metric(ref, hyp, detailed=True)
                        total = c["total"] or 1
                        row["der"] = round(100 * c["diarization error rate"], 2)
                        row["miss"] = round(100 * c["missed detection"] / total, 2)
                        row["fa"] = round(100 * c["false alarm"] / total, 2)
                        row["confusion"] = round(100 * c["confusion"] / total, 2)
                    out_json = args.out.parent / "responses" / f"{audio.stem}__{name}{'__n' if n else ''}.json"
                    out_json.parent.mkdir(parents=True, exist_ok=True)
                    out_json.write_text(json.dumps(resp, ensure_ascii=False), encoding="utf-8")
                rows.append(row)
                print(row, flush=True)

    keys = ["file", "config", "num_speakers_given", "ref_speakers", "hyp_speakers", "der",
            "miss", "fa", "confusion", "duration", "seconds", "rtfx", "error"]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
