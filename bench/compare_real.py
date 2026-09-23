"""Compare diarization configs on real audio without ground truth.

For each file: detected speakers, talk-time share of the top speakers, the
disagreement between two configs (DER of B measured against A, optimal
speaker mapping), and a few sample turns around speaker changes for manual
checking.

    python bench/compare_real.py --a cohere+pyannote --b cohere+nemotron
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyannote.metrics.diarization import DiarizationErrorRate

from run_bench import hypothesis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", type=Path, default=Path("bench/responses"))
    ap.add_argument("--files", default="presseclub,lanz,maischberger,lagedernation,hartaberfair")
    ap.add_argument("--a", default="cohere+pyannote")
    ap.add_argument("--b", default="cohere+nemotron")
    ap.add_argument("--samples", type=int, default=0, help="print N speaker changes per file")
    args = ap.parse_args()
    metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    for name in args.files.split(","):
        ra = json.loads((args.responses / f"{name}__{args.a}.json").read_text(encoding="utf-8"))
        rb = json.loads((args.responses / f"{name}__{args.b}.json").read_text(encoding="utf-8"))
        ha, hb = hypothesis(ra), hypothesis(rb)
        c = metric(ha, hb, detailed=True)
        share = lambda h: [round(100 * d / max(h.get_timeline().duration(), 1e-9))  # noqa: E731
                           for _, d in h.chart()[:6]]
        print(f"{name:14} {args.a}: {len(ha.labels())} spk {share(ha)}  "
              f"{args.b}: {len(hb.labels())} spk {share(hb)}  "
              f"disagreement {100 * c['confusion'] / c['total']:.1f}% confusion")
        if args.samples:
            for label, resp in ((args.a, ra), (args.b, rb)):
                segs = resp["segments"]
                step = max(1, len(segs) // args.samples)
                print(f"  -- {label}")
                for s in segs[::step][: args.samples]:
                    print(f"  {s['start']:7.1f} {s.get('speaker')}: {s['text'][:90]}")


if __name__ == "__main__":
    main()
