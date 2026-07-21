"""Track in-flight transcription jobs for queue display in the UI.

Keeps a realtime-factor EMA (audio seconds transcribed per wall second) so
clients can estimate how long the queue ahead of them will take.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Job:
    duration: float
    started: float


@dataclass
class JobTracker:
    _jobs: dict[int, _Job] = field(default_factory=dict)
    _ids: itertools.count = field(default_factory=itertools.count)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    # Audio seconds per wall second; None until the first job finishes.
    rtf_ema: float | None = None

    def enter(self, duration: float) -> int:
        with self._lock:
            job_id = next(self._ids)
            self._jobs[job_id] = _Job(duration=duration, started=time.monotonic())
            return job_id

    def exit(self, job_id: int) -> None:
        with self._lock:
            job = self._jobs.pop(job_id, None)
            if job is None:
                return
            wall = time.monotonic() - job.started
            if wall <= 0 or job.duration <= 0:
                return
            rtf = job.duration / wall
            self.rtf_ema = rtf if self.rtf_ema is None else 0.7 * self.rtf_ema + 0.3 * rtf

    def snapshot(self) -> dict:
        """Queue state for /health: active jobs + ETA estimate in seconds."""
        with self._lock:
            now = time.monotonic()
            jobs = [
                {
                    "duration": round(j.duration, 1),
                    "elapsed": round(now - j.started, 1),
                }
                for j in self._jobs.values()
            ]
        rtf = self.rtf_ema or 10.0  # conservative default before first measurement
        remaining = 0.0
        for j in jobs:
            done_audio = min(j["duration"], j["elapsed"] * rtf)
            remaining += max(0.0, j["duration"] - done_audio)
        return {
            "active_jobs": len(jobs),
            "jobs": jobs,
            "rtf": round(rtf, 2) if self.rtf_ema else None,
            "queue_eta_seconds": round(remaining / rtf, 1) if jobs else 0.0,
        }


tracker = JobTracker()
