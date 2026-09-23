"""Synthetic German multi-speaker conversations with exact ground truth.

Uses the internal supertonic TTS (10 voices, F1-F5 / M1-M5) to render meeting
style turns, stitches them with natural gaps, crosstalk overlaps and
backchannels, and writes <name>.wav (16 kHz mono) + <name>.rttm.

    uv run python bench/make_synth.py --out bench/data
"""

from __future__ import annotations

import argparse
import hashlib
import io
import random
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

TTS_URL = "http://192.168.107.103:8012/v1/audio/speech"
SR = 16000
VOICES = ["M3", "F2", "M1", "F4", "M5", "F1", "M2", "F3", "M4", "F5"]

LINES = [
    "Ich würde vorschlagen, dass wir zuerst über das Budget für das nächste Quartal sprechen.",
    "Das sehe ich ähnlich, aber die Zahlen aus dem Vertrieb liegen noch nicht vollständig vor.",
    "Wann genau können wir mit den finalen Zahlen rechnen?",
    "Voraussichtlich Ende nächster Woche, wenn die Abrechnung aus Leipzig da ist.",
    "Dann sollten wir den Termin für die Freigabe entsprechend verschieben.",
    "Moment, wir haben dem Kunden aber schon den fünfzehnten zugesagt.",
    "Das ist richtig, allerdings war das unter Vorbehalt.",
    "Ich kann gerne noch einmal mit dem Projektleiter telefonieren und nachfragen.",
    "Die VR-Schulung für die Instandhaltung läuft übrigens sehr gut, die Rückmeldungen sind positiv.",
    "Wie viele Teilnehmer hatten wir denn in der letzten Runde?",
    "Insgesamt waren es zweiundvierzig, davon zwölf aus der Frühschicht.",
    "Das ist deutlich mehr als erwartet, damit hatte ich nicht gerechnet.",
    "Wir brauchen dafür aber mehr Headsets, die aktuellen reichen nicht aus.",
    "Können wir die nicht einfach über den Rahmenvertrag bestellen?",
    "Theoretisch ja, praktisch dauert die Lieferung gerade sechs bis acht Wochen.",
    "Dann sollten wir die Bestellung am besten heute noch auslösen.",
    "Einverstanden. Wer kümmert sich darum?",
    "Das übernehme ich, ich schicke euch nachher die Bestellnummer.",
    "Noch ein anderes Thema: Der Server im Keller macht seit Montag komische Geräusche.",
    "Ja, das habe ich auch gehört. Ich glaube, das ist der Lüfter vom Netzteil.",
    "Haben wir dafür noch ein Ersatzteil auf Lager?",
    "Ich schaue gleich nach, aber ich meine, wir hatten noch eins.",
    "Bitte nicht vergessen, vorher ein Backup zu machen.",
    "Selbstverständlich, das Backup läuft ohnehin jede Nacht automatisch.",
    "Und wie weit sind wir mit der neuen Szene für den Brandschutz?",
    "Die Modelle sind fertig, es fehlen nur noch die Animationen für den Feuerlöscher.",
    "Sieht das auf der Brille flüssig aus oder ruckelt es noch?",
    "Auf der Quest läuft es mit zweiundsiebzig Bildern pro Sekunde, das passt.",
    "Super, dann können wir das nächste Woche beim Kunden vorführen.",
    "Ich würde trotzdem gerne vorher einen internen Testlauf machen.",
    "Gute Idee, sagen wir Donnerstag um zehn Uhr?",
    "Donnerstag passt mir leider nicht, da bin ich in Dresden.",
    "Dann eben Freitag früh, ginge das für alle?",
    "Für mich ist Freitag in Ordnung.",
    "Ich trage es direkt in den Kalender ein und schicke eine Einladung.",
    "Wer schreibt eigentlich heute das Protokoll?",
    "Das mache ich, ich habe ohnehin schon mitgeschrieben.",
    "Dann halten wir fest: Budget nächste Woche, Headsets heute bestellen, Testlauf Freitag.",
    "Habe ich noch etwas vergessen?",
    "Nur die Frage, ob wir die Schulung auch auf Englisch anbieten wollen.",
    "Das sollten wir mit der Geschäftsführung besprechen, bevor wir etwas versprechen.",
    "Die Übersetzung selbst wäre kein großes Problem, die Sprachausgabe schon eher.",
    "Wir könnten die Sprecherstimmen ja synthetisch erzeugen, das geht inzwischen ziemlich gut.",
    "Hauptsache, die Aussprache der Fachbegriffe stimmt.",
]
BACKCHANNELS = ["Ja.", "Genau.", "Mhm, stimmt.", "Richtig.", "Okay.", "Ach so.", "Ja, klar."]


def tts(text: str, voice: str, cache: Path) -> np.ndarray:
    key = hashlib.sha1(f"{voice}|{text}".encode()).hexdigest()[:16]
    f = cache / f"{key}.wav"
    if not f.exists():
        r = httpx.post(
            TTS_URL,
            json={"input": text, "voice": voice, "response_format": "wav", "lang": "de",
                  "speed": 1.05},
            timeout=120,
        )
        r.raise_for_status()
        f.write_bytes(r.content)
    audio, sr = sf.read(io.BytesIO(f.read_bytes()), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    # Trim TTS leading/trailing silence so RTTM boundaries are tight.
    idx = np.flatnonzero(np.abs(audio) > 0.01)
    return audio[idx[0]: idx[-1] + 1] if idx.size else audio


def render(name: str, n_speakers: int, n_turns: int, overlap_p: float, seed: int,
           out: Path, cache: Path) -> None:
    rng = random.Random(seed)
    voices = VOICES[:n_speakers]
    # Zipf-ish talk share: a few speakers dominate, like real meetings.
    weights = [1.0 / (i + 1) ** 0.7 for i in range(n_speakers)]
    events: list[tuple[float, str, np.ndarray]] = []  # (start, speaker, audio)
    t, prev = 0.5, None
    lines = LINES[:]
    rng.shuffle(lines)
    for i in range(n_turns):
        # Everybody speaks at least once early on.
        spk = voices[i] if i < n_speakers else rng.choices(voices, weights)[0]
        if spk == prev:
            spk = rng.choice([v for v in voices if v != prev])
        audio = tts(lines[i % len(lines)], spk, cache)
        if prev is not None and rng.random() < overlap_p:
            gap = -rng.uniform(0.3, 1.2)  # crosstalk: starts before the other ends
        else:
            gap = rng.uniform(0.15, 0.9)
        start = max(0.0, t + gap)
        events.append((start, spk, audio))
        t = start + len(audio) / SR
        # Occasional backchannel from a third person during this turn.
        if n_speakers > 2 and rng.random() < 0.2:
            other = rng.choice([v for v in voices if v != spk])
            bc = tts(rng.choice(BACKCHANNELS), other, cache)
            bstart = start + rng.uniform(0.3, 0.7) * len(audio) / SR
            events.append((bstart, other, bc))
        prev = spk

    total = max(s + len(a) / SR for s, _, a in events) + 0.5
    mix = np.zeros(int(total * SR), dtype=np.float32)
    rttm = []
    for s, spk, a in events:
        i0 = int(s * SR)
        mix[i0: i0 + len(a)] += a
        rttm.append(
            f"SPEAKER {name} 1 {s:.3f} {len(a) / SR:.3f} <NA> <NA> {spk} <NA> <NA>"
        )
    # Light background noise so VAD has something to reject.
    mix += np.random.default_rng(seed).normal(0, 0.003, mix.shape).astype(np.float32)
    mix /= max(1.0, float(np.abs(mix).max()) / 0.95)
    sf.write(out / f"{name}.wav", mix, SR)
    (out / f"{name}.rttm").write_text("\n".join(rttm) + "\n", encoding="utf-8")
    print(f"{name}: {total / 60:.1f} min, {n_speakers} speakers, {len(events)} turns")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("bench/data"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / ".tts-cache"
    cache.mkdir(exist_ok=True)
    for n, turns, ov in [(2, 40, 0.15), (4, 60, 0.15), (6, 70, 0.2), (8, 80, 0.2), (10, 90, 0.2)]:
        render(f"synth-{n:02d}spk", n, turns, ov, seed=n, out=args.out, cache=cache)


if __name__ == "__main__":
    main()
