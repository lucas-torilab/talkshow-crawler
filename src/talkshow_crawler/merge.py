"""Merge a transcript (text/start/duration) with a diarization (speaker/start/end).

Both come out of this project in plain seconds-based JSON, so aligning them is a
numeric interval join: each transcript segment is tagged with whichever speaker
covers the most of its [start, start+duration] window.
"""

from __future__ import annotations

import json
from pathlib import Path

from talkshow_crawler.youtube import TranscriptError

UNKNOWN_SPEAKER = "UNKNOWN"


def _load_json_list(path: Path, what: str) -> list[dict]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TranscriptError(f"Could not read {what} from {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise TranscriptError(f"{path} does not look like a {what} file (expected a list).")
    return raw


def load_transcript_segments(path: Path) -> list[dict]:
    """Load a `download -f json` transcript: [{"text", "start", "duration"}, ...]."""
    raw = _load_json_list(path, "transcript JSON")
    try:
        return [{"text": s["text"], "start": float(s["start"]), "duration": float(s["duration"])} for s in raw]
    except (KeyError, TypeError, ValueError) as exc:
        raise TranscriptError(f"{path} is missing text/start/duration fields: {exc}") from exc


def load_diarization_segments(path: Path) -> list[dict]:
    """Load a `diarize` output: [{"speaker", "start", "end"}, ...]."""
    raw = _load_json_list(path, "diarization JSON")
    try:
        return [{"speaker": s["speaker"], "start": float(s["start"]), "end": float(s["end"])} for s in raw]
    except (KeyError, TypeError, ValueError) as exc:
        raise TranscriptError(f"{path} is missing speaker/start/end fields: {exc}") from exc


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _nearest_speaker(mid: float, diar: list[dict], i: int) -> str:
    """Fallback for a transcript segment that overlaps no diarization turn at all
    (e.g. music/silence): pick whichever turn is closest in time.
    """
    candidates = [d for d in (diar[i - 1] if i > 0 else None, diar[i] if i < len(diar) else None) if d]
    if not candidates:
        return UNKNOWN_SPEAKER

    def dist(d: dict) -> float:
        if mid < d["start"]:
            return d["start"] - mid
        if mid > d["end"]:
            return mid - d["end"]
        return 0.0

    return min(candidates, key=dist)["speaker"]


def merge_transcript_with_diarization(transcript: list[dict], diarization: list[dict]) -> list[dict]:
    """Tag each transcript segment with the diarization speaker that overlaps it most.

    Runs a two-pointer sweep over both lists sorted by start time (O(n + m) amortized)
    rather than a naive O(n * m) comparison.
    """
    diar = sorted(diarization, key=lambda d: d["start"])
    segments = sorted(transcript, key=lambda s: s["start"])
    n = len(diar)
    i = 0
    merged = []

    for seg in segments:
        start = seg["start"]
        end = start + seg["duration"]

        # Diarization turns that ended before this segment starts can never
        # matter again, since segment starts only increase from here.
        while i < n and diar[i]["end"] <= start:
            i += 1

        overlap_by_speaker: dict[str, float] = {}
        j = i
        while j < n and diar[j]["start"] < end:
            ov = _overlap(start, end, diar[j]["start"], diar[j]["end"])
            if ov > 0:
                overlap_by_speaker[diar[j]["speaker"]] = overlap_by_speaker.get(diar[j]["speaker"], 0.0) + ov
            j += 1

        if overlap_by_speaker:
            speaker = max(overlap_by_speaker, key=overlap_by_speaker.get)
        else:
            speaker = _nearest_speaker((start + end) / 2, diar, i)

        merged.append({"start": start, "end": end, "speaker": speaker, "text": seg["text"]})

    return merged


def merge_files(transcript_path: Path, diarization_path: Path) -> list[dict]:
    transcript = load_transcript_segments(transcript_path)
    diarization = load_diarization_segments(diarization_path)
    return merge_transcript_with_diarization(transcript, diarization)


def save_merged(merged: list[dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path
