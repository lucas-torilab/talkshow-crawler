"""Speaker diarization via the pyannote.ai API (https://docs.pyannote.ai/api-reference)."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import requests

API_BASE = "https://api.pyannote.ai/v1"
API_KEY_ENV_VAR = "PYANNOTE_API_KEY"

DEFAULT_MODEL = "precision-2"
MODELS = ("precision-2", "community-1")

DEFAULT_POLL_INTERVAL = 5.0
DEFAULT_TIMEOUT = 30 * 60.0  # seconds

_TERMINAL_STATUSES = {"succeeded", "failed", "canceled"}


class DiarizationError(Exception):
    """Raised for any user-facing pyannote.ai diarization failure."""


def _headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _request(method: str, url: str, **kwargs) -> requests.Response:
    try:
        return requests.request(method, url, **kwargs)
    except requests.RequestException as exc:
        raise DiarizationError(f"Network error calling pyannote.ai ({method} {url}): {exc}") from exc


def upload_media(audio_path: Path, api_key: str) -> str:
    """Upload a local audio file to pyannote.ai's temporary storage.

    Returns the `media://...` url to pass as the diarize job's `url` field.
    """
    object_key = f"talkshow-crawler/{uuid.uuid4().hex}{audio_path.suffix}"
    media_url = f"media://{object_key}"

    resp = _request(
        "POST", f"{API_BASE}/media/input", headers=_headers(api_key), json={"url": media_url}, timeout=30
    )
    if resp.status_code != 201:
        raise DiarizationError(f"Failed to create upload URL ({resp.status_code}): {resp.text}")
    presigned_url = resp.json()["url"]

    with audio_path.open("rb") as f:
        put_resp = _request("PUT", presigned_url, data=f, timeout=600)
    if not put_resp.ok:
        raise DiarizationError(f"Failed to upload audio ({put_resp.status_code}): {put_resp.text}")

    return media_url


def submit_diarization_job(
    media_url: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> str:
    """Submit a diarize job for a previously-uploaded media:// url. Returns the jobId."""
    if model not in MODELS:
        raise DiarizationError(f"Unsupported model: {model!r} (choose one of {MODELS})")

    payload: dict = {"url": media_url, "model": model}
    if num_speakers is not None:
        payload["numSpeakers"] = num_speakers
    if min_speakers is not None:
        payload["minSpeakers"] = min_speakers
    if max_speakers is not None:
        payload["maxSpeakers"] = max_speakers

    resp = _request("POST", f"{API_BASE}/diarize", headers=_headers(api_key), json=payload, timeout=30)
    if resp.status_code != 200:
        raise DiarizationError(f"Failed to submit diarization job ({resp.status_code}): {resp.text}")
    return resp.json()["jobId"]


def get_job(job_id: str, api_key: str) -> dict:
    resp = _request("GET", f"{API_BASE}/jobs/{job_id}", headers=_headers(api_key), timeout=30)
    if resp.status_code != 200:
        raise DiarizationError(f"Failed to fetch job {job_id} ({resp.status_code}): {resp.text}")
    return resp.json()


def wait_for_job(
    job_id: str,
    api_key: str,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    timeout: float = DEFAULT_TIMEOUT,
    on_poll: Optional[Callable[[str], None]] = None,
) -> dict:
    """Poll a job until it reaches a terminal status; returns the final job dict."""
    start = time.monotonic()
    while True:
        job = get_job(job_id, api_key)
        status = job.get("status", "unknown")
        if on_poll:
            on_poll(status)

        if status == "succeeded":
            return job
        if status in _TERMINAL_STATUSES:
            error = (job.get("output") or {}).get("error")
            raise DiarizationError(f"Diarization job {job_id} ended with status {status!r}: {error or job}")
        if time.monotonic() - start > timeout:
            raise DiarizationError(
                f"Timed out after {timeout:.0f}s waiting for diarization job {job_id} "
                f"(last status: {status!r})"
            )
        time.sleep(poll_interval)


def diarize_audio(
    audio_path: Path,
    api_key: str,
    model: str = DEFAULT_MODEL,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    timeout: float = DEFAULT_TIMEOUT,
    on_stage: Optional[Callable[[str], None]] = None,
    on_poll: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """Run the full pipeline (upload -> submit -> poll) for a local audio file.

    Returns the list of diarization segments: [{"speaker", "start", "end", ...}, ...].
    """
    if on_stage:
        on_stage("Uploading audio")
    media_url = upload_media(audio_path, api_key)

    if on_stage:
        on_stage("Submitting diarization job")
    job_id = submit_diarization_job(
        media_url,
        api_key,
        model=model,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    if on_stage:
        on_stage("Waiting for diarization")
    job = wait_for_job(job_id, api_key, poll_interval=poll_interval, timeout=timeout, on_poll=on_poll)

    output = job.get("output") or {}
    diarization = output.get("diarization")
    if diarization is None:
        raise DiarizationError(f"Job {job_id} succeeded but returned no diarization output: {output}")
    return diarization
