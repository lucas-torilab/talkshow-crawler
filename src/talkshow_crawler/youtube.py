"""Helpers for resolving YouTube video ids and fetching/formatting transcripts."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests
import yt_dlp
from youtube_transcript_api import (
    FetchedTranscript,
    FetchedTranscriptSnippet,
    YouTubeTranscriptApi,
)
from youtube_transcript_api.formatters import (
    JSONFormatter,
    SRTFormatter,
    TextFormatter,
    WebVTTFormatter,
)
from youtube_transcript_api.proxies import GenericProxyConfig, WebshareProxyConfig

from talkshow_crawler.settings import get_settings

DEFAULT_OUTPUT_DIR = Path("outputs")

# Audio formats that require ffmpeg to re-encode into; "best" keeps yt-dlp's
# original downloaded container (no re-encode, no ffmpeg needed).
AUDIO_FORMATS = ("best", "mp3", "m4a", "opus", "wav", "flac")

# Bare 11-char YouTube video ids look like this.
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

FORMATTERS = {
    "txt": TextFormatter(),
    "srt": SRTFormatter(),
    "vtt": WebVTTFormatter(),
    "json": JSONFormatter(),
}


class TranscriptError(Exception):
    """Raised for any user-facing transcript-fetching failure."""


def _build_api() -> YouTubeTranscriptApi:
    """Build a YouTubeTranscriptApi, routed through a proxy if one is configured.

    Only the *caption-fetch* endpoint tends to get IP-blocked after high request
    volume (listing available languages is unaffected) — see README's "IP
    blocks" section. Set WEBSHARE_PROXY_USERNAME/PASSWORD (or YOUTUBE_PROXY_URL)
    in the environment or .env to route requests through a proxy; otherwise this
    behaves exactly as before (direct connection, no proxy).
    """
    settings = get_settings()
    if settings.webshare_proxy_username and settings.webshare_proxy_password:
        proxy_config = WebshareProxyConfig(
            proxy_username=settings.webshare_proxy_username,
            proxy_password=settings.webshare_proxy_password,
        )
    elif settings.youtube_proxy_url:
        proxy_config = GenericProxyConfig(
            http_url=settings.youtube_proxy_url, https_url=settings.youtube_proxy_url
        )
    else:
        proxy_config = None
    return YouTubeTranscriptApi(proxy_config=proxy_config)


def extract_video_id(value: str) -> str:
    """Accept a bare video id or a youtube.com/youtu.be URL and return the video id."""
    value = value.strip()
    if _VIDEO_ID_RE.match(value):
        return value

    parsed = urlparse(value)
    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("m.")

    if host in {"youtu.be"}:
        video_id = parsed.path.lstrip("/")
    elif host in {"youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
            video_id = parsed.path.split("/")[2] if len(parsed.path.split("/")) > 2 else ""
        else:
            video_id = ""
    else:
        video_id = ""

    if not video_id or not _VIDEO_ID_RE.match(video_id):
        raise TranscriptError(f"Could not extract a YouTube video id from: {value!r}")
    return video_id


def list_available_languages(video_id: str) -> list[dict]:
    """Return metadata about every transcript available for a video."""
    api = _build_api()
    transcript_list = api.list(video_id)
    return [
        {
            "language": t.language,
            "language_code": t.language_code,
            "is_generated": t.is_generated,
            "is_translatable": t.is_translatable,
        }
        for t in transcript_list
    ]


def fetch_transcript(
    video_id: str,
    languages: list[str],
    preserve_formatting: bool = False,
    on_stage=None,
):
    """Fetch a transcript, preferring the given languages in order.

    This is `list(video_id).find_transcript(languages).fetch(...)` broken into its
    two network stages so a caller can drive a progress bar via `on_stage(label)`,
    called right before each stage starts.
    """
    api = _build_api()
    try:
        if on_stage:
            on_stage("Resolving transcript list")
        transcript = api.list(video_id).find_transcript(languages)
        if on_stage:
            on_stage("Downloading transcript")
        return transcript.fetch(preserve_formatting=preserve_formatting)
    except Exception as exc:  # youtube_transcript_api raises its own rich exception types
        raise TranscriptError(str(exc)) from exc


def format_transcript(fetched, fmt: str) -> str:
    formatter = FORMATTERS.get(fmt)
    if formatter is None:
        raise TranscriptError(f"Unsupported format: {fmt!r} (choose one of {sorted(FORMATTERS)})")
    return formatter.format_transcript(fetched)


def save_transcript(content: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def collapse_consecutive_speakers(segments: list[dict]) -> list[dict]:
    """Merge consecutive segments sharing the same 'speaker' into a single turn.

    Each input dict needs 'start', 'text', and either 'end' or 'duration', plus
    an optional 'speaker'. Segments with no speaker (or a falsy one) are left as
    separate entries — only equal, truthy speaker labels get combined. Returns
    normalized {"start", "end", "speaker", "text"} dicts, oldest first.
    """
    collapsed: list[dict] = []
    for seg in segments:
        start = seg["start"]
        end = seg["end"] if "end" in seg else start + seg["duration"]
        speaker = seg.get("speaker")
        text = seg["text"]
        if collapsed and speaker and collapsed[-1]["speaker"] == speaker:
            collapsed[-1]["end"] = end
            collapsed[-1]["text"] = f"{collapsed[-1]['text']} {text}".strip()
        else:
            collapsed.append({"start": start, "end": end, "speaker": speaker, "text": text})
    return collapsed


def load_transcript_json(json_path: Path, collapse_speakers: bool = False) -> FetchedTranscript:
    """Load a transcript JSON back into a FetchedTranscript, for `convert` to re-render.

    Accepts two shapes:
    - plain transcript (`download -f json`): {"text", "start", "duration"}
    - speaker-tagged (`merge`'s output): {"text", "start", "end", "speaker"} — the
      speaker label is prefixed onto each line's text (e.g. "SPEAKER_00: ...") so
      it survives into txt/srt/vtt, which have no separate speaker field.

    If `collapse_speakers` is set, consecutive same-speaker segments are merged
    into one turn (via `collapse_consecutive_speakers`) before the prefix is
    applied — a no-op for input with no speaker field.
    """
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TranscriptError(f"Could not read transcript JSON from {json_path}: {exc}") from exc

    if not isinstance(raw, list):
        raise TranscriptError(f"{json_path} does not look like a transcript JSON file (expected a list).")

    try:
        normalized = []
        for item in raw:
            start = float(item["start"])
            end = float(item["end"]) if "end" in item else start + float(item["duration"])
            normalized.append({"start": start, "end": end, "speaker": item.get("speaker"), "text": item["text"]})
    except (KeyError, TypeError, ValueError) as exc:
        raise TranscriptError(
            f"{json_path} does not look like a transcript JSON file "
            f"(each entry needs 'text', 'start', and either 'duration' or 'end'): {exc}"
        ) from exc

    if collapse_speakers:
        normalized = collapse_consecutive_speakers(normalized)

    snippets = [
        FetchedTranscriptSnippet(
            text=f"{seg['speaker']}: {seg['text']}" if seg["speaker"] else seg["text"],
            start=seg["start"],
            duration=seg["end"] - seg["start"],
        )
        for seg in normalized
    ]

    return FetchedTranscript(
        snippets=snippets,
        video_id=json_path.stem,
        language="unknown",
        language_code="unknown",
        is_generated=False,
    )


def fetch_video_title(video_id: str) -> str | None:
    """Best-effort video title lookup via YouTube's public oEmbed endpoint (no API key)."""
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=5,
        )
        resp.raise_for_status()
        title = resp.json().get("title")
        return title.strip() if title else None
    except Exception:
        return None


def slugify(text: str, max_length: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:max_length].rstrip("-") or "video"


def build_stem(video_id: str, title: str | None = None) -> str:
    """'<title-slug>-<video_id>', falling back to just the video id if no title is known."""
    if title is None:
        title = fetch_video_title(video_id)
    return f"{slugify(title)}-{video_id}" if title else video_id


def video_output_dir(
    video_id: str,
    base_dir: Path = DEFAULT_OUTPUT_DIR,
    title: str | None = None,
) -> Path:
    """One folder per video: './outputs/<title-slug>-<video_id>/'."""
    return base_dir / build_stem(video_id, title)


def default_output_path(
    video_id: str,
    fmt: str,
    language_code: str,
    base_dir: Path = DEFAULT_OUTPUT_DIR,
    title: str | None = None,
) -> Path:
    """Build './outputs/<title-slug>-<video_id>/transcript.<lang>.<fmt>'."""
    return video_output_dir(video_id, base_dir, title) / f"transcript.{language_code}.{fmt}"


def default_audio_output_path(
    video_id: str,
    base_dir: Path = DEFAULT_OUTPUT_DIR,
    title: str | None = None,
) -> Path:
    """Build './outputs/<title-slug>-<video_id>/audio' (extension is added by download_audio)."""
    return video_output_dir(video_id, base_dir, title) / "audio"


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def download_audio(
    video_id: str,
    output_path: Path,
    audio_format: str = "mp3",
    progress_hooks: list | None = None,
    postprocessor_hooks: list | None = None,
) -> Path:
    """Download the best-available audio track for a video.

    `output_path`'s suffix is ignored/replaced: the real extension is whatever
    `audio_format` resolves to ("best" keeps yt-dlp's original container, e.g.
    m4a/webm/opus; anything else re-encodes via ffmpeg, which must be on PATH).
    `progress_hooks`/`postprocessor_hooks` are passed straight through to yt-dlp
    (see its `progress_hooks`/`postprocessor_hooks` params) so callers can drive
    their own progress bar. Returns the actual path the audio was saved to.
    """
    if audio_format not in AUDIO_FORMATS:
        raise TranscriptError(f"Unsupported audio format: {audio_format!r} (choose one of {AUDIO_FORMATS})")
    if audio_format != "best" and not ffmpeg_available():
        raise TranscriptError(
            f"--audio-format {audio_format!r} requires ffmpeg, which was not found on PATH. "
            "Install ffmpeg, or pass --audio-format best to keep the original audio container "
            "without re-encoding."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.with_suffix("")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{stem}.%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": progress_hooks or [],
        "postprocessor_hooks": postprocessor_hooks or [],
    }
    if audio_format != "best":
        ydl_opts["postprocessors"] = [
            {"key": "FFmpegExtractAudio", "preferredcodec": audio_format, "preferredquality": "0"}
        ]

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise TranscriptError(f"Failed to download audio: {exc}") from exc

    requested = info.get("requested_downloads") or []
    if requested and requested[0].get("filepath"):
        return Path(requested[0]["filepath"])
    return Path(f"{stem}.{info.get('ext', audio_format)}")  # best-effort fallback
