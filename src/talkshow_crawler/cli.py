"""Typer-based CLI for downloading YouTube video transcripts."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

from talkshow_crawler.diarization import (
    API_KEY_ENV_VAR,
    DEFAULT_MODEL,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_TIMEOUT,
    MODELS,
    DiarizationError,
    diarize_audio,
)
from talkshow_crawler.logging_utils import get_logger, setup_logging
from talkshow_crawler.merge import merge_files, save_merged
from talkshow_crawler.settings import get_settings
from talkshow_crawler.youtube import (
    AUDIO_FORMATS,
    TranscriptError,
    collapse_consecutive_speakers,
    default_audio_output_path,
    default_output_path,
    download_audio,
    extract_video_id,
    fetch_transcript,
    fetch_video_title,
    format_transcript,
    list_available_languages,
    load_transcript_json,
    save_transcript,
)

app = typer.Typer(
    name="talkshow-crawler",
    help="Download and save YouTube video transcripts.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)
logger = get_logger()


@app.callback()
def _init(ctx: typer.Context) -> None:
    """Download and save YouTube video transcripts."""
    command = ctx.invoked_subcommand
    if command is None:
        return
    log_path = setup_logging(command)
    logger.info("Invoked `%s`", command)
    console.print(f"[dim]Log -> {log_path}[/dim]")


@app.command()
def download(
    videos: list[str] = typer.Argument(
        ..., help="One or more YouTube video ids/URLs (watch/youtu.be/shorts/embed)."
    ),
    lang: list[str] = typer.Option(
        ["en"],
        "--lang",
        "-l",
        help="Preferred language code(s), in priority order. Repeatable, e.g. -l en -l ja.",
    ),
    fmt: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="Output format: json, txt, srt, or vtt (json keeps per-segment start/duration, "
        "handy for LLM input or merging with speaker diarization; use `convert` to get srt/vtt later).",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Output file path (single VIDEO only). Defaults to a per-video folder "
            "under ./outputs/, e.g. outputs/<video-title>-<video_id>/transcript.<lang>.<format>."
        ),
    ),
    preserve_formatting: bool = typer.Option(
        False,
        "--preserve-formatting",
        help="Keep HTML formatting tags (e.g. <b>, <i>) from the original transcript.",
    ),
    audio: bool = typer.Option(
        False,
        "--audio",
        "-a",
        help="Also download the video's audio track (requires ffmpeg unless --audio-format best).",
    ),
    audio_format: str = typer.Option(
        "mp3",
        "--audio-format",
        help=f"Audio format for --audio: {', '.join(AUDIO_FORMATS)} ('best' = original container, no re-encode).",
    ),
    audio_output: Optional[Path] = typer.Option(
        None,
        "--audio-output",
        help=(
            "Audio output file path (single VIDEO only; extension is set by --audio-format). "
            "Defaults to 'audio.<ext>' alongside the transcript in the same per-video folder."
        ),
    ),
    workers: int = typer.Option(
        4,
        "--workers",
        "-w",
        min=1,
        help="Number of VIDEOS to download concurrently (only matters when more than one is given).",
    ),
) -> None:
    """Download transcripts (optionally audio) for one or more VIDEOS and save to disk.

    A single VIDEO gets a live per-stage progress bar. Multiple VIDEOS download
    concurrently (--workers controls how many at once) with one progress bar
    tracking overall completion.
    """
    if len(videos) > 1 and output is not None:
        err_console.print("[bold red]Error:[/bold red] --output can only be used with a single VIDEO.")
        raise typer.Exit(code=1)
    if len(videos) > 1 and audio_output is not None:
        err_console.print("[bold red]Error:[/bold red] --audio-output can only be used with a single VIDEO.")
        raise typer.Exit(code=1)

    if len(videos) == 1:
        _download_one_verbose(
            videos[0], lang, fmt, output, preserve_formatting, audio, audio_format, audio_output
        )
        return

    _download_many_parallel(videos, lang, fmt, preserve_formatting, audio, audio_format, workers)


def _download_one_verbose(
    video: str,
    lang: list[str],
    fmt: str,
    output: Optional[Path],
    preserve_formatting: bool,
    audio: bool,
    audio_format: str,
    audio_output: Optional[Path],
) -> None:
    """Download a single VIDEO with a live per-stage progress bar (original single-video UX)."""
    try:
        video_id = extract_video_id(video)
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}[/cyan]"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Starting", total=3)
            stage_started = False

            def on_stage(label: str) -> None:
                nonlocal stage_started
                if stage_started:
                    progress.advance(task_id)
                progress.update(task_id, description=label)
                stage_started = True

            fetched = fetch_transcript(
                video_id, languages=lang, preserve_formatting=preserve_formatting, on_stage=on_stage
            )
            on_stage("Formatting transcript")
            content = format_transcript(fetched, fmt)
            progress.update(task_id, description="Done", completed=3)
    except TranscriptError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    # Fetch the title once (if needed) and reuse it for both default paths below.
    needs_title = output is None or (audio and audio_output is None)
    title = fetch_video_title(video_id) if needs_title else None

    dest = output or default_output_path(video_id, fmt, fetched.language_code, title=title)
    saved_path = save_transcript(content, dest)
    console.print(
        f"[green]Saved[/green] transcript for [bold]{video_id}[/bold] "
        f"({fetched.language}, {len(fetched)} segments) -> [bold]{saved_path}[/bold]"
    )

    if audio:
        audio_dest = audio_output or default_audio_output_path(video_id, title=title)
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]{task.description}[/cyan]"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task("Audio", total=None)
                last_total: dict[str, Optional[int]] = {"value": None}

                def on_progress(d: dict) -> None:
                    total = d.get("total_bytes") or d.get("total_bytes_estimate")
                    if total:
                        last_total["value"] = total
                    if d["status"] == "downloading":
                        progress.update(
                            task_id,
                            total=total or last_total["value"],
                            completed=d.get("downloaded_bytes", 0),
                        )
                    elif d["status"] == "finished":
                        final_total = total or last_total["value"] or d.get("downloaded_bytes", 0)
                        progress.update(task_id, total=final_total, completed=final_total)

                def on_postprocess(d: dict) -> None:
                    if d["status"] == "started":
                        progress.update(task_id, description="Converting audio")
                    elif d["status"] == "finished":
                        progress.update(task_id, description="Converting audio (done)")

                saved_audio_path = download_audio(
                    video_id,
                    audio_dest,
                    audio_format=audio_format,
                    progress_hooks=[on_progress],
                    postprocessor_hooks=[on_postprocess],
                )
        except TranscriptError as exc:
            err_console.print(f"[bold red]Audio error:[/bold red] {exc}")
            raise typer.Exit(code=1)
        console.print(
            f"[green]Saved[/green] audio ({audio_format}) -> [bold]{saved_audio_path}[/bold]"
        )


def _download_many_parallel(
    videos: list[str],
    lang: list[str],
    fmt: str,
    preserve_formatting: bool,
    audio: bool,
    audio_format: str,
    workers: int,
) -> None:
    """Download transcripts (+ optional audio) for several VIDEOS concurrently.

    Each video runs the whole fetch -> [audio download] -> save pipeline in its
    own worker thread (network/subprocess I/O releases the GIL, so a thread
    pool gives real concurrency here); a single overall progress bar tracks
    completion count and a lock keeps updates to it safe across threads.
    """
    progress_lock = threading.Lock()
    results: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}[/cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(f"Downloading {len(videos)} videos ({workers} parallel)", total=len(videos))

        def work(video: str) -> dict:
            try:
                video_id = extract_video_id(video)
                logger.info("[%s] fetching transcript", video_id)
                fetched = fetch_transcript(video_id, languages=lang, preserve_formatting=preserve_formatting)
                content = format_transcript(fetched, fmt)
                title = fetch_video_title(video_id)
                dest = default_output_path(video_id, fmt, fetched.language_code, title=title)
                saved_path = save_transcript(content, dest)
                logger.info("[%s] transcript saved -> %s (%d segments)", video_id, saved_path, len(fetched))

                audio_path = None
                if audio:
                    audio_dest = default_audio_output_path(video_id, title=title)
                    audio_path = download_audio(video_id, audio_dest, audio_format=audio_format)
                    logger.info("[%s] audio saved -> %s", video_id, audio_path)

                return {
                    "video": video,
                    "video_id": video_id,
                    "ok": True,
                    "path": saved_path,
                    "audio_path": audio_path,
                    "segments": len(fetched),
                }
            except TranscriptError as exc:
                logger.error("[%s] failed: %s", video, exc)
                return {"video": video, "ok": False, "error": str(exc)}
            finally:
                with progress_lock:
                    progress.advance(task_id)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(work, videos):
                results.append(result)

    for r in results:
        if r["ok"]:
            audio_suffix = f", audio -> [bold]{r['audio_path']}[/bold]" if r.get("audio_path") else ""
            console.print(
                f"[green]Saved[/green] transcript for [bold]{r['video_id']}[/bold] "
                f"({r['segments']} segments) -> [bold]{r['path']}[/bold]{audio_suffix}"
            )
        else:
            err_console.print(f"[bold red]Error[/bold red] ({r['video']}): {r['error']}")

    ok_count = sum(1 for r in results if r["ok"])
    fail_count = len(results) - ok_count
    console.print(f"\n[bold]{ok_count}/{len(results)}[/bold] succeeded" + (f", [bold red]{fail_count} failed[/bold red]" if fail_count else ""))
    if fail_count:
        raise typer.Exit(code=1)


@app.command("list-languages")
def list_languages(
    video: str = typer.Argument(..., help="YouTube video id or URL."),
) -> None:
    """List transcript languages available for VIDEO."""
    try:
        video_id = extract_video_id(video)
        languages = list_available_languages(video_id)
    except TranscriptError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    if not languages:
        console.print("No transcripts available for this video.")
        return

    table = Table(title=f"Available transcripts for {video_id}")
    table.add_column("Language")
    table.add_column("Code")
    table.add_column("Generated")
    table.add_column("Translatable")
    for entry in languages:
        table.add_row(
            entry["language"],
            entry["language_code"],
            "yes" if entry["is_generated"] else "no",
            "yes" if entry["is_translatable"] else "no",
        )
    console.print(table)


@app.command()
def convert(
    json_file: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Transcript JSON (from `download -f json`) or speaker-tagged JSON (from `merge`).",
    ),
    fmt: str = typer.Option(
        "srt",
        "--format",
        "-f",
        help="Output format: srt, vtt, or txt.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path. Defaults to the input file with its extension swapped to --format.",
    ),
    collapse_speakers: bool = typer.Option(
        False,
        "--collapse-speakers/--no-collapse-speakers",
        "-c",
        help=(
            "Merge consecutive segments from the same speaker into one turn before "
            "converting (only affects speaker-tagged input from `merge`; no-op otherwise)."
        ),
    ),
) -> None:
    """Convert a saved transcript JSON_FILE into another format (srt, vtt, txt).

    If JSON_FILE has a "speaker" field per segment (i.e. `merge`'s output), each
    line is prefixed with its speaker label (e.g. "SPEAKER_00: ...").
    """
    try:
        fetched = load_transcript_json(json_file, collapse_speakers=collapse_speakers)
        content = format_transcript(fetched, fmt)
    except TranscriptError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    dest = output or json_file.with_suffix(f".{fmt}")
    saved_path = save_transcript(content, dest)
    console.print(
        f"[green]Converted[/green] {len(fetched)} segments from [bold]{json_file}[/bold] "
        f"-> [bold]{saved_path}[/bold]"
    )


@app.command()
def diarize(
    audio_file: Path = typer.Argument(
        ..., exists=True, readable=True, dir_okay=False, help="Local audio file to diarize."
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help=f"pyannote.ai API key. Defaults to {API_KEY_ENV_VAR} from the environment or a .env file.",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        help=f"Diarization model: {', '.join(MODELS)}.",
    ),
    num_speakers: Optional[int] = typer.Option(
        None, "--num-speakers", help="Exact number of speakers, if known."
    ),
    min_speakers: Optional[int] = typer.Option(None, "--min-speakers", help="Minimum number of speakers."),
    max_speakers: Optional[int] = typer.Option(None, "--max-speakers", help="Maximum number of speakers."),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSON file path. Defaults to 'diarization.json' next to AUDIO_FILE.",
    ),
    poll_interval: float = typer.Option(
        DEFAULT_POLL_INTERVAL, "--poll-interval", help="Seconds between job-status polls."
    ),
    timeout: float = typer.Option(
        DEFAULT_TIMEOUT, "--timeout", help="Max seconds to wait for the job to finish."
    ),
) -> None:
    """Run speaker diarization on AUDIO_FILE via pyannote.ai and save the result as JSON."""
    resolved_api_key = api_key or get_settings().pyannote_api_key
    if not resolved_api_key:
        err_console.print(
            f"[bold red]Error:[/bold red] No pyannote.ai API key. "
            f"Set {API_KEY_ENV_VAR} in the environment or a .env file, or pass --api-key."
        )
        raise typer.Exit(code=1)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}[/cyan]"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Starting", total=None)

            def on_stage(label: str) -> None:
                progress.update(task_id, description=label)

            def on_poll(status: str) -> None:
                progress.update(task_id, description=f"Waiting for diarization (status: {status})")

            segments = diarize_audio(
                audio_file,
                resolved_api_key,
                model=model,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                poll_interval=poll_interval,
                timeout=timeout,
                on_stage=on_stage,
                on_poll=on_poll,
            )
            progress.update(task_id, description="Done")
    except DiarizationError as exc:
        err_console.print(f"[bold red]Diarization error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    dest = output or audio_file.with_name("diarization.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(segments, indent=2), encoding="utf-8")

    speakers = sorted({seg["speaker"] for seg in segments if seg.get("speaker")})
    console.print(
        f"[green]Saved[/green] diarization ({len(segments)} segments, {len(speakers)} speaker(s): "
        f"{', '.join(speakers)}) -> [bold]{dest}[/bold]"
    )


@app.command()
def merge(
    transcript_json: Path = typer.Argument(
        ..., exists=True, readable=True, dir_okay=False, help="Transcript JSON (from `download -f json`)."
    ),
    diarization_json: Path = typer.Argument(
        ..., exists=True, readable=True, dir_okay=False, help="Diarization JSON (from `diarize`)."
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path. Defaults to 'merged.json' next to TRANSCRIPT_JSON.",
    ),
    collapse_speakers: bool = typer.Option(
        False,
        "--collapse-speakers/--no-collapse-speakers",
        "-c",
        help="Merge consecutive segments from the same speaker into one turn.",
    ),
) -> None:
    """Tag each transcript segment with its speaker from a diarization file.

    Aligns TRANSCRIPT_JSON's (text, start, duration) segments against
    DIARIZATION_JSON's (speaker, start, end) turns by time overlap, and writes
    out (start, end, speaker, text) segments.
    """
    try:
        merged = merge_files(transcript_json, diarization_json)
    except TranscriptError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    if collapse_speakers:
        merged = collapse_consecutive_speakers(merged)

    dest = output or transcript_json.with_name("merged.json")
    saved_path = save_merged(merged, dest)

    speakers = sorted({seg["speaker"] for seg in merged})
    console.print(
        f"[green]Merged[/green] {len(merged)} segments across {len(speakers)} speaker(s) "
        f"({', '.join(speakers)}) -> [bold]{saved_path}[/bold]"
    )


@app.command()
def pipeline(
    videos: list[str] = typer.Argument(
        ..., help="One or more YouTube video ids/URLs (watch/youtu.be/shorts/embed)."
    ),
    lang: list[str] = typer.Option(
        ["en"], "--lang", "-l", help="Preferred transcript language code(s), in priority order."
    ),
    audio_format: str = typer.Option(
        "mp3",
        "--audio-format",
        help=f"Audio format to download for diarization: {', '.join(AUDIO_FORMATS)}.",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help=f"pyannote.ai API key. Defaults to {API_KEY_ENV_VAR} from the environment or a .env file.",
    ),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help=f"Diarization model: {', '.join(MODELS)}."),
    num_speakers: Optional[int] = typer.Option(None, "--num-speakers", help="Exact number of speakers, if known."),
    min_speakers: Optional[int] = typer.Option(None, "--min-speakers", help="Minimum number of speakers."),
    max_speakers: Optional[int] = typer.Option(None, "--max-speakers", help="Maximum number of speakers."),
    poll_interval: float = typer.Option(
        DEFAULT_POLL_INTERVAL, "--poll-interval", help="Seconds between diarization job-status polls."
    ),
    timeout: float = typer.Option(
        DEFAULT_TIMEOUT, "--timeout", help="Max seconds to wait for each diarization job to finish."
    ),
    collapse_speakers: bool = typer.Option(
        True,
        "--collapse-speakers/--no-collapse-speakers",
        "-c",
        help="Merge consecutive same-speaker segments in the final merge (default: on).",
    ),
    fmt: Optional[str] = typer.Option(
        None,
        "--format",
        "-f",
        help="Also convert the merged output to this format (srt/vtt/txt) alongside merged.json.",
    ),
    workers: int = typer.Option(
        2,
        "--workers",
        "-w",
        min=1,
        help=(
            "Number of VIDEOS to run through the full pipeline concurrently. Diarization is "
            "billed pyannote.ai API usage, not local compute — keep this modest."
        ),
    ),
) -> None:
    """Run download -> diarize -> merge (--collapse-speakers) as one pipeline, per VIDEO.

    For each video: downloads the transcript (json) + audio into its usual per-video
    folder, diarizes the audio via pyannote.ai, then merges the two into
    <video-folder>/merged.json (speaker turns collapsed by default). Multiple VIDEOS
    run concurrently (--workers), each through its own full pipeline. Every stage is
    logged to ./logs/ in addition to the console summary below.
    """
    resolved_api_key = api_key or get_settings().pyannote_api_key
    if not resolved_api_key:
        err_console.print(
            f"[bold red]Error:[/bold red] No pyannote.ai API key. "
            f"Set {API_KEY_ENV_VAR} in the environment or a .env file, or pass --api-key."
        )
        raise typer.Exit(code=1)

    progress_lock = threading.Lock()
    results: list[dict] = []
    stages_per_video = 3  # download, diarize, merge

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}[/cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(
            f"Pipeline for {len(videos)} video(s) ({workers} parallel)", total=len(videos) * stages_per_video
        )

        def advance() -> None:
            with progress_lock:
                progress.advance(task_id)

        def work(video: str) -> dict:
            video_id = video
            try:
                video_id = extract_video_id(video)
                logger.info("[%s] pipeline: starting", video_id)

                # 1. download transcript (json) + audio
                fetched = fetch_transcript(video_id, languages=lang)
                title = fetch_video_title(video_id)
                transcript_path = save_transcript(
                    format_transcript(fetched, "json"),
                    default_output_path(video_id, "json", fetched.language_code, title=title),
                )
                logger.info("[%s] transcript saved -> %s (%d segments)", video_id, transcript_path, len(fetched))
                audio_path = download_audio(
                    video_id, default_audio_output_path(video_id, title=title), audio_format=audio_format
                )
                logger.info("[%s] audio saved -> %s", video_id, audio_path)
                advance()

                # 2. diarize
                segments = diarize_audio(
                    audio_path,
                    resolved_api_key,
                    model=model,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    poll_interval=poll_interval,
                    timeout=timeout,
                    on_stage=lambda label: logger.info("[%s] diarize: %s", video_id, label),
                    on_poll=lambda status: logger.debug("[%s] diarize: status=%s", video_id, status),
                )
                diarization_path = audio_path.with_name("diarization.json")
                diarization_path.write_text(json.dumps(segments, indent=2), encoding="utf-8")
                logger.info("[%s] diarization saved -> %s (%d segments)", video_id, diarization_path, len(segments))
                advance()

                # 3. merge (+ optional convert)
                merged = merge_files(transcript_path, diarization_path)
                if collapse_speakers:
                    merged = collapse_consecutive_speakers(merged)
                merged_path = save_merged(merged, transcript_path.with_name("merged.json"))
                speakers = sorted({s["speaker"] for s in merged if s.get("speaker")})
                logger.info(
                    "[%s] merged -> %s (%d segments, speakers: %s)",
                    video_id, merged_path, len(merged), ", ".join(speakers),
                )

                extra_path = None
                if fmt:
                    reloaded = load_transcript_json(merged_path, collapse_speakers=False)
                    extra_path = save_transcript(format_transcript(reloaded, fmt), merged_path.with_suffix(f".{fmt}"))
                    logger.info("[%s] converted -> %s", video_id, extra_path)
                advance()

                logger.info("[%s] pipeline: done", video_id)
                return {
                    "video": video, "video_id": video_id, "ok": True, "merged_path": merged_path,
                    "extra_path": extra_path, "segments": len(merged), "speakers": speakers,
                }
            except (TranscriptError, DiarizationError) as exc:
                logger.error("[%s] pipeline failed: %s", video_id, exc)
                return {"video": video, "ok": False, "error": str(exc)}
            except Exception as exc:  # keep one bad video from sinking the whole batch
                logger.exception("[%s] pipeline: unexpected error", video_id)
                return {"video": video, "ok": False, "error": f"unexpected error: {exc}"}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(work, videos):
                results.append(result)

    for r in results:
        if r["ok"]:
            extra = f", also -> [bold]{r['extra_path']}[/bold]" if r.get("extra_path") else ""
            console.print(
                f"[green]Done[/green] [bold]{r['video_id']}[/bold]: {r['segments']} segments, "
                f"{len(r['speakers'])} speaker(s) ({', '.join(r['speakers'])}) -> [bold]{r['merged_path']}[/bold]{extra}"
            )
        else:
            err_console.print(f"[bold red]Failed[/bold red] ({r['video']}): {r['error']}")

    ok_count = sum(1 for r in results if r["ok"])
    fail_count = len(results) - ok_count
    console.print(
        f"\n[bold]{ok_count}/{len(results)}[/bold] succeeded"
        + (f", [bold red]{fail_count} failed[/bold red]" if fail_count else "")
    )
    if fail_count:
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
