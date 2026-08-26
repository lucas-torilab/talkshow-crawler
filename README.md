# talkshow-crawler

CLI to download and save YouTube video transcripts, built with [Typer](https://typer.tiangolo.com/)
(on top of Click) and [`youtube-transcript-api`](https://github.com/jdepoix/youtube-transcript-api).

## Setup

```bash
uv sync
```

## Usage

Download a transcript (accepts a bare video id or any youtube.com/youtu.be URL):

```bash
uv run talkshow-crawler download https://www.youtube.com/watch?v=VIDEO_ID
```

Options:

- `-l/--lang` — preferred language code(s), in priority order, repeatable (default: `en`)
- `-f/--format` — output format: `json`, `txt`, `srt`, or `vtt` (default: `json` — per-segment
  `{text, start, duration}`, the best fit for LLM input or merging with speaker diarization; `srt`/
  `vtt` also carry timestamps, plain `txt` is timestamp-free prose)
- `-o/--output` — transcript file path. If omitted, auto-named in a **per-video folder** under
  `./outputs/`: `outputs/<video-title-slug>-<video_id>/transcript.<lang>.<format>` (falls back to
  just the video id if the title can't be looked up)
- `--preserve-formatting` — keep HTML tags (`<b>`, `<i>`, ...) from the original transcript
- `-a/--audio` — also download the video's audio track (via `yt-dlp`) into the **same** per-video
  folder
- `--audio-format` — `best` (original container, e.g. m4a/webm/opus, no re-encode — no ffmpeg
  needed), or `mp3`/`m4a`/`opus`/`wav`/`flac` to re-encode (requires `ffmpeg` on PATH). Default: `mp3`
- `--audio-output` — audio file path. If omitted, defaults to `audio.<ext>` in the same per-video
  folder as the transcript

Every video gets its own folder, so transcript and audio always end up together:

```
outputs/
  rick-astley-never-gonna-give-you-up-dQw4w9WgXcQ/
    transcript.en.json
    audio.mp3
```

Need `srt`/`vtt`/`txt` too? Convert the saved JSON instead of re-downloading:

```bash
uv run talkshow-crawler convert outputs/.../transcript.en.json -f srt
uv run talkshow-crawler convert outputs/.../transcript.en.json -f vtt
```

By default this writes next to the input file with the extension swapped (`-o` to override).

Example with audio:

```bash
uv run talkshow-crawler download https://youtu.be/VIDEO_ID --audio
uv run talkshow-crawler download https://youtu.be/VIDEO_ID --audio --audio-format best  # no ffmpeg needed
```

Examples:

```bash
uv run talkshow-crawler download dQw4w9WgXcQ -f srt -o transcripts/rickroll.srt  # explicit path overrides the folder default
uv run talkshow-crawler download https://youtu.be/dQw4w9WgXcQ -l ja -l en
```

List the transcript languages available for a video:

```bash
uv run talkshow-crawler list-languages https://www.youtube.com/watch?v=VIDEO_ID
```

### Speaker diarization

Run speaker diarization on any local audio file via [pyannote.ai](https://docs.pyannote.ai/api-reference/diarize).

Put your key in a `.env` file at the repo root (copy `.env.example`; `.env` is gitignored, so it's
never committed) — settings are loaded via [`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/):

```
PYANNOTE_API_KEY=sk_...   # get one at https://dashboard.pyannote.ai/
```

Or export it / pass `--api-key` instead — either works, no `.env` required:

```bash
uv run talkshow-crawler diarize outputs/<video-folder>/audio.mp3
```

This uploads the file to pyannote.ai's temporary storage, submits a diarize job, polls until it
finishes, and saves the result as `diarization.json` next to the audio file (`-o` to override):

```json
[
  {"speaker": "SPEAKER_00", "start": 15.0, "end": 30.5},
  {"speaker": "SPEAKER_01", "start": 31.0, "end": 42.3}
]
```

Options:

- `--api-key` — overrides the `PYANNOTE_API_KEY` env var
- `--model` — `precision-2` (default) or `community-1`
- `--num-speakers` / `--min-speakers` / `--max-speakers` — hint the speaker count if known
- `--poll-interval` / `--timeout` — job-polling cadence and max wait (default: 5s / 30min)

Note: pyannote.ai deletes job output 24 hours after completion — the `diarization.json` this
command saves locally is your permanent copy.

### Merging transcript + diarization

```bash
uv run talkshow-crawler merge outputs/<video-folder>/transcript.en.json outputs/<video-folder>/diarization.json
```

Tags each transcript segment with whichever speaker covers most of its `[start, start+duration]`
window (a numeric interval overlap, since both files use the same seconds-based units), falling
back to the nearest speaker in time for segments that land in a diarization gap (e.g. music/silence).
Defaults to `merged.json` next to the transcript file (`-o` to override):

```json
[
  {"start": 4.5, "end": 6.5, "speaker": "SPEAKER_01", "text": "thanks for having me"}
]
```

Same caveat as before: this aligns at caption-segment granularity (~2-5s chunks), not per-word —
for tighter alignment you'd need word-level timestamps from a forced-aligner or Whisper instead of
YouTube's captions.

Add `-c`/`--collapse-speakers` to merge consecutive same-speaker segments into one turn — since
diarization turns are usually much longer than individual caption lines, this turns a wall of
one-line-per-caption cues into one cue per actual speaker turn:

```bash
uv run talkshow-crawler merge transcript.en.json diarization.json --collapse-speakers
```

`convert` takes the same flag, so you can collapse at conversion time instead of (or in addition
to) at merge time:

```bash
uv run talkshow-crawler convert merged.json -f srt --collapse-speakers
```

It's a no-op on input with no `speaker` field (e.g. a plain `download -f json` transcript).

### Parallel downloads

`download` accepts multiple videos and runs them concurrently (`-w`/`--workers`, default 4):

```bash
uv run talkshow-crawler download VIDEO_ID_1 VIDEO_ID_2 VIDEO_ID_3 -f srt -w 6
```

A single video still gets the original live per-stage progress bar; `--output`/`--audio-output`
only work with exactly one video (there's nowhere sensible for them to point with several).

### One-command pipeline: download -> diarize -> merge

`pipeline` runs the whole download -> diarize -> merge chain per video (speaker turns collapsed
by default), and also accepts multiple videos to run concurrently:

```bash
uv run talkshow-crawler pipeline VIDEO_ID_1 VIDEO_ID_2 -w 2
```

For each video this downloads the transcript + audio, diarizes the audio via pyannote.ai, and
writes `<video-folder>/merged.json`. Add `-f srt` (or `vtt`/`txt`) to also get a converted file
alongside it. `--no-collapse-speakers` keeps one entry per caption line instead. `-w`/`--workers`
controls how many videos run through the full pipeline at once — keep this modest, since
diarization is billed pyannote.ai API usage, not local compute.

### Logs

Every command writes a timestamped log file to `./logs/` (e.g. `logs/pipeline-20250101-120000.log`)
recording each stage as it happens — handy for `download`/`pipeline` runs across many videos,
where the console summary only shows up at the end.

### Working around YouTube IP blocks

After enough transcript requests, YouTube blocks the *caption-fetch* endpoint for your IP
(`list-languages` keeps working — only the actual transcript download is affected) and starts
returning 429s. This happens to `yt-dlp` too, so it's YouTube-side rate limiting, not a bug in
either library. `youtube_transcript_api` (which this project uses) has built-in proxy support to
work around it — set one of these in `.env` (see `.env.example`) and every `download`/`pipeline`/
`list-languages` call routes through it automatically:

```
WEBSHARE_PROXY_USERNAME=...   # + WEBSHARE_PROXY_PASSWORD — rotating residential proxies via
WEBSHARE_PROXY_PASSWORD=...   # webshare.io (what the upstream library's own docs recommend)
```

or, for any proxy you already have:

```
YOUTUBE_PROXY_URL=http://user:pass@host:port
```

Without either set, requests go direct — exactly as before. If you don't have a proxy, the other
option is to just wait: these blocks are usually volume-based and clear after a cooldown period.
