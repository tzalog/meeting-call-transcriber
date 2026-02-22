import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore[assignment]


OPENAI_SAFE_MAX_BYTES = 24 * 1024 * 1024  # keep margin below 25 MB
PCM16_MONO_16K_BYTES_PER_SEC = 16000 * 2
SUPPORTED_INPUT_EXTENSIONS = {".mp4", ".mkv", ".mp3", ".wav"}


@dataclass
class SegmentSpec:
    index: int
    start_sec: float
    end_sec: float


def get_openai_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")


def detect_backend(backend_arg: str) -> str:
    if backend_arg != "auto":
        return backend_arg
    return "openai" if get_openai_api_key() else "local"


def build_openai_client():
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY (or openai_api_key) environment variable.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing `openai` package. Install with: pip install openai") from exc
    return OpenAI(api_key=api_key)


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> io.BytesIO:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    raw = io.BytesIO()
    with wave.open(raw, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    raw.seek(0)
    raw.name = "segment.wav"  # type: ignore[attr-defined]
    return raw


def transcribe_segment_openai(
    client,
    audio: np.ndarray,
    sample_rate: int,
    model_name: str,
    language: str | None,
    prompt: str | None,
) -> str:
    kwargs = {"model": model_name, "file": audio_to_wav_bytes(audio, sample_rate)}
    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["prompt"] = prompt

    response = client.audio.transcriptions.create(**kwargs)
    if isinstance(response, str):
        return response.strip()
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text.strip()
    if hasattr(response, "model_dump"):
        payload = response.model_dump()
        maybe_text = payload.get("text")
        if isinstance(maybe_text, str):
            return maybe_text.strip()
    return str(response).strip()


def transcribe_segment_local(
    model: "WhisperModel",
    audio: np.ndarray,
    language: str | None,
    beam_size: int,
    prompt: str | None,
) -> str:
    segments, _info = model.transcribe(
        audio,
        language=language,
        beam_size=max(1, beam_size),
        best_of=1,
        vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt=prompt,
        temperature=0.0,
    )
    parts = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def normalize_token(token: str) -> str:
    return re.sub(r"[^\w]+", "", token.lower(), flags=re.UNICODE)


def trim_overlap_duplicate(prev_tail_text: str, current_text: str, min_match_words: int = 5) -> str:
    if not prev_tail_text or not current_text:
        return current_text.strip()

    prev_words = prev_tail_text.split()
    curr_words = current_text.split()
    if not prev_words or not curr_words:
        return current_text.strip()

    prev_norm = [normalize_token(w) for w in prev_words]
    prev_norm = [w for w in prev_norm if w]
    max_k = min(len(prev_norm), len(curr_words), 100)
    if max_k < min_match_words:
        return current_text.strip()

    curr_norm_full = [normalize_token(w) for w in curr_words]
    match_k = 0
    for k in range(max_k, min_match_words - 1, -1):
        prev_slice = [w for w in prev_norm[-k:] if w]
        curr_slice = [w for w in curr_norm_full[:k] if w]
        if len(prev_slice) == len(curr_slice) == k and prev_slice == curr_slice:
            match_k = k
            break

    if match_k == 0:
        return current_text.strip()
    return " ".join(curr_words[match_k:]).strip()


def trim_repeated_suffix_from_recent(
    current_text: str,
    recent_texts: "deque[str]",
    min_match_words: int = 20,
    min_repeated_fraction: float = 0.45,
) -> tuple[str, bool]:
    if not current_text or not recent_texts:
        return current_text, False

    curr_words = current_text.split()
    if len(curr_words) < min_match_words:
        return current_text, False

    recent_words: list[str] = []
    for txt in recent_texts:
        recent_words.extend(txt.split())
    if len(recent_words) < min_match_words:
        return current_text, False

    recent_norm = [normalize_token(w) for w in recent_words]
    recent_norm = [w for w in recent_norm if w]
    if len(recent_norm) < min_match_words:
        return current_text, False

    corpus = " " + " ".join(recent_norm) + " "
    curr_norm_full = [normalize_token(w) for w in curr_words]
    best_cut_idx = None
    best_suffix_len = 0

    for start_idx in range(0, len(curr_words)):
        suffix_norm = [w for w in curr_norm_full[start_idx:] if w]
        suffix_len = len(suffix_norm)
        if suffix_len < min_match_words:
            break
        if suffix_len < int(len(curr_words) * min_repeated_fraction):
            continue
        needle = " " + " ".join(suffix_norm) + " "
        if needle in corpus:
            best_cut_idx = start_idx
            best_suffix_len = suffix_len
            break

    if best_cut_idx is None:
        return current_text, False

    trimmed = " ".join(curr_words[:best_cut_idx]).strip()
    if best_suffix_len >= max(min_match_words, int(len(curr_words) * 0.8)):
        return "", True
    return trimmed, True


def trim_silence_edges(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: int = 30,
    rms_threshold: float = 0.001,
    dynamic_multiplier: float = 1.5,
    pad_ms: int = 250,
) -> tuple[np.ndarray, dict]:
    if audio.size == 0:
        return audio, {"active_ratio": 0.0, "active_seconds": 0.0, "trimmed": False}

    frame_samples = max(1, int(sample_rate * frame_ms / 1000))
    pad_samples = max(0, int(sample_rate * pad_ms / 1000))
    n_frames = int(np.ceil(audio.shape[0] / frame_samples))
    rms_values: list[float] = []

    for i in range(n_frames):
        start = i * frame_samples
        end = min(audio.shape[0], start + frame_samples)
        frame = audio[start:end]
        if frame.size == 0:
            rms_values.append(0.0)
            continue
        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
        rms_values.append(rms)

    rms_arr = np.asarray(rms_values, dtype=np.float32)
    dynamic_threshold = max(rms_threshold, float(np.median(rms_arr) * dynamic_multiplier))
    active_mask = rms_arr >= dynamic_threshold

    active_frames = int(np.count_nonzero(active_mask))
    active_ratio = active_frames / max(1, n_frames)
    active_seconds = active_frames * (frame_samples / sample_rate)
    if active_frames == 0:
        return np.zeros((0,), dtype=np.float32), {
            "active_ratio": 0.0,
            "active_seconds": 0.0,
            "trimmed": True,
        }

    first_frame = int(np.argmax(active_mask))
    last_frame = int(len(active_mask) - 1 - np.argmax(active_mask[::-1]))
    start_sample = max(0, first_frame * frame_samples - pad_samples)
    end_sample = min(audio.shape[0], (last_frame + 1) * frame_samples + pad_samples)
    return audio[start_sample:end_sample].astype(np.float32, copy=False), {
        "active_ratio": active_ratio,
        "active_seconds": active_seconds,
        "trimmed": start_sample > 0 or end_sample < audio.shape[0],
    }


def format_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_prompt(base_prompt: str | None, carry_context: bool, prompt_history: "deque[str]") -> str | None:
    parts: list[str] = []
    if base_prompt:
        parts.append(base_prompt.strip())
    if carry_context and prompt_history:
        parts.append("Previous transcript context: " + " ".join(prompt_history))
    if not parts:
        return None
    return "\n".join(parts)


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg to process mp4/mkv/mp3/wav files.")


def collect_input_files(path_arg: str) -> list[Path]:
    path = Path(path_arg).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
            raise ValueError("Unsupported input extension. Use .mp4, .mkv, .mp3, or .wav")
        return [path]

    if path.is_dir():
        files = [
            p
            for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS
        ]
        files.sort(key=lambda p: p.name.lower())
        if not files:
            raise ValueError(
                "Input directory contains no supported files (.mp4, .mkv, .mp3, .wav)."
            )
        return files

    raise ValueError(f"Unsupported input path type: {path}")


def normalize_media_to_wav(input_path: Path, temp_dir: Path) -> Path:
    ensure_ffmpeg()
    ext = input_path.suffix.lower()
    if ext not in SUPPORTED_INPUT_EXTENSIONS:
        raise ValueError("Unsupported input extension. Use .mp4, .mkv, .mp3, or .wav")

    if ext == ".wav":
        try:
            with wave.open(str(input_path), "rb") as wf:
                if wf.getnchannels() == 1 and wf.getframerate() == 16000 and wf.getsampwidth() == 2:
                    return input_path
        except wave.Error:
            pass

    out_wav = temp_dir / f"{input_path.stem}.{ext.lstrip('.')}.mono16k.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(out_wav),
    ]
    print("Preparing normalized WAV audio with ffmpeg...")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed while preparing audio.\n{stderr}") from exc
    return out_wav


def inspect_wav(wav_path: Path) -> dict:
    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
    duration = nframes / rate if rate else 0.0
    return {
        "channels": channels,
        "rate": rate,
        "sampwidth": sampwidth,
        "nframes": nframes,
        "duration_sec": duration,
    }


def load_wav_segment(wav_path: Path, start_sec: float, end_sec: float) -> np.ndarray:
    with wave.open(str(wav_path), "rb") as wf:
        rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        channels = wf.getnchannels()
        if sampwidth != 2:
            raise RuntimeError("Expected 16-bit PCM WAV after normalization.")
        start_frame = max(0, int(start_sec * rate))
        end_frame = max(start_frame, int(end_sec * rate))
        wf.setpos(min(start_frame, wf.getnframes()))
        raw = wf.readframes(max(0, end_frame - start_frame))

    if not raw:
        return np.zeros((0,), dtype=np.float32)

    audio = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return (audio.astype(np.float32) / 32768.0).astype(np.float32, copy=False)


def compute_rms_timeline(wav_path: Path, frame_ms: int = 30) -> tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        if sampwidth != 2:
            raise RuntimeError("Expected 16-bit PCM WAV after normalization.")
        frame_samples = max(1, int(rate * frame_ms / 1000))
        rms_values: list[float] = []

        while True:
            raw = wf.readframes(frame_samples)
            if not raw:
                break
            audio = np.frombuffer(raw, dtype=np.int16)
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            audio = audio.astype(np.float32) / 32768.0
            if audio.size == 0:
                rms_values.append(0.0)
            else:
                rms_values.append(float(np.sqrt(np.mean(np.square(audio), dtype=np.float64))))

    return np.asarray(rms_values, dtype=np.float32), rate


def find_cut_near_silence(
    rms_timeline: np.ndarray,
    frame_ms: int,
    target_sec: float,
    search_window_sec: float,
    min_sec: float,
    max_sec: float,
    silence_threshold: float,
    min_silence_frames: int,
) -> float | None:
    if rms_timeline.size == 0:
        return None

    frame_dur = frame_ms / 1000.0
    lo = max(min_sec, target_sec - search_window_sec)
    hi = min(max_sec, target_sec + search_window_sec)
    if hi <= lo:
        return None

    start_idx = max(0, int(lo / frame_dur))
    end_idx = min(len(rms_timeline) - 1, int(hi / frame_dur))
    if end_idx < start_idx:
        return None

    mask = rms_timeline[start_idx : end_idx + 1] <= silence_threshold
    if not np.any(mask):
        return None

    candidates: list[int] = []
    run_start = None
    for rel_idx, is_silent in enumerate(mask):
        if is_silent and run_start is None:
            run_start = rel_idx
        elif not is_silent and run_start is not None:
            run_len = rel_idx - run_start
            if run_len >= min_silence_frames:
                mid = run_start + run_len // 2
                candidates.append(start_idx + mid)
            run_start = None
    if run_start is not None:
        run_len = len(mask) - run_start
        if run_len >= min_silence_frames:
            mid = run_start + run_len // 2
            candidates.append(start_idx + mid)

    if not candidates:
        return None

    target_idx = int(target_sec / frame_dur)
    best_idx = min(candidates, key=lambda idx: abs(idx - target_idx))
    return best_idx * frame_dur


def build_segments_with_silence_boundaries(
    duration_sec: float,
    chunk_seconds: float,
    overlap_seconds: float,
    rms_timeline: np.ndarray,
    frame_ms: int,
    boundary_rms_threshold: float,
    boundary_dynamic_multiplier: float,
    boundary_search_window_sec: float,
    min_chunk_seconds: float,
) -> list[SegmentSpec]:
    if duration_sec <= 0:
        return []

    overlap_seconds = max(0.0, overlap_seconds)
    chunk_seconds = max(1.0, chunk_seconds)
    if overlap_seconds >= chunk_seconds:
        overlap_seconds = chunk_seconds * 0.2
    step_seconds = max(1.0, chunk_seconds - overlap_seconds)
    min_chunk_seconds = min(max(1.0, min_chunk_seconds), chunk_seconds)

    nonzero = rms_timeline[rms_timeline > 0]
    median_rms = float(np.median(nonzero)) if nonzero.size else 0.0
    silence_threshold = max(boundary_rms_threshold, median_rms * boundary_dynamic_multiplier)
    min_silence_frames = max(2, int(round(0.09 / (frame_ms / 1000.0))))  # ~90ms

    segments: list[SegmentSpec] = []
    start = 0.0
    idx = 1
    while start < duration_sec:
        remaining = duration_sec - start
        if remaining <= chunk_seconds:
            end = duration_sec
        else:
            target = start + chunk_seconds
            min_cut = start + min_chunk_seconds
            max_cut = min(duration_sec, start + max(chunk_seconds, min_chunk_seconds))
            candidate = find_cut_near_silence(
                rms_timeline=rms_timeline,
                frame_ms=frame_ms,
                target_sec=target,
                search_window_sec=boundary_search_window_sec,
                min_sec=min_cut,
                max_sec=max_cut,
                silence_threshold=silence_threshold,
                min_silence_frames=min_silence_frames,
            )
            end = candidate if candidate is not None else target
            end = max(min_cut, min(end, duration_sec))

        if end - start < 0.5:
            break
        segments.append(SegmentSpec(index=idx, start_sec=start, end_sec=end))
        idx += 1

        if end >= duration_sec:
            break

        next_start = max(0.0, end - overlap_seconds)
        if next_start <= start:
            next_start = start + step_seconds
        start = next_start

    return segments


def estimate_openai_chunk_limit_sec() -> float:
    # WAV adds header and small metadata overhead; keep margin.
    return max(60.0, (OPENAI_SAFE_MAX_BYTES - 64_000) / PCM16_MONO_16K_BYTES_PER_SEC)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe mp4/mkv/mp3/wav recordings (including very long files) using OpenAI or local Whisper."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input .mp4/.mkv/.mp3/.wav file OR a directory with such files (processed alphabetically).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "openai", "local"],
        default="auto",
        help="Transcription backend. auto = OpenAI if API key is present, otherwise local faster-whisper.",
    )
    parser.add_argument("--openai-model", type=str, default="gpt-4o-transcribe", help="OpenAI transcription model.")
    parser.add_argument(
        "--model",
        type=str,
        default="large-v3",
        help="Local faster-whisper model (tiny/base/small/medium/large-v3).",
    )
    parser.add_argument("--language", type=str, default="pl", help="Language code, e.g. pl, en. Empty = auto.")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for local faster-whisper backend (cpu or cuda).",
    )
    parser.add_argument("--compute-type", type=str, default="int8", help="Local whisper compute type.")
    parser.add_argument("--beam-size", type=int, default=5, help="Local whisper beam size.")
    parser.add_argument("--no-carry-context", action="store_true", help="Disable prompt context from prior chunks.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="This is a transcription of a spoken lecture/meeting. Preserve wording and punctuation.",
        help="Additional transcription prompt/context.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=300.0,
        help="Target chunk length in seconds (default 300 = 5 min).",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=8.0,
        help="Overlap between chunks in seconds.",
    )
    parser.add_argument(
        "--min-chunk-seconds",
        type=float,
        default=120.0,
        help="Minimum chunk length when snapping chunk end to silence.",
    )
    parser.add_argument(
        "--boundary-search-window-seconds",
        type=float,
        default=20.0,
        help="Search window around target chunk end to find a silence boundary.",
    )
    parser.add_argument(
        "--boundary-rms-threshold",
        type=float,
        default=0.001,
        help="Static RMS threshold used to detect silence for chunk boundaries.",
    )
    parser.add_argument(
        "--boundary-dynamic-multiplier",
        type=float,
        default=1.5,
        help="Dynamic silence threshold multiplier for chunk boundary selection.",
    )
    parser.add_argument(
        "--silence-rms-threshold",
        type=float,
        default=0.001,
        help="RMS threshold for trimming silence at chunk edges before transcription.",
    )
    parser.add_argument(
        "--silence-dynamic-multiplier",
        type=float,
        default=1.5,
        help="Dynamic multiplier for edge-silence trimming before transcription.",
    )
    parser.add_argument(
        "--silence-min-active-seconds",
        type=float,
        default=0.5,
        help="If active audio is below this and ratio threshold, chunk is treated as silence.",
    )
    parser.add_argument(
        "--silence-min-active-ratio",
        type=float,
        default=0.005,
        help="Minimum active-audio ratio before a chunk is treated as silence.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        input_files = collect_input_files(args.input_file)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    input_path = Path(args.input_file).expanduser().resolve()

    backend = detect_backend(args.backend)
    language = args.language or None
    processing_started = datetime.now()
    output_file = Path(f"lecture-{processing_started:%Y-%m-%d}-{processing_started:%H-%M-%S}.txt")

    if input_path.is_dir():
        print(f"Input directory: {input_path}")
        print(f"Files to process (alphabetical): {len(input_files)}")
    else:
        print(f"Input file: {input_path}")
    print(f"Transcription backend: {backend}")
    if backend == "openai":
        print(f"OpenAI model: {args.openai_model}")
    else:
        print(f"Local model: {args.model} ({args.device}, {args.compute_type})")
    print(f"Output file: {output_file}")

    if backend == "openai":
        max_openai_chunk_sec = estimate_openai_chunk_limit_sec()
        if args.chunk_seconds > max_openai_chunk_sec:
            print(
                f"[WARN] --chunk-seconds={args.chunk_seconds:.1f}s may exceed OpenAI 25MB upload limit for WAV chunks."
            )
            print(f"[WARN] Consider <= {max_openai_chunk_sec:.0f}s for mono 16k PCM WAV.")

    client = None
    local_model = None
    if backend == "openai":
        client = build_openai_client()
    else:
        if WhisperModel is None:
            print("[ERROR] faster-whisper is not installed. Use --backend openai or install dependencies.", file=sys.stderr)
            return 1
        print("Loading local whisper model...")
        local_model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    prompt_history: "deque[str]" = deque(maxlen=6)
    prev_tail_words: "deque[str]" = deque(maxlen=180)
    recent_segment_texts: "deque[str]" = deque(maxlen=6)
    prev_tail_text = ""
    global_chunk_index = 1
    processed_files = 0

    with tempfile.TemporaryDirectory(prefix="lecture_transcribe_") as tmp:
        temp_dir = Path(tmp)
        with output_file.open("a", encoding="utf-8") as f:
            f.write(f"# Start: {processing_started.isoformat(timespec='seconds')}\n")
            f.write(f"# Input: {input_path}\n")
            if input_path.is_dir():
                f.write(f"# Files (alphabetical): {len(input_files)}\n")
            f.write(f"# Backend: {backend}\n")
            f.write(f"# Model: {args.openai_model if backend == 'openai' else args.model}\n")
            f.flush()

            for file_idx, media_path in enumerate(input_files, start=1):
                print(f"--- Source file {file_idx}/{len(input_files)}: {media_path.name} ---")
                f.write(f"\n# Source file {file_idx}/{len(input_files)}: {media_path.name}\n")
                f.flush()

                # Reset overlap/dedup state at file boundaries (common case: recording restarted during break).
                prev_tail_words.clear()
                prev_tail_text = ""
                recent_segment_texts.clear()
                if not args.no_carry_context:
                    prompt_history.clear()

                try:
                    wav_path = normalize_media_to_wav(media_path, temp_dir)
                except Exception as exc:
                    print(f"[ERROR] Failed to prepare audio for {media_path.name}: {exc}", file=sys.stderr)
                    f.write(f"# ERROR preparing audio for {media_path.name}: {exc}\n")
                    f.flush()
                    continue

                try:
                    info = inspect_wav(wav_path)
                except Exception as exc:
                    print(f"[ERROR] Failed to read WAV info for {media_path.name}: {exc}", file=sys.stderr)
                    f.write(f"# ERROR reading WAV info for {media_path.name}: {exc}\n")
                    f.flush()
                    continue

                if info["channels"] != 1 or info["rate"] != 16000 or info["sampwidth"] != 2:
                    msg = f"[ERROR] Internal error: expected normalized mono 16k PCM16 WAV for {media_path.name}."
                    print(msg, file=sys.stderr)
                    f.write(f"# {msg}\n")
                    f.flush()
                    continue

                print(
                    f"Prepared WAV: {wav_path.name} | duration={info['duration_sec']/3600:.2f}h | "
                    f"{info['duration_sec']:.0f}s"
                )
                print("Scanning audio energy to choose chunk boundaries near silence...")

                try:
                    rms_timeline, _rate = compute_rms_timeline(wav_path, frame_ms=30)
                except Exception as exc:
                    print(f"[ERROR] Failed while scanning {media_path.name}: {exc}", file=sys.stderr)
                    f.write(f"# ERROR scanning audio for {media_path.name}: {exc}\n")
                    f.flush()
                    continue

                segments = build_segments_with_silence_boundaries(
                    duration_sec=info["duration_sec"],
                    chunk_seconds=args.chunk_seconds,
                    overlap_seconds=args.overlap_seconds,
                    rms_timeline=rms_timeline,
                    frame_ms=30,
                    boundary_rms_threshold=args.boundary_rms_threshold,
                    boundary_dynamic_multiplier=args.boundary_dynamic_multiplier,
                    boundary_search_window_sec=args.boundary_search_window_seconds,
                    min_chunk_seconds=args.min_chunk_seconds,
                )
                if not segments:
                    print(f"[WARN] No chunks were produced for {media_path.name}. Skipping.")
                    f.write(f"# WARN no chunks produced for {media_path.name}\n")
                    f.flush()
                    continue
                print(f"Planned chunks for {media_path.name}: {len(segments)}")
                processed_files += 1

                for seg in segments:
                    raw_audio = load_wav_segment(wav_path, seg.start_sec, seg.end_sec)
                    audio, stats = trim_silence_edges(
                        raw_audio,
                        sample_rate=16000,
                        rms_threshold=args.silence_rms_threshold,
                        dynamic_multiplier=args.silence_dynamic_multiplier,
                    )

                    header = (
                        f"[{global_chunk_index:04d}] {media_path.name} "
                        f"{format_hms(seg.start_sec)}-{format_hms(seg.end_sec)}"
                    )
                    if (
                        stats["active_seconds"] < args.silence_min_active_seconds
                        and stats["active_ratio"] < args.silence_min_active_ratio
                    ):
                        if not args.no_carry_context:
                            prompt_history.clear()
                        msg = f"{header} (silence / no speech - skipped)"
                        print(msg)
                        f.write(msg + "\n")
                        global_chunk_index += 1
                        continue

                    prompt = build_prompt(args.prompt, not args.no_carry_context, prompt_history)

                    try:
                        if backend == "openai":
                            text = transcribe_segment_openai(
                                client=client,
                                audio=audio,
                                sample_rate=16000,
                                model_name=args.openai_model,
                                language=language,
                                prompt=prompt,
                            )
                        else:
                            text = transcribe_segment_local(
                                model=local_model,
                                audio=audio,
                                language=language,
                                beam_size=args.beam_size,
                                prompt=prompt,
                            )
                    except Exception as exc:
                        print(f"[ERROR] Transcription failed for {header}: {exc}", file=sys.stderr)
                        global_chunk_index += 1
                        continue

                    text = " ".join(text.strip().split())
                    if not text:
                        global_chunk_index += 1
                        continue

                    emitted = trim_overlap_duplicate(prev_tail_text, text)
                    if not emitted:
                        global_chunk_index += 1
                        continue
                    emitted, repeated = trim_repeated_suffix_from_recent(emitted, recent_segment_texts)
                    if repeated and not emitted:
                        msg = f"{header} (repeated chunk - skipped)"
                        print(msg)
                        f.write(msg + "\n")
                        global_chunk_index += 1
                        continue
                    if not emitted:
                        global_chunk_index += 1
                        continue

                    for word in emitted.split():
                        prev_tail_words.append(word)
                    prev_tail_text = " ".join(prev_tail_words)
                    if not args.no_carry_context:
                        prompt_history.append(emitted)
                    recent_segment_texts.append(emitted)

                    processed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    full_header = f"{header} | processed {processed_at}"
                    print(full_header)
                    print(emitted)

                    f.write(full_header + "\n")
                    f.write(emitted + "\n\n")
                    f.flush()
                    global_chunk_index += 1

    if processed_files == 0:
        print("[ERROR] No input files were processed successfully.", file=sys.stderr)
        return 1

    print(f"Transcript saved to: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
