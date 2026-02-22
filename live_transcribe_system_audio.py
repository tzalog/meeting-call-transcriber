import argparse
import io
import os
import queue
import re
import sys
import threading
import time
import warnings
import wave
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import soundcard as sc

try:
    from faster_whisper import WhisperModel
except ImportError:  # Optional when using OpenAI API backend only.
    WhisperModel = None  # type: ignore[assignment]


# Hide noisy loopback warnings from soundcard while keeping other warnings visible.
warnings.filterwarnings(
    "ignore",
    message="data discontinuity in recording",
    module=r"soundcard\.mediafoundation",
)


@dataclass
class AudioChunk:
    data: np.ndarray
    sample_rate: int


@dataclass
class SegmentTask:
    index: int
    audio: np.ndarray
    sample_rate: int
    start_sec: float
    end_sec: float
    is_final: bool = False


def list_output_devices() -> None:
    speakers = sc.all_speakers()
    default_speaker = sc.default_speaker()
    print("Available output devices (loopback-capable):")
    for idx, speaker in enumerate(speakers, start=1):
        marker = " [domyslne]" if default_speaker and speaker.name == default_speaker.name else ""
        print(f"  {idx}. {speaker.name}{marker}")


def find_speaker_by_name(name_part: str):
    name_part_lower = name_part.lower()
    for speaker in sc.all_speakers():
        if name_part_lower in speaker.name.lower():
            return speaker
    return None


def choose_loopback_mic(speaker_name: str | None):
    speaker = find_speaker_by_name(speaker_name) if speaker_name else sc.default_speaker()
    if speaker is None:
        raise RuntimeError("Audio device not found.")
    return sc.get_microphone(id=str(speaker.name), include_loopback=True), speaker


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        mono = audio
    else:
        mono = audio.mean(axis=1)
    return mono.astype(np.float32, copy=False)


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    duration = audio.shape[0] / src_rate
    dst_len = max(1, int(round(duration * dst_rate)))
    src_x = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    out = np.interp(dst_x, src_x, audio)
    return out.astype(np.float32, copy=False)


def audio_capture_worker(
    out_queue: "queue.Queue[AudioChunk | None]",
    stop_event: threading.Event,
    speaker_name: str | None,
    capture_rate: int,
    frame_ms: int,
) -> None:
    try:
        loopback_mic, speaker = choose_loopback_mic(speaker_name)
        frames = max(1, int(capture_rate * frame_ms / 1000))
        print(f"Capturing system audio from: {speaker.name}")
        if "hands-free" in speaker.name.lower():
            print("[WARN] Bluetooth Hands-Free mode detected. Transcription quality will likely be poor.")
        with loopback_mic.recorder(samplerate=capture_rate) as recorder:
            while not stop_event.is_set():
                block = recorder.record(numframes=frames)
                if block is None or len(block) == 0:
                    continue
                out_queue.put(AudioChunk(data=block, sample_rate=capture_rate), timeout=2.0)
    except queue.Full:
        print("[ERROR] Audio queue is full. Reduce model/beam-size or use --backend openai.", file=sys.stderr)
    except Exception as exc:
        print(f"[ERROR] Audio capture: {exc}", file=sys.stderr)
    finally:
        try:
            out_queue.put(None, timeout=0.2)
        except Exception:
            pass


def segmenter_worker(
    in_queue: "queue.Queue[AudioChunk | None]",
    out_queue: "queue.Queue[SegmentTask | None]",
    stop_event: threading.Event,
    segment_seconds: float,
    overlap_seconds: float,
    target_rate: int,
    min_final_seconds: float,
) -> None:
    sample_buffer = np.zeros((0,), dtype=np.float32)
    buffer_start_sample = 0
    segment_index = 1

    window_samples = max(1, int(segment_seconds * target_rate))
    overlap_samples = max(0, int(overlap_seconds * target_rate))
    if overlap_samples >= window_samples:
        overlap_samples = max(0, window_samples // 4)
    step_samples = max(1, window_samples - overlap_samples)

    try:
        while True:
            if stop_event.is_set() and in_queue.empty():
                break

            try:
                item = in_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                break

            mono = to_mono_float32(item.data)
            mono_16k = resample_linear(mono, item.sample_rate, target_rate)
            if mono_16k.size:
                sample_buffer = np.concatenate([sample_buffer, mono_16k])

            while sample_buffer.shape[0] >= window_samples:
                segment = sample_buffer[:window_samples].copy()
                out_queue.put(
                    SegmentTask(
                        index=segment_index,
                        audio=segment,
                        sample_rate=target_rate,
                        start_sec=buffer_start_sample / target_rate,
                        end_sec=(buffer_start_sample + window_samples) / target_rate,
                    )
                )
                segment_index += 1
                sample_buffer = sample_buffer[step_samples:]
                buffer_start_sample += step_samples

        min_final_samples = int(min_final_seconds * target_rate)
        if sample_buffer.size >= max(1, min_final_samples):
            out_queue.put(
                SegmentTask(
                    index=segment_index,
                    audio=sample_buffer.copy(),
                    sample_rate=target_rate,
                    start_sec=buffer_start_sample / target_rate,
                    end_sec=(buffer_start_sample + sample_buffer.shape[0]) / target_rate,
                    is_final=True,
                )
            )
    except Exception as exc:
        print(f"[ERROR] Segmenter: {exc}", file=sys.stderr)
    finally:
        try:
            out_queue.put(None, timeout=0.2)
        except Exception:
            pass


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
    task: SegmentTask,
    model_name: str,
    language: str | None,
    prompt: str | None,
) -> str:
    kwargs = {
        "model": model_name,
        "file": audio_to_wav_bytes(task.audio, task.sample_rate),
    }
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
    task: SegmentTask,
    language: str | None,
    beam_size: int,
    prompt: str | None,
) -> str:
    segments, _info = model.transcribe(
        task.audio,
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

    if not rms_values:
        return audio, {"active_ratio": 0.0, "active_seconds": 0.0, "trimmed": False}

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
    trimmed_audio = audio[start_sample:end_sample].astype(np.float32, copy=False)

    return trimmed_audio, {
        "active_ratio": active_ratio,
        "active_seconds": active_seconds,
        "trimmed": start_sample > 0 or end_sample < audio.shape[0],
    }


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
    # Keep index alignment with curr_words, but empty normalized tokens are allowed there.
    max_k = min(len(prev_norm), len(curr_words), 80)
    if max_k < min_match_words:
        return current_text.strip()

    # Rebuild comparable normalized current sequence with same length as curr_words.
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

    trimmed = " ".join(curr_words[match_k:]).strip()
    return trimmed


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

    best_cut_idx: int | None = None
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
    # If almost whole segment is repeated, skip it entirely.
    if best_suffix_len >= max(min_match_words, int(len(curr_words) * 0.8)):
        return "", True
    return trimmed, True


def format_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_segment_header(task: SegmentTask, recording_started_at: datetime) -> str:
    abs_start = recording_started_at + timedelta(seconds=task.start_sec)
    abs_end = recording_started_at + timedelta(seconds=task.end_sec)
    rel_range = f"{format_hms(task.start_sec)}-{format_hms(task.end_sec)}"
    abs_range = f"{abs_start:%Y-%m-%d %H:%M:%S} -> {abs_end:%Y-%m-%d %H:%M:%S}"
    return f"[{task.index:04d}] {rel_range} | {abs_range}"


def build_prompt(base_prompt: str | None, carry_context: bool, prompt_history: "deque[str]") -> str | None:
    parts: list[str] = []
    if base_prompt:
        parts.append(base_prompt.strip())
    if carry_context and prompt_history:
        parts.append("Poprzedni kontekst transkrypcji: " + " ".join(prompt_history))
    if not parts:
        return None
    return "\n".join(parts)


def transcribe_worker(
    in_queue: "queue.Queue[SegmentTask | None]",
    stop_event: threading.Event,
    output_file: Path,
    backend: str,
    language: str | None,
    local_model_name: str,
    compute_type: str,
    beam_size: int,
    carry_context: bool,
    base_prompt: str | None,
    openai_model: str,
    recording_started_at: datetime,
    silence_rms_threshold: float,
    silence_dynamic_multiplier: float,
    silence_min_active_seconds: float,
    silence_min_active_ratio: float,
) -> None:
    client = None
    local_model = None
    prompt_history: "deque[str]" = deque(maxlen=6)
    prev_tail_text = ""
    prev_tail_words: "deque[str]" = deque(maxlen=160)
    recent_segment_texts: "deque[str]" = deque(maxlen=6)

    if backend == "openai":
        client = build_openai_client()
    elif backend == "local":
        if WhisperModel is None:
            raise RuntimeError("Missing faster-whisper. Install it or use --backend openai.")
        print(f"Loading local model: {local_model_name} ({compute_type})...")
        local_model = WhisperModel(local_model_name, device="cpu", compute_type=compute_type)
    else:
        raise RuntimeError(f"Unknown backend: {backend}")

    with output_file.open("a", encoding="utf-8") as f:
        f.write(f"# Start: {recording_started_at.isoformat(timespec='seconds')}\n")
        f.write(f"# Backend: {backend}\n")
        if backend == "openai":
            f.write(f"# Model: {openai_model}\n")
        else:
            f.write(f"# Model: {local_model_name}\n")
        f.flush()

        while True:
            if stop_event.is_set() and in_queue.empty():
                break

            try:
                task = in_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if task is None:
                break

            prepared_audio, audio_stats = trim_silence_edges(
                task.audio,
                task.sample_rate,
                rms_threshold=silence_rms_threshold,
                dynamic_multiplier=silence_dynamic_multiplier,
            )
            if (
                audio_stats["active_seconds"] < silence_min_active_seconds
                and audio_stats["active_ratio"] < silence_min_active_ratio
            ):
                if carry_context:
                    prompt_history.clear()
                print(
                    f"{format_segment_header(task, recording_started_at)} "
                    f"(silence / no speech - skipped)"
                )
                continue

            transcribe_task = SegmentTask(
                index=task.index,
                audio=prepared_audio,
                sample_rate=task.sample_rate,
                start_sec=task.start_sec,
                end_sec=task.end_sec,
                is_final=task.is_final,
            )

            prompt = build_prompt(base_prompt=base_prompt, carry_context=carry_context, prompt_history=prompt_history)
            try:
                if backend == "openai":
                    raw_text = transcribe_segment_openai(
                        client=client,
                        task=transcribe_task,
                        model_name=openai_model,
                        language=language,
                        prompt=prompt,
                    )
                else:
                    raw_text = transcribe_segment_local(
                        model=local_model,
                        task=transcribe_task,
                        language=language,
                        beam_size=beam_size,
                        prompt=prompt,
                    )
            except Exception as exc:
                print(f"[ERROR] Transcription failed for segment {task.index}: {exc}", file=sys.stderr)
                continue

            raw_text = " ".join(raw_text.strip().split())
            if not raw_text:
                continue

            emitted_text = trim_overlap_duplicate(prev_tail_text, raw_text)
            if not emitted_text:
                continue
            emitted_text, trimmed_repeat = trim_repeated_suffix_from_recent(emitted_text, recent_segment_texts)
            if trimmed_repeat and not emitted_text:
                print(
                    f"{format_segment_header(task, recording_started_at)} "
                    f"(repeated segment - skipped)"
                )
                continue
            if not emitted_text:
                continue

            for word in emitted_text.split():
                prev_tail_words.append(word)
            prev_tail_text = " ".join(prev_tail_words)

            if carry_context:
                prompt_history.append(emitted_text)
            recent_segment_texts.append(emitted_text)

            header = format_segment_header(task, recording_started_at)
            print(header)
            print(emitted_text, flush=True)

            f.write(header + "\n")
            f.write(emitted_text + "\n\n")
            f.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transkrypcja segmentowa dzwieku z komputera (Windows loopback / Stereo Mix), z priorytetem jakosci."
    )
    parser.add_argument("--list-devices", action="store_true", help="Pokaz dostepne urzadzenia wyjsciowe.")
    parser.add_argument(
        "--speaker-name",
        type=str,
        default=None,
        help="Fragment nazwy urzadzenia wyjsciowego (np. 'Speakers', 'Realtek', 'Stereo').",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "openai", "local"],
        default="auto",
        help="Backend transkrypcji. auto = OpenAI gdy jest klucz, inaczej lokalny faster-whisper.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-4o-transcribe",
        help="Model OpenAI do transkrypcji audio.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large-v3",
        help="Lokalny model faster-whisper: tiny, base, small, medium, large-v3.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="pl",
        help="Kod jezyka, np. pl, en. Puste = auto-detekcja.",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        help="Typ obliczen faster-whisper (np. int8, float16, float32).",
    )
    parser.add_argument(
        "--segment-seconds",
        "--chunk-seconds",
        dest="segment_seconds",
        type=float,
        default=60.0,
        help="Dlugosc segmentu audio do transkrypcji (sekundy). Domyslnie 60.",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=10.0,
        help="Nakladka miedzy segmentami (sekundy), poprawia spojnosc na granicach segmentow.",
    )
    parser.add_argument(
        "--min-final-seconds",
        type=float,
        default=8.0,
        help="Minimalna dlugosc ostatniego segmentu do wyslania (sekundy).",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Tylko backend=local. Szerokosc beam search (wieksze = lepsza jakosc, wolniej).",
    )
    parser.add_argument(
        "--no-carry-context",
        action="store_true",
        help="Wylacz przekazywanie krotkiego kontekstu z poprzednich segmentow.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="To jest transkrypcja wykladu online po polsku. Zachowaj poprawne polskie slowa i interpunkcje.",
        help="Dodatkowy prompt dla backendu transkrypcji.",
    )
    parser.add_argument(
        "--capture-rate",
        type=int,
        default=48000,
        help="Czestotliwosc probkowania przechwytywania (domyslnie 48000).",
    )
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=100,
        help="Wielkosc bloku odczytu z loopback w ms (domyslnie 100).",
    )
    parser.add_argument(
        "--target-rate",
        type=int,
        default=16000,
        help="Czestotliwosc probkowania wysylana do backendu (domyslnie 16000).",
    )
    parser.add_argument(
        "--silence-rms-threshold",
        type=float,
        default=0.001,
        help="Prog RMS dla wykrywania ciszy (mniejszy = mniej agresywne pomijanie segmentow).",
    )
    parser.add_argument(
        "--silence-dynamic-multiplier",
        type=float,
        default=1.5,
        help="Mnoznik adaptacyjnego progu ciszy bazujacego na medianie RMS segmentu.",
    )
    parser.add_argument(
        "--silence-min-active-seconds",
        type=float,
        default=0.5,
        help="Minimalna liczba sekund aktywnego dzwieku, ponizej ktorej segment moze byc uznany za cisze.",
    )
    parser.add_argument(
        "--silence-min-active-ratio",
        type=float,
        default=0.005,
        help="Minimalny udzial aktywnego dzwieku w segmencie (np. 0.005 = 0.5%%).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.list_devices:
        list_output_devices()
        return 0

    language = args.language or None
    backend = detect_backend(args.backend)

    started = datetime.now()
    output_name = f"lecture-{started:%Y-%m-%d}-{started:%H-%M-%S}.txt"
    output_file = Path(output_name)

    print(f"Transcription backend: {backend}")
    if backend == "openai":
        print(f"Model OpenAI: {args.openai_model}")
    else:
        print(f"Local model: {args.model}")
    print(f"Segments: {args.segment_seconds:.1f}s, overlap: {args.overlap_seconds:.1f}s")
    print(f"Output file: {output_file}")
    print("Press Ctrl+C to stop.\n")

    raw_audio_queue: "queue.Queue[AudioChunk | None]" = queue.Queue(maxsize=400)
    segment_queue: "queue.Queue[SegmentTask | None]" = queue.Queue()
    stop_event = threading.Event()

    capture_thread = threading.Thread(
        target=audio_capture_worker,
        args=(raw_audio_queue, stop_event, args.speaker_name, args.capture_rate, args.frame_ms),
        daemon=True,
    )
    segmenter_thread = threading.Thread(
        target=segmenter_worker,
        args=(
            raw_audio_queue,
            segment_queue,
            stop_event,
            args.segment_seconds,
            args.overlap_seconds,
            args.target_rate,
            args.min_final_seconds,
        ),
        daemon=True,
    )

    capture_thread.start()
    segmenter_thread.start()

    try:
        transcribe_worker(
            in_queue=segment_queue,
            stop_event=stop_event,
            output_file=output_file,
            backend=backend,
            language=language,
            local_model_name=args.model,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            carry_context=not args.no_carry_context,
            base_prompt=args.prompt,
            openai_model=args.openai_model,
            recording_started_at=started,
            silence_rms_threshold=args.silence_rms_threshold,
            silence_dynamic_multiplier=args.silence_dynamic_multiplier,
            silence_min_active_seconds=args.silence_min_active_seconds,
            silence_min_active_ratio=args.silence_min_active_ratio,
        )
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        for _ in range(40):
            if not capture_thread.is_alive() and not segmenter_thread.is_alive():
                break
            time.sleep(0.05)

    print(f"Transcript saved to: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
