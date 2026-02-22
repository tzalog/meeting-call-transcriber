# Transcribe Live Lecture (System Audio -> Transcript)

Python tool for transcribing system audio (for example an online lecture in Microsoft Teams) with priority on transcription quality rather than strict real-time output.

The script:
- captures audio from the system output device (loopback),
- splits audio into segments (for example 45-60 seconds) with overlap,
- sends segments sequentially to transcription (`OpenAI API` or local `faster-whisper`),
- prints each transcribed segment to the console,
- saves the transcript to `lecture-{date}-{start-time}.txt`.

## Requirements

- Windows (tested with loopback audio via `soundcard`)
- Python 3.10+
- (Optional) `OPENAI_API_KEY` or `openai_api_key` environment variable for the OpenAI backend

## Installation

```powershell
pip install -r requirements.txt
```

## Local GPU (CUDA) Support for Whisper

Both scripts support local `faster-whisper` on GPU via:
- `--device cuda`

Recommended for `large-v3` on NVIDIA GPU:
- `--compute-type float16`

Important (Windows):
- `nvidia-smi` showing your GPU is **not enough** by itself.
- `faster-whisper` / `ctranslate2` also needs CUDA runtime libraries (for example `cublas64_12.dll`) available on the system.
- If you just installed CUDA, restart your terminal / VS Code so the updated `PATH` is visible.

Typical requirement for this project:
- CUDA 12.x runtime/toolkit (because errors may reference `cublas64_12.dll`)

Example (live system audio, local GPU):

```powershell
python live_transcribe_system_audio.py --backend local --model large-v3 --device cuda --compute-type float16 --language pl
```

Quick verification (Windows):

```powershell
nvidia-smi
where cublas64_12.dll
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cuda', compute_type='float16'); print('GPU init ok')"
```

## OpenAI API Configuration

Set your API key in environment variables (either one is enough):

```powershell
$env:OPENAI_API_KEY="your_key"
```

or

```powershell
$env:openai_api_key="your_key"
```

## List Audio Devices

```powershell
python live_transcribe_system_audio.py --list-devices
```

Choose the output device that actually plays the lecture audio (prefer `Stereo`, not `Hands-Free`).

## Example Run (OpenAI API)

```powershell
python live_transcribe_system_audio.py --backend openai --speaker-name "Realtek" --language pl --segment-seconds 45 --overlap-seconds 5

```

## Offline File Transcription (mp4/mkv/mp3/wav)

Use `transcribe_media_file.py` to transcribe recorded files (including very long recordings, e.g. several hours).
It also supports a directory input (processed alphabetically) for cases where one lecture/meeting was recorded into multiple files (for example one file per block between breaks).

Supported inputs:
- a single file
- a directory (all supported files processed alphabetically by filename)

Supported file formats:
- `.mp4`
- `.mkv`
- `.mp3`
- `.wav`

How it works:
- If the input is `mp4`, `mkv`, or `mp3`, the script uses `ffmpeg` to extract/normalize audio first.
- If the input is a directory, files are processed in alphabetical order and merged into one transcript output.
- The file is processed in chunks (with overlap) for OpenAI/local Whisper.
- Chunk boundaries are selected near silence when possible to reduce sentence cuts.
- Output is saved to `lecture-{date}-{time}.txt`.

### Additional Requirement for Offline File Transcription

- `ffmpeg` must be installed and available in `PATH`

### Example Run (OpenAI, mp4)

```powershell
python transcribe_media_file.py "C:\path\to\recording.mp4" --backend openai --language pl --chunk-seconds 300 --overlap-seconds 8
```

### Example Run (OpenAI, mkv)

```powershell
python transcribe_media_file.py "C:\path\to\recording.mkv" --backend openai --language pl --chunk-seconds 300 --overlap-seconds 8
```

### Example Run (OpenAI, mp3)

```powershell
python transcribe_media_file.py "C:\path\to\recording.mp3" --backend openai --language pl
```

### Example Run (Local Whisper, wav)

```powershell
python transcribe_media_file.py "C:\path\to\recording.wav" --backend local --model large-v3 --device cuda --compute-type float16 --language pl --chunk-seconds 300 --overlap-seconds 8
```

### Example Run (Directory with multiple recording parts)

```powershell
python transcribe_media_file.py "C:\path\to\lecture_parts" --backend openai --language pl --chunk-seconds 300 --overlap-seconds 8
```

### Notes for Long Files (up to ~8 hours)

- Prefer larger chunks (for example `300s` to `600s`) for better cost/quality balance.
- Keep a small overlap (for example `5s` to `10s`) to preserve sentence continuity.
- If using OpenAI API, very large chunks may exceed the file upload limit; reduce `--chunk-seconds` if needed.

## How It Works (Segments and Overlap)

- `--segment-seconds` sets the audio segment length sent for transcription.
- `--overlap-seconds` creates overlap between consecutive segments to reduce sentence cuts at segment boundaries.
- The script removes overlap duplicates and tries to suppress repeated text after silence/breaks.

## Key Parameters

- `--backend {auto,openai,local}`: transcription backend
- `--speaker-name "..."`: part of the output device name
- `--language pl`: transcription language
- `--device {cpu,cuda}`: local Whisper device (GPU support for NVIDIA CUDA)
- `--compute-type float16`: recommended on CUDA for `large-v3` (local backend)
- `--segment-seconds 45`: segment length
- `--overlap-seconds 5`: segment overlap
- `--no-carry-context`: disables context from previous segments
- `--openai-model gpt-4o-transcribe`: OpenAI transcription model
- `--model large-v3`: local `faster-whisper` model (for `--backend local`)

## Stopping the Program

Press `Ctrl + C` in the terminal.

## Quality Notes

- Source audio quality has the biggest impact on transcription quality.
- Bluetooth `Hands-Free` mode (telephone profile) usually degrades transcription quality significantly.
- If possible, use headphones in `Stereo` mode and set the Teams microphone separately (for example laptop microphone).

## Troubleshooting

### 1. Very poor transcription quality / weird words

Most common cause: the audio device is running in Bluetooth `Hands-Free` mode (telephone profile), not `Stereo`.

What to do:
- In Teams, set the speaker device to the `Stereo` version of the headphones (not `Hands-Free`).
- In Teams, set the microphone to the laptop microphone or another mic so the headphones do not switch to `Hands-Free`.
- Check available devices:

```powershell
python live_transcribe_system_audio.py --list-devices
```

### 2. Repeated text after a break (silence)

This can happen because ASR models may hallucinate after a longer silence, especially with long segments.

The script includes safeguards (silence skipping and deduplication), but you can also:
- reduce `--segment-seconds` (for example to `45`)
- reduce `--overlap-seconds` (for example to `5`)
- disable context as a fallback: `--no-carry-context`

Example:

```powershell
python live_transcribe_system_audio.py --backend openai --speaker-name "Realtek" --language pl --segment-seconds 45 --overlap-seconds 5
```

### 3. The program does not capture Teams audio

Check:
- whether Teams is playing audio on the same device you pass in `--speaker-name`
- whether the selected device is actually an output device (`--list-devices`)
- whether Windows/Teams switched to another device after plugging/unplugging headphones

### 4. `data discontinuity in recording`

This means audio capture drops/discontinuities occurred.

The warning is hidden in the script, but the underlying issue can still affect transcription quality.

Common causes:
- Bluetooth `Hands-Free`
- CPU overload
- unstable audio device/driver

What helps:
- switch to `Stereo`
- shorter segments / smaller overlap
- use `--backend openai` instead of a heavy local model on a slower computer

### 5. OpenAI API does not work with `--backend openai`

Check:
- whether `OPENAI_API_KEY` or `openai_api_key` is set
- whether the `openai` package is installed

```powershell
pip install -r requirements.txt
```

### 6. Which backend should I use: `openai` or `local`?

- `openai`: usually better quality on difficult audio (Teams/Bluetooth), but requires API usage and cost
- `local`: works offline, but quality depends strongly on model size, CPU, and audio quality

### 7. Local GPU (`--device cuda`) does not work

Check:
- NVIDIA GPU is available (`nvidia-smi`)
- `faster-whisper` and `ctranslate2` are installed correctly
- use `--compute-type float16` for CUDA first
- CUDA runtime/toolkit is installed (CUDA 12.x if error mentions `cublas64_12.dll`)
- `cublas64_12.dll` is present (often under `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`)
- restart terminal/IDE after CUDA installation so `PATH` updates are applied

Fallback options:
- `--device cpu`
- `--compute-type int8` (CPU or sometimes useful if GPU memory is tight)

Common error example:
- `Library cublas64_12.dll is not found or cannot be loaded`
  - This usually means CUDA runtime DLLs are missing from the system or current shell environment.
  - Install CUDA 12.x and reopen the terminal.
