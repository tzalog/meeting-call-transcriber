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

## How It Works (Segments and Overlap)

- `--segment-seconds` sets the audio segment length sent for transcription.
- `--overlap-seconds` creates overlap between consecutive segments to reduce sentence cuts at segment boundaries.
- The script removes overlap duplicates and tries to suppress repeated text after silence/breaks.

## Key Parameters

- `--backend {auto,openai,local}`: transcription backend
- `--speaker-name "..."`: part of the output device name
- `--language pl`: transcription language
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
