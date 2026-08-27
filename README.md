# Video Dubbing Workflow

A fully local, GPU-accelerated video dubbing pipeline running in Docker Desktop on Windows.

The goal is to take a video containing dialogue and produce a professionally dubbed version while preserving the original video, music, ambience, sound effects, speaker identity, timing, emotion, and overall scene feel.

The workflow is designed to run locally using an NVIDIA GPU, with support for different GPU configurations such as:

- NVIDIA RTX 2060 6 GB
- NVIDIA RTX 4070 Ti 12 GB

No cloud services are required for the core workflow.

---

## Project Goal

The goal is to build a production-quality local video dubbing system that can:

1. Inspect an input video.
2. Extract and analyze its audio.
3. Transcribe dialogue with timestamps.
4. Identify individual speakers.
5. Build persistent speaker profiles.
6. Translate dialogue while preserving context and natural speech.
7. Generate translated speech using the original speakers' voices.
8. Preserve timing, emotion, prosody, vocalizations, and speaking style.
9. Separate dialogue from music, ambience, and sound effects.
10. Mix the generated dialogue back into the original sound environment.
11. Render a final dubbed video.

The final result should sound and feel as close as reasonably possible to the original performance, while using locally executed AI models.

---

# Architecture

The project runs inside Docker Desktop on Windows.

```text
Windows
│
├── Docker Desktop
├── WSL2
└── NVIDIA GPU
      │
      ▼
Docker Container
│
├── FFmpeg / FFprobe
├── PyTorch / CUDA
├── Faster-Whisper
├── Speech diarization
├── Translation model
├── Source separation
├── Voice cloning / TTS
└── Audio processing
````

Persistent directories are mounted from the host:

```text
input/
work/
models/
output/
```

This keeps generated data and downloaded models outside the container filesystem.

---

# Current Pipeline

The workflow is being developed incrementally.

Current implemented stages:

```text
Input video
    │
    ▼
GPU detection
    │
    ▼
Video inspection
    │
    ├── video_metadata.json
    │
    ▼
Audio extraction
    │
    ├── source_audio.wav
    │
    └── speech_audio.wav
    │
    ▼
Faster-Whisper
    │
    ▼
transcript.json
```

## Current Status

| Stage                         | Status                 |
| ----------------------------- | ---------------------- |
| Docker / CUDA                 | Complete               |
| NVIDIA GPU detection          | Complete               |
| RTX 2060 profile              | Complete               |
| RTX 4070 Ti profile           | Planned / configurable |
| FFmpeg                        | Complete               |
| FFprobe                       | Complete               |
| Video inspection              | Complete               |
| Source audio extraction       | Complete               |
| Speech audio normalization    | Complete               |
| Faster-Whisper                | Complete               |
| Timestamped transcription     | Complete               |
| Word timestamps               | Complete               |
| Speaker diarization           | Next                   |
| Speaker profiles              | Planned                |
| Translation                   | Planned                |
| Voice cloning / TTS           | Planned                |
| Dialogue timing               | Planned                |
| Source separation             | Planned                |
| Audio mixing                  | Planned                |
| Final video rendering         | Planned                |
| Model initialization/cache    | Planned                |
| GPU optimization              | Planned                |
| Production voice preservation | Planned                |

---

# Hardware

The workflow is designed around NVIDIA GPUs with different VRAM capacities.

## RTX 2060

Current development target:

```text
GPU: NVIDIA GeForce RTX 2060
VRAM: 6 GB
Compute Capability: 7.5
```

The RTX 2060 has already been successfully tested with:

```text
Whisper model: large-v3
Compute type:  int8_float16
Device:        CUDA
```

This configuration is an important baseline for the project.

## RTX 4070 Ti

The workflow will also support:

```text
GPU: NVIDIA GeForce RTX 4070 Ti
VRAM: 12 GB
```

The application detects the GPU automatically and can select an appropriate configuration.

The goal is to use higher-quality or faster models/settings when additional VRAM is available.

---

# Local-Only Architecture

The core workflow is designed to run locally.

No cloud service should be required for:

* Speech-to-text
* Speaker diarization
* Translation
* Source separation
* Voice cloning
* TTS
* Audio processing
* Video processing

Models are downloaded and cached locally.

```text
models/
```

is used as persistent model storage so models do not need to be downloaded every time the Docker container is rebuilt.

Once the required models are available locally, the pipeline should be capable of operating without an internet connection.

---

# Docker

The project uses an NVIDIA CUDA runtime image.

Current base image:

```dockerfile
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
```

The container includes:

* Python
* pip
* Python virtual environment
* FFmpeg
* PyTorch
* Faster-Whisper

GPU access is provided by Docker Desktop and the NVIDIA container runtime.

---

# Directory Structure

The project is organized approximately as follows:

```text
workflow2/
│
├── app/
│   ├── main.py
│   ├── gpu.py
│   ├── media.py
│   └── transcription.py
│
├── input/
│   └── .gitkeep
│
├── work/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── output/
│   └── .gitkeep
│
├── Dockerfile
├── compose.yaml
├── .gitignore
└── README.md
```

Generated files should not be committed to Git.

---

# Input

Place a video file in:

```text
input/
```

For example:

```text
input/
└── test.mp4
```

Supported video extensions currently include:

```text
.mp4
.mkv
.mov
.avi
.webm
.m4v
```

The pipeline currently expects a single input video.

---

# Generated Working Files

Intermediate pipeline files are stored in:

```text
work/
```

Examples:

```text
work/
├── video_metadata.json
├── source_audio.wav
├── speech_audio.wav
└── transcript.json
```

These files are generated automatically and should remain outside version control.

---

# Models

AI models are stored in:

```text
models/
```

The directory is persistent across Docker rebuilds.

For example:

```text
models/
└── whisper/
```

The current known-good Whisper configuration is:

```text
WHISPER_MODEL=large-v3
WHISPER_COMPUTE_TYPE=int8_float16
```

This configuration has been successfully tested on an RTX 2060 6 GB.

---

# Transcription

Speech recognition uses Faster-Whisper.

The transcription stage generates:

```text
work/transcript.json
```

The transcript contains:

* Detected language
* Language probability
* Segment timestamps
* Segment text
* Word timestamps
* Word probabilities
* Model information
* Transcription timing

Example:

```json
{
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "Example dialogue.",
      "words": [
        {
          "start": 0.0,
          "end": 0.4,
          "word": "Example",
          "probability": 0.98
        }
      ]
    }
  ]
}
```

Word-level timestamps are retained because they will be useful for later speaker assignment, translation timing, and TTS synchronization.

---

# Audio Processing

Two audio representations are created from the source video.

## Source Audio

The original audio is converted to PCM WAV while preserving its source characteristics as much as possible.

Example:

```text
PCM s16le
44.1 kHz
Stereo
```

This audio is intended for later processing such as source separation and final mixing.

## Speech Audio

A normalized speech-processing version is created:

```text
PCM s16le
16 kHz
Mono
```

This version is optimized for speech recognition.

The original audio is preserved separately so that downstream processing does not have to repeatedly decode the original compressed audio.

---

# Planned Pipeline

The complete target workflow is:

```text
                INPUT VIDEO
                     │
                     ▼
              Video Inspection
                     │
             ┌───────┴────────┐
             ▼                ▼
        Video Metadata     Audio
                              │
                              ▼
                       Audio Extraction
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
         Speech Audio                 Source Audio
                │                           │
                ▼                           │
          Speech-to-Text                    │
                │                           │
                ▼                           │
          Speaker Diarization              │
                │                           │
                ▼                           │
        Speaker Profiles                    │
                │                           │
                ▼                           │
           Translation                      │
                │                           │
                ▼                           │
        Voice Cloning / TTS                 │
                │                           │
                ▼                           │
          Timing Alignment                  │
                │                           │
                └──────────────┐            │
                               ▼            ▼
                         Dialogue     Source Separation
                               │            │
                               └─────┬──────┘
                                     ▼
                                Audio Mixing
                                     │
                                     ▼
                              Video Rendering
                                     │
                                     ▼
                              DUBBED VIDEO
```

---

# Speaker Diarization

The next major stage is speaker diarization.

Whisper answers:

> What was said and when?

Diarization answers:

> Who was speaking when?

The two outputs will eventually be combined into speaker-specific dialogue:

```text
SPEAKER_00
00:00.00 → 00:03.42
"Dialogue..."

SPEAKER_01
00:03.42 → 00:06.81
"Response..."

SPEAKER_00
00:06.81 → 00:09.12
"Dialogue..."
```

This information will later drive speaker-specific voice generation.

---

# Speaker Profiles

Each detected speaker will eventually have a persistent profile containing information such as:

```text
speaker_00/
├── profile.json
├── reference_audio/
└── generated/
```

The profile will eventually contain:

* Speaker identifier
* Clean reference recordings
* Preferred reference samples
* Voice characteristics
* Language information
* TTS configuration
* Quality information

The system should prefer clean, representative samples rather than simply selecting the longest available segment.

---

# Translation

Translation will be performed locally.

The goal is not literal word-for-word translation.

The translation stage should preserve:

* Meaning
* Context
* Character intent
* Natural spoken language
* Tone
* Approximate sentence length
* Timing constraints

The translated dialogue will later be passed to the voice generation stage.

---

# Voice Cloning / TTS

The target system should generate translated speech using the appropriate speaker's voice.

The system should eventually preserve:

* Speaker identity
* Voice characteristics
* Prosody
* Emotion
* Speaking style
* Vocalizations
* Non-verbal sounds
* Appropriate pauses
* Scene context

The system should not simply assign a generic TTS voice to every character.

---

# Timing

Translated speech will rarely have exactly the same duration as the original dialogue.

The timing stage will therefore need to handle:

* Speech duration
* Pauses
* Sentence boundaries
* Word timing
* Scene timing
* Speaker overlaps
* Natural speaking rate

The goal is to fit the generated dialogue into the original dialogue windows without making the speech sound unnaturally compressed or stretched.

---

# Source Separation

The original audio often contains:

```text
Dialogue
Music
Ambience
Sound effects
```

The dubbing system should remove or suppress the original dialogue while preserving the other elements.

Conceptually:

```text
Original Mix
     │
     ▼
Source Separation
     │
     ├── Dialogue
     └── Background
           ├── Music
           ├── Ambience
           └── SFX
```

The generated dialogue is then mixed back into the preserved background.

---

# Audio Mixing

The final mix should match the original production as closely as possible.

This includes:

* Dialogue level
* Background level
* Loudness
* Dynamic range
* Stereo positioning
* Transitions
* Room ambience
* Scene changes

The goal is for the dubbed dialogue to feel like part of the original soundtrack rather than an audio track pasted over the video.

---

# Final Rendering

The final stage will replace the original audio track with the generated mix.

Conceptually:

```text
Original Video
      │
      ├── Original Video Stream
      │
      └── Original Audio
               │
               ▼
        Dubbing Pipeline
               │
               ▼
          Final Mix
               │
               ▼
      ┌────────┴────────┐
      │                 │
Original Video       Dubbed Audio
      │                 │
      └────────┬────────┘
               ▼
          Final Video
```

Output will be written to:

```text
output/
```

---

# GPU Profiles

The application detects the available NVIDIA GPU and selects a hardware profile.

Example:

```text
GPU: NVIDIA GeForce RTX 2060
VRAM: 6 GB
Compute: 7.5
Profile: rtx2060
```

Future profiles may include:

```text
rtx2060
rtx4070ti
```

The profile can determine:

* Model selection
* Quantization
* Compute type
* Batch sizes
* GPU memory limits
* CPU/GPU allocation
* Processing strategy

---

# Model Memory Management

Because the RTX 2060 has 6 GB of VRAM, the complete pipeline should not assume that every AI model can remain loaded simultaneously.

The eventual architecture should support sequential model loading:

```text
Load Whisper
    ↓
Transcribe
    ↓
Unload Whisper
    ↓
Load Diarization
    ↓
Diarize
    ↓
Unload Diarization
    ↓
Load Translation
    ↓
Translate
    ↓
Unload Translation
    ↓
Load TTS
    ↓
Generate voices
    ↓
Unload TTS
```

The RTX 4070 Ti profile can use more aggressive configurations where useful.

---

# Model Initialization

A dedicated initialization stage will eventually verify and download all required models.

For example:

```text
models/
├── whisper/
├── diarization/
├── translation/
├── separation/
└── tts/
```

Models should be:

* Downloaded once
* Cached locally
* Reused between runs
* Excluded from Git
* Version/configuration tracked separately

---

# Git Repository

The repository contains source code and configuration only.

Large generated files should not be committed.

The following directories are intended to remain outside Git:

```text
input/
work/
models/
output/
```

except for `.gitkeep` files used to preserve the directory structure.

---

# Development Philosophy

The workflow is being built incrementally.

Each stage should:

1. Work independently.
2. Have a clear input and output.
3. Be testable inside Docker.
4. Preserve intermediate files where useful.
5. Avoid unnecessary coupling between AI models.
6. Support local execution.
7. Respect available GPU memory.
8. Be reproducible.
9. Be benchmarkable.
10. Be replaceable as better local models become available.

The project should favor **measurable pipeline stages over one large opaque application**.

---

# Current Development Target

The immediate development path is:

```text
1. Docker / CUDA                 ✅
2. GPU detection                 ✅
3. FFmpeg / FFprobe              ✅
4. Video inspection              ✅
5. Audio extraction              ✅
6. Faster-Whisper                ✅
7. Speaker diarization           ← NEXT
8. Speaker profiles
9. Local translation
10. Local voice cloning / TTS
11. Timing alignment
12. Source separation
13. Audio mixing
14. Final rendering
15. Model initialization
16. GPU optimization
17. Production-quality refinement
```

---

# Requirements

Basic requirements:

* Windows
* Docker Desktop
* WSL2
* NVIDIA GPU
* NVIDIA drivers supporting Docker GPU access
* Sufficient disk space for AI models and intermediate media

Recommended:

* RTX 2060 6 GB or better
* SSD storage
* Additional RAM for CPU-based processing stages

---

# License

License information will be added as the project develops.

Individual AI models and third-party dependencies may have their own licenses and usage restrictions. Always review the license of each model before using it commercially.
