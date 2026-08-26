import os
from pathlib import Path
import time

import torch
from faster_whisper import WhisperModel


AUDIO_PATH = Path("/work/speech_audio.wav")

MODEL_SIZE = os.getenv(
    "WHISPER_MODEL",
    "large-v3",
)

COMPUTE_TYPE = os.getenv(
    "WHISPER_COMPUTE_TYPE",
    "int8_float16",
)


def main():
    print("=" * 60)
    print("FASTER-WHISPER GPU TEST")
    print("=" * 60)

    print(f"PyTorch:        {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"GPU:            {torch.cuda.get_device_name(0)}")
    print(f"Model:          {MODEL_SIZE}")
    print(f"Compute type:   {COMPUTE_TYPE}")
    print(f"Audio:          {AUDIO_PATH}")

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_PATH}"
        )

    print()
    print("Loading Whisper model...")

    start = time.perf_counter()

    model = WhisperModel(
        MODEL_SIZE,
        device="cuda",
        compute_type=COMPUTE_TYPE,
        download_root="/models/whisper",
    )

    load_time = time.perf_counter() - start

    print(f"Model loaded in {load_time:.2f} seconds")

    print()
    print("Transcribing...")

    start = time.perf_counter()

    segments, info = model.transcribe(
        str(AUDIO_PATH),
        beam_size=5,
        vad_filter=True,
    )

    segments = list(segments)

    transcription_time = time.perf_counter() - start

    print()
    print("=" * 60)
    print("TRANSCRIPTION")
    print("=" * 60)

    print(f"Detected language: {info.language}")
    print(
        f"Language probability: "
        f"{info.language_probability:.3f}"
    )
    print(f"Segments: {len(segments)}")

    print()

    for segment in segments:
        print(
            f"[{segment.start:8.2f} --> "
            f"{segment.end:8.2f}] "
            f"{segment.text.strip()}"
        )

    print()
    print(
        f"Transcription time: "
        f"{transcription_time:.2f} seconds"
    )

    print()
    print("=" * 60)
    print("FASTER-WHISPER TEST: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
