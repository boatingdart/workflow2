import os
from pathlib import Path

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline


WHISPER_MODEL_DIR = Path("/models/whisper")
PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"


def initialize_whisper():
    model_name = os.getenv("WHISPER_MODEL", "large-v3")

    WHISPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Initializing Whisper...")
    print(f"Model: {model_name}")
    print(f"Cache: {WHISPER_MODEL_DIR}")

    WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        download_root=str(WHISPER_MODEL_DIR),
    )

    print("Whisper model: READY")


def initialize_pyannote():
    print()
    print("Initializing pyannote...")
    print(f"Model: {PYANNOTE_MODEL}")

    token = os.getenv("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is required to initialize the pyannote model."
        )

    Pipeline.from_pretrained(
        PYANNOTE_MODEL,
        token=token,
    )

    print("pyannote model: READY")


def main():
    print("=" * 60)
    print("MODEL INITIALIZATION")
    print("=" * 60)
    print()

    initialize_whisper()
    initialize_pyannote()

    print()
    print("=" * 60)
    print("ALL MODELS: READY")
    print("=" * 60)


if __name__ == "__main__":
    main()
