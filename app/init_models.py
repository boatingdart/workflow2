from pathlib import Path

from faster_whisper import WhisperModel


MODEL_DIR = Path("/models/whisper")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MODEL INITIALIZATION")
    print("=" * 60)
    print()
    print("Downloading Whisper model...")
    print("This may take some time the first time.")
    print()

    WhisperModel(
        "large-v3",
        device="cpu",
        compute_type="int8",
        download_root=str(MODEL_DIR),
    )

    print()
    print("=" * 60)
    print("Whisper model: READY")
    print("=" * 60)


if __name__ == "__main__":
    main()
