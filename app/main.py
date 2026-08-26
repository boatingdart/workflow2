from pathlib import Path

from gpu import print_gpu_info
from media import (
    extract_source_audio,
    extract_speech_audio,
    inspect_video,
    save_metadata,
)


INPUT_DIR = Path("/input")
WORK_DIR = Path("/work")

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".m4v",
}


def find_input_video():
    videos = [
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not videos:
        raise FileNotFoundError(
            "No supported input video found."
        )

    if len(videos) > 1:
        raise RuntimeError(
            "Multiple input videos found. "
            "Please leave only one video in input/."
        )

    return videos[0]


def main():
    print("Starting video dubbing workflow...")

    gpu = print_gpu_info()

    print()
    print(f"Selected GPU profile: {gpu['profile']}")

    video = find_input_video()

    print()
    print("=" * 60)
    print("VIDEO INSPECTION")
    print("=" * 60)
    print(f"Input: {video}")

    metadata = inspect_video(video)

    metadata_path = WORK_DIR / "video_metadata.json"

    save_metadata(
        metadata,
        metadata_path,
    )

    print(f"Metadata: {metadata_path}")

    print()
    print("=" * 60)
    print("SOURCE AUDIO EXTRACTION")
    print("=" * 60)

    source_audio = WORK_DIR / "source_audio.wav"

    extract_source_audio(
        video,
        source_audio,
    )

    print(f"Source audio: {source_audio}")

    print()
    print("=" * 60)
    print("SPEECH AUDIO EXTRACTION")
    print("=" * 60)

    speech_audio = WORK_DIR / "speech_audio.wav"

    extract_speech_audio(
        video,
        speech_audio,
    )

    print(f"Speech audio: {speech_audio}")

    print()
    print("=" * 60)
    print("MEDIA EXTRACTION: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
