from pathlib import Path

from gpu import print_gpu_info
from media import inspect_video, save_metadata


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

def main():
    print("Starting video dubbing workflow...")

    gpu = print_gpu_info()

    print()
    print(f"Selected GPU profile: {gpu['profile']}")

    videos = [
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not videos:
        print()
        print("No input video found.")
        print("Place a video in the input directory.")
        return

    video = videos[0]

    print()
    print("=" * 60)
    print("VIDEO INSPECTION")
    print("=" * 60)
    print(f"Input: {video}")

    metadata = inspect_video(video)

    output_path = WORK_DIR / "video_metadata.json"

    save_metadata(
        metadata,
        output_path,
    )

    print(f"Metadata: {output_path}")

    print()
    print("VIDEO INSPECTION: PASS")


if __name__ == "__main__":
    main()
