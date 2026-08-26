import json
import subprocess
from pathlib import Path


def run_command(command):
    """Run a command and return stdout."""

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n"
            f"{' '.join(command)}\n\n"
            f"{result.stderr}"
        )

    return result.stdout


def inspect_video(video_path):
    """Return FFprobe metadata for a video."""

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    output = run_command(command)

    return json.loads(output)


def save_metadata(metadata, output_path):
    """Save metadata as formatted JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )


def extract_source_audio(video_path, output_path):
    """
    Extract the original audio as PCM WAV.

    The original channel layout and sample rate are preserved.
    """

    video_path = Path(video_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    run_command(command)


def extract_speech_audio(video_path, output_path):
    """
    Extract audio normalized for speech recognition.

    Output:
        PCM signed 16-bit
        16 kHz
        mono
    """

    video_path = Path(video_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    run_command(command)
