import os
import subprocess
from pathlib import Path

from faster_whisper import WhisperModel


INPUT_DIR = Path("/data/input")
OUTPUT_DIR = Path("/data/output")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_COMPUTE_TYPE = os.getenv(
    "WHISPER_COMPUTE_TYPE",
    "float16",
)


def find_video():
    extensions = {".mp4", ".mkv", ".mov", ".avi", ".webm"}

    videos = [
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ]

    return videos[0] if videos else None


def extract_audio(video):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_file = OUTPUT_DIR / f"{video.stem}.wav"

    print("=" * 60)
    print("EXTRACTING AUDIO")
    print("=" * 60)

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(audio_file),
    ]

    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError("FFmpeg audio extraction failed")

    return audio_file


def transcribe(audio_file):
    print("=" * 60)
    print("LOADING WHISPER")
    print("=" * 60)

    print(f"Model: {WHISPER_MODEL}")
    print(f"Compute type: {WHISPER_COMPUTE_TYPE}")

    model = WhisperModel(
        WHISPER_MODEL,
        device="cuda",
        compute_type=WHISPER_COMPUTE_TYPE,
        download_root="/models/whisper",
    )

    print("Model loaded.")
    print()
    print("TRANSCRIBING")
    print("=" * 60)

    segments, info = model.transcribe(
        str(audio_file),
        beam_size=5,
    )

    print(f"Detected language: {info.language}")
    print(f"Language probability: {info.language_probability:.2f}")
    print()

    transcript_file = OUTPUT_DIR / "transcript.txt"

    with transcript_file.open("w", encoding="utf-8") as f:
        for segment in segments:
            line = (
                f"[{segment.start:8.2f} → "
                f"{segment.end:8.2f}] "
                f"{segment.text.strip()}"
            )

            print(line)
            f.write(line + "\n")

    return transcript_file


def main():
    video = find_video()

    if video is None:
        print(f"No video found in {INPUT_DIR}")
        return

    print(f"Input video: {video}")

    audio_file = extract_audio(video)

    print(f"Audio: {audio_file}")

    transcript_file = transcribe(audio_file)

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Transcript: {transcript_file}")


if __name__ == "__main__":
    main()
