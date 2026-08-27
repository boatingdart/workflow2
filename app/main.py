from pathlib import Path

from gpu import print_gpu_info
from media import (
    extract_source_audio,
    extract_speech_audio,
    inspect_video,
    save_metadata,
)
from transcription import transcribe_audio
from diarize import main as run_diarization
from alignment import align
from segment_dialogue import main as run_segmentation
from clean_dialogue import main as run_cleanup
from translate_local import main as run_translation


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
    print()
    print("=" * 60)
    print("VIDEO DUBBING WORKFLOW")
    print("=" * 60)
    print()

    # ---------------------------------------------------------
    # GPU
    # ---------------------------------------------------------

    gpu = print_gpu_info()

    print()
    print(f"Selected GPU profile: {gpu['profile']}")

    # ---------------------------------------------------------
    # Find input video
    # ---------------------------------------------------------

    video = find_input_video()

    print()
    print(f"Input video: {video}")

    # ---------------------------------------------------------
    # Video inspection
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("VIDEO INSPECTION")
    print("=" * 60)

    metadata = inspect_video(video)

    metadata_path = WORK_DIR / "video_metadata.json"

    save_metadata(
        metadata,
        metadata_path,
    )

    print(f"Metadata: {metadata_path}")

    # ---------------------------------------------------------
    # Source audio
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Speech audio
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Whisper transcription
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("SPEECH-TO-TEXT")
    print("=" * 60)

    transcript_path = WORK_DIR / "transcript.json"

    transcribe_audio(
        speech_audio,
        transcript_path,
    )

    # ---------------------------------------------------------
    # Speaker diarization
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("SPEAKER DIARIZATION")
    print("=" * 60)

    run_diarization()

    # ---------------------------------------------------------
    # Word / speaker alignment
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("TRANSCRIPTION / DIARIZATION ALIGNMENT")
    print("=" * 60)

    align()

    # ---------------------------------------------------------
    # Dialogue segmentation
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("DIALOGUE SEGMENTATION")
    print("=" * 60)

    run_segmentation()

    # ---------------------------------------------------------
    # Dialogue cleanup
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("DIALOGUE CLEANUP")
    print("=" * 60)

    run_cleanup()

    # ---------------------------------------------------------
    # Local Japanese → English translation
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("LOCAL TRANSLATION")
    print("=" * 60)

    run_translation()

    # ---------------------------------------------------------
    # Complete
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("VIDEO DUBBING WORKFLOW: PASS")
    print("=" * 60)

    print()
    print("Generated files:")
    print(f"  - {metadata_path}")
    print(f"  - {source_audio}")
    print(f"  - {speech_audio}")
    print(f"  - {WORK_DIR / 'transcript.json'}")
    print(f"  - {WORK_DIR / 'diarization.json'}")
    print(f"  - {WORK_DIR / 'aligned_transcript.json'}")
    print(f"  - {WORK_DIR / 'dialogue.json'}")
    print(f"  - {WORK_DIR / 'clean_dialogue.json'}")
    print(f"  - {WORK_DIR / 'translated_dialogue.json'}")
    print()


if __name__ == "__main__":
    main()
