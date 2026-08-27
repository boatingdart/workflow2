import json
import os
from pathlib import Path

from pyannote.audio import Pipeline


AUDIO_FILE = Path("/work/speech_audio.wav")
OUTPUT_FILE = Path("/work/diarization.json")

MODEL_NAME = "pyannote/speaker-diarization-community-1"


def main():
    print("=" * 60)
    print("SPEAKER DIARIZATION")
    print("=" * 60)
    print(f"Input:  {AUDIO_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    # ---------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------

    if not AUDIO_FILE.exists():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_FILE}"
        )

    # ---------------------------------------------------------
    # Hugging Face token
    # ---------------------------------------------------------

    token = os.environ.get("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set."
        )

    # ---------------------------------------------------------
    # Load diarization pipeline
    # ---------------------------------------------------------

    print("Loading diarization pipeline...")

    pipeline = Pipeline.from_pretrained(
        MODEL_NAME,
        token=token,
    )

    print("Pipeline loaded.")
    print()

    # ---------------------------------------------------------
    # Run diarization
    # ---------------------------------------------------------

    print("Running diarization...")
    print()

    output = pipeline(str(AUDIO_FILE))

    # pyannote.audio 4.x returns a DiarizeOutput.
    #
    # We use exclusive_speaker_diarization because it contains
    # non-overlapping speaker turns and is better suited for
    # matching speaker turns against Whisper timestamps.
    diarization = output.exclusive_speaker_diarization

    # ---------------------------------------------------------
    # Convert pyannote output to JSON-friendly structure
    # ---------------------------------------------------------

    segments = []

    for turn, _, speaker in diarization.itertracks(
        yield_label=True
    ):
        segments.append(
            {
                "speaker": speaker,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
            }
        )

    # ---------------------------------------------------------
    # Build result
    # ---------------------------------------------------------

    result = {
        "audio": str(AUDIO_FILE),
        "model": MODEL_NAME,
        "segments": segments,
    }

    # ---------------------------------------------------------
    # Save result
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    speaker_names = sorted(
        {
            segment["speaker"]
            for segment in segments
        }
    )

    print(f"Speaker segments: {len(segments)}")
    print(f"Speakers detected: {len(speaker_names)}")

    if speaker_names:
        print()
        print("Speakers:")
        for speaker in speaker_names:
            print(f"  - {speaker}")

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print()

    print("=" * 60)
    print("SPEAKER DIARIZATION: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
