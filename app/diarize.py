import json
import os
from pathlib import Path

from pyannote.audio import Pipeline


AUDIO_FILE = Path("/work/speech_audio.wav")
OUTPUT_FILE = Path("/work/diarization.json")

MODEL_NAME = "pyannote/speaker-diarization-community-1"


# ------------------------------------------------------------------
# Speaker count configuration
# ------------------------------------------------------------------
#
# Supported values:
#
#   SPEAKER_COUNT=auto
#   SPEAKER_COUNT=4
#   SPEAKER_COUNT=5
#
# Automatic mode is used when the variable is not set.
#
# Examples:
#
#   SPEAKER_COUNT=auto
#   SPEAKER_COUNT=4
#   SPEAKER_COUNT=5
#
# ------------------------------------------------------------------

SPEAKER_COUNT = os.environ.get(
    "SPEAKER_COUNT",
    "auto",
).strip().lower()


def parse_speaker_count():
    """
    Convert SPEAKER_COUNT into a pyannote-compatible configuration.

    Returns:
        None for automatic speaker estimation.
        Integer for an explicitly requested number of speakers.
    """

    if SPEAKER_COUNT in {
        "",
        "auto",
        "automatic",
        "none",
    }:
        return None

    try:
        count = int(SPEAKER_COUNT)
    except ValueError:
        raise ValueError(
            "Invalid SPEAKER_COUNT value: "
            f"{SPEAKER_COUNT!r}. "
            "Use 'auto', '4', or '5'."
        )

    if count not in {4, 5}:
        raise ValueError(
            f"Unsupported SPEAKER_COUNT: {count}. "
            "This workflow currently supports "
            "automatic, 4, or 5 speakers."
        )

    return count


def load_pipeline(token):
    """
    Load the pyannote diarization pipeline.
    """

    print("Loading diarization pipeline...")

    pipeline = Pipeline.from_pretrained(
        MODEL_NAME,
        token=token,
    )

    print("Pipeline loaded.")

    return pipeline


def run_diarization(
    pipeline,
    speaker_count,
):
    """
    Run diarization using either automatic speaker estimation
    or an explicitly requested number of speakers.
    """

    print("Running diarization...")
    print()

    if speaker_count is None:
        print(
            "Speaker count: AUTOMATIC"
        )

        output = pipeline(
            str(AUDIO_FILE)
        )

    else:
        print(
            f"Speaker count: FORCED TO {speaker_count}"
        )

        output = pipeline(
            str(AUDIO_FILE),
            num_speakers=speaker_count,
        )

    return output


def convert_to_segments(diarization):
    """
    Convert pyannote diarization output into JSON-friendly
    non-overlapping speaker segments.

    We use exclusive_speaker_diarization because the downstream
    alignment stage works with a single speaker assignment for
    each point in time.
    """

    segments = []

    for turn, _, speaker in diarization.itertracks(
        yield_label=True
    ):
        start = float(turn.start)
        end = float(turn.end)

        if end <= start:
            continue

        segments.append(
            {
                "speaker": speaker,
                "start": round(start, 3),
                "end": round(end, 3),
            }
        )

    return segments


def get_speaker_names(segments):
    """
    Return unique speaker labels.
    """

    return sorted(
        {
            segment["speaker"]
            for segment in segments
            if segment.get("speaker") is not None
        }
    )


def get_speaker_durations(segments):
    """
    Calculate total diarized duration for each speaker.

    This is diagnostic information only.
    """

    durations = {}

    for segment in segments:
        speaker = segment["speaker"]

        duration = (
            segment["end"]
            - segment["start"]
        )

        durations[speaker] = (
            durations.get(speaker, 0.0)
            + duration
        )

    return durations


def save_result(result):
    """
    Save diarization JSON.
    """

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main():
    print()
    print("=" * 60)
    print("SPEAKER DIARIZATION")
    print("=" * 60)

    print(
        f"Input:  {AUDIO_FILE}"
    )
    print(
        f"Output: {OUTPUT_FILE}"
    )
    print(
        f"Model:  {MODEL_NAME}"
    )
    print()

    # --------------------------------------------------------------
    # Validate audio
    # --------------------------------------------------------------

    if not AUDIO_FILE.exists():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_FILE}"
        )

    # --------------------------------------------------------------
    # Parse speaker configuration
    # --------------------------------------------------------------

    speaker_count = parse_speaker_count()

    if speaker_count is None:
        print(
            "Configuration: automatic speaker detection"
        )
    else:
        print(
            f"Configuration: {speaker_count} speakers"
        )

    print()

    # --------------------------------------------------------------
    # Hugging Face token
    # --------------------------------------------------------------

    token = os.environ.get(
        "HF_TOKEN"
    )

    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set."
        )

    # --------------------------------------------------------------
    # Load pipeline
    # --------------------------------------------------------------

    pipeline = load_pipeline(
        token
    )

    print()

    # --------------------------------------------------------------
    # Run diarization
    # --------------------------------------------------------------

    output = run_diarization(
        pipeline,
        speaker_count,
    )

    # --------------------------------------------------------------
    # Select exclusive diarization
    # --------------------------------------------------------------
    #
    # pyannote.audio 4.x returns a DiarizeOutput.
    #
    # exclusive_speaker_diarization gives non-overlapping speaker
    # turns and is therefore easier to align against Whisper words.
    #
    # The underlying diarization output may contain overlapping
    # speech, but this file intentionally stores the exclusive
    # representation for downstream alignment.
    # --------------------------------------------------------------

    diarization = (
        output.exclusive_speaker_diarization
    )

    # --------------------------------------------------------------
    # Convert to JSON
    # --------------------------------------------------------------

    segments = convert_to_segments(
        diarization
    )

    speaker_names = get_speaker_names(
        segments
    )

    speaker_durations = (
        get_speaker_durations(
            segments
        )
    )

    # --------------------------------------------------------------
    # Build result
    # --------------------------------------------------------------

    result = {
        "audio": str(AUDIO_FILE),
        "model": MODEL_NAME,
        "speaker_count_mode": (
            "automatic"
            if speaker_count is None
            else speaker_count
        ),
        "speakers_detected": len(
            speaker_names
        ),
        "segments": segments,
    }

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    save_result(
        result
    )

    # --------------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------------

    print()
    print(
        f"Speaker segments: {len(segments)}"
    )

    print(
        f"Speakers detected: {len(speaker_names)}"
    )

    if speaker_names:
        print()
        print("Speakers:")

        for speaker in speaker_names:
            duration = speaker_durations.get(
                speaker,
                0.0,
            )

            print(
                f"  - {speaker}: "
                f"{duration:.3f}s"
            )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print()
    print("=" * 60)
    print("SPEAKER DIARIZATION: PASS")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()
