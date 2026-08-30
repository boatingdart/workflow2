import json
from pathlib import Path

from pyannote.audio import Pipeline


# ================================================================
# CONFIGURATION
# ================================================================

AUDIO_FILE = Path("/work/speech_audio.wav")
DIARIZATION_FILE = Path("/work/diarization.json")
OUTPUT_FILE = Path("/work/refined_diarization.json")

MODEL_NAME = "pyannote/speaker-diarization-community-1"

# Set to:
#
#   0 = automatic
#   4 = force 4 speakers
#   5 = force 5 speakers
#
# For the current test clip we use 5.
SPEAKER_COUNT = 5


# ================================================================
# JSON HELPERS
# ================================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ================================================================
# LOAD PIPELINE
# ================================================================

def load_pipeline():
    token = __import__("os").environ.get("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set."
        )

    print("Loading diarization pipeline...")

    pipeline = Pipeline.from_pretrained(
        MODEL_NAME,
        token=token,
    )

    print("Pipeline loaded.")
    print()

    return pipeline


# ================================================================
# RUN DIARIZATION
# ================================================================

def run_diarization(pipeline):
    print("Running diarization...")
    print()

    if SPEAKER_COUNT == 0:
        print("Speaker count: AUTOMATIC")

        output = pipeline(
            str(AUDIO_FILE)
        )

    else:
        print(
            f"Speaker count: FORCED TO {SPEAKER_COUNT}"
        )

        output = pipeline(
            str(AUDIO_FILE),
            num_speakers=SPEAKER_COUNT,
        )

    print()

    return output


# ================================================================
# CONVERT PYANNOTE OUTPUT
# ================================================================

def extract_segments(output):
    """
    Extract non-overlapping speaker turns.

    pyannote.audio 4.x returns a DiarizeOutput.
    """

    diarization = output.exclusive_speaker_diarization

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

    segments.sort(
        key=lambda x: (
            x["start"],
            x["end"],
        )
    )

    return segments


# ================================================================
# MERGE ADJACENT SEGMENTS
# ================================================================

def merge_adjacent_segments(segments):
    """
    Merge consecutive segments belonging to the same speaker.

    This removes unnecessary fragmentation from pyannote.
    """

    if not segments:
        return []

    result = []

    for segment in segments:

        if not result:
            result.append(dict(segment))
            continue

        previous = result[-1]

        if (
            previous["speaker"]
            == segment["speaker"]
            and segment["start"]
            - previous["end"]
            <= 0.50
        ):
            previous["end"] = segment["end"]

        else:
            result.append(dict(segment))

    return result


# ================================================================
# REMOVE VERY SHORT FRAGMENTS
# ================================================================

def remove_tiny_fragments(segments):
    """
    Remove extremely short diarization fragments.

    We do NOT remove them blindly.

    A tiny segment is retained when it represents a genuine
    isolated speaker turn. This function only removes fragments
    shorter than 50 ms when they are completely surrounded by
    the same speaker.
    """

    if len(segments) < 3:
        return segments

    result = []

    for index, segment in enumerate(segments):

        duration = (
            segment["end"]
            - segment["start"]
        )

        if duration >= 0.05:
            result.append(segment)
            continue

        previous = segments[index - 1]
        following = segments[index + 1]

        if (
            previous["speaker"]
            == following["speaker"]
            and previous["speaker"]
            != segment["speaker"]
        ):
            previous_copy = dict(previous)

            if result:
                result[-1]["end"] = following["end"]

            # Skip tiny fragment.
            continue

        result.append(segment)

    return result


# ================================================================
# BUILD SPEAKER STATISTICS
# ================================================================

def speaker_statistics(segments):
    statistics = {}

    for segment in segments:

        speaker = segment["speaker"]

        duration = (
            segment["end"]
            - segment["start"]
        )

        if speaker not in statistics:
            statistics[speaker] = {
                "segments": 0,
                "duration": 0.0,
            }

        statistics[speaker]["segments"] += 1
        statistics[speaker]["duration"] += duration

    return statistics


# ================================================================
# PRINT SEGMENTS
# ================================================================

def print_segments(segments):
    print()
    print("=" * 80)
    print("REFINED SPEAKER SEGMENTS")
    print("=" * 80)

    print(
        f"{'TIME':<18}"
        f"{'SPEAKER':<18}"
        f"{'DURATION':<12}"
    )

    print("-" * 80)

    for segment in segments:

        duration = (
            segment["end"]
            - segment["start"]
        )

        print(
            f"{segment['start']:6.2f} - "
            f"{segment['end']:6.2f}    "
            f"{segment['speaker']:<18}"
            f"{duration:6.2f}s"
        )

    print()


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 60)
    print("SPEAKER REFINEMENT")
    print("=" * 60)

    print(
        f"Audio:       {AUDIO_FILE}"
    )

    print(
        f"Input:       {DIARIZATION_FILE}"
    )

    print(
        f"Output:      {OUTPUT_FILE}"
    )

    print(
        f"Model:       {MODEL_NAME}"
    )

    print()

    # ------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------

    if not AUDIO_FILE.exists():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_FILE}"
        )

    # ------------------------------------------------------------
    # Load existing diarization
    # ------------------------------------------------------------

    existing = load_json(
        DIARIZATION_FILE
    )

    print(
        "Existing diarization:"
    )

    print(
        f"  Speakers detected: "
        f"{existing.get('speakers_detected', '?')}"
    )

    print(
        f"  Segments: "
        f"{len(existing.get('segments', []))}"
    )

    print()

    # ------------------------------------------------------------
    # Load pyannote
    # ------------------------------------------------------------

    pipeline = load_pipeline()

    # ------------------------------------------------------------
    # Run fresh diarization
    # ------------------------------------------------------------

    output = run_diarization(
        pipeline
    )

    segments = extract_segments(
        output
    )

    print(
        f"Raw segments: {len(segments)}"
    )

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------

    segments = merge_adjacent_segments(
        segments
    )

    segments = remove_tiny_fragments(
        segments
    )

    print(
        f"Refined segments: {len(segments)}"
    )

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------

    statistics = speaker_statistics(
        segments
    )

    speakers = sorted(
        statistics.keys()
    )

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    result = {
        "audio": str(AUDIO_FILE),
        "model": MODEL_NAME,
        "speaker_count_mode": (
            "automatic"
            if SPEAKER_COUNT == 0
            else SPEAKER_COUNT
        ),
        "speakers_detected": len(
            speakers
        ),
        "segments": segments,
        "speaker_statistics": {
            speaker: {
                "segments": data["segments"],
                "duration": round(
                    data["duration"],
                    3,
                ),
            }
            for speaker, data
            in statistics.items()
        },
    }

    save_json(
        OUTPUT_FILE,
        result,
    )

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    print_segments(
        segments
    )

    print(
        "=" * 60
    )

    print(
        f"Speakers detected: "
        f"{len(speakers)}"
    )

    print()

    for speaker in speakers:

        data = statistics[speaker]

        print(
            f"  {speaker}: "
            f"{data['duration']:.3f}s "
            f"({data['segments']} segments)"
        )

    print()

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print()

    print("=" * 60)
    print("SPEAKER REFINEMENT: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
