import json
from pathlib import Path


INPUT_PATH = Path("/work/dialogue.json")
OUTPUT_PATH = Path("/work/clean_dialogue.json")

# A long pause is a strong indication that the next words
# belong to a new utterance.
LONG_PAUSE = 1.20

# Very short fragments can be attached to the following
# utterance when they belong to the same speaker.
SHORT_FRAGMENT_LENGTH = 2


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def clean_segments(segments):
    """
    Perform conservative cleanup.

    Important:
    - Do NOT rewrite Japanese.
    - Do NOT invent missing words.
    - Do NOT alter timestamps unnecessarily.
    - Do NOT change speaker assignments.
    """

    if not segments:
        return []

    cleaned = []

    for segment in segments:
        text = str(
            segment.get("text", "")
        ).strip()

        if not text:
            continue

        cleaned.append(
            {
                "speaker": segment.get("speaker"),
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": text,
            }
        )

    return cleaned


def merge_fragments(segments):
    """
    Merge very small transcription fragments when they are likely
    to be pieces of a larger utterance.

    Speaker assignment is treated conservatively:
    - Same-speaker fragments are safe to merge.
    - A tiny fragment immediately before another segment may be
      merged into the following segment when the timing strongly
      suggests it is part of the same utterance.
    """

    if not segments:
        return []

    result = []
    i = 0

    while i < len(segments):
        current = segments[i]

        if i + 1 < len(segments):
            following = segments[i + 1]

            gap = (
                following["start"]
                - current["end"]
            )

            same_speaker = (
                current["speaker"]
                == following["speaker"]
            )

            tiny = (
                len(current["text"])
                <= SHORT_FRAGMENT_LENGTH
            )

            # -------------------------------------------------
            # Same speaker
            # -------------------------------------------------

            if (
                same_speaker
                and tiny
                and gap <= LONG_PAUSE
            ):
                merged = {
                    "speaker": current["speaker"],
                    "start": current["start"],
                    "end": following["end"],
                    "text": (
                        current["text"]
                        + following["text"]
                    ),
                }

                result.append(merged)
                i += 2
                continue

            # -------------------------------------------------
            # Tiny cross-speaker fragment
            # -------------------------------------------------
            #
            # Example:
            #
            #   SPEAKER_00: 綺
            #   SPEAKER_01: 麗ね
            #
            # → 綺麗ね
            #
            # The following segment gets the resulting speaker
            # because the larger portion belongs to that speaker.
            # -------------------------------------------------

            if (
                not same_speaker
                and tiny
                and gap <= 0.10
                and len(following["text"]) >= 2
            ):
                merged = {
                    "speaker": following["speaker"],
                    "start": current["start"],
                    "end": following["end"],
                    "text": (
                        current["text"]
                        + following["text"]
                    ),
                }

                result.append(merged)
                i += 2
                continue

        result.append(current)
        i += 1

    return result


def add_ids(segments):
    """
    Give each clean dialogue line a stable sequential ID.
    """

    result = []

    for index, segment in enumerate(segments):
        result.append(
            {
                "id": index,
                "speaker": segment["speaker"],
                "start": round(
                    segment["start"],
                    3,
                ),
                "end": round(
                    segment["end"],
                    3,
                ),
                "text": segment["text"],
            }
        )

    return result


def clean_dialogue(
    input_path=INPUT_PATH,
    output_path=OUTPUT_PATH,
):
    """
    Clean dialogue.json and write clean_dialogue.json.

    Can be called directly from main.py while still
    supporting standalone execution.
    """

    print()
    print("=" * 60)
    print("DIALOGUE CLEANUP")
    print("=" * 60)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()

    data = load_json(input_path)

    segments = data.get(
        "segments",
        [],
    )

    print(
        f"Input dialogue lines: {len(segments)}"
    )

    segments = clean_segments(segments)

    after_cleanup = len(segments)

    segments = merge_fragments(segments)

    after_merge = len(segments)

    segments = add_ids(segments)

    speakers = sorted(
        {
            segment["speaker"]
            for segment in segments
            if segment["speaker"] is not None
        }
    )

    result = {
        "audio": data.get("audio"),
        "transcription_model": data.get(
            "transcription_model"
        ),
        "diarization_model": data.get(
            "diarization_model"
        ),
        "language": data.get("language"),
        "segments": segments,
    }

    save_json(
        output_path,
        result,
    )

    print(
        f"After cleanup:      {after_cleanup}"
    )
    print(
        f"After merging:      {after_merge}"
    )
    print(
        f"Speakers:            {len(speakers)}"
    )

    for speaker in speakers:
        print(f"  - {speaker}")

    print()
    print(f"Saved: {output_path}")

    print()
    print("=" * 60)
    print("DIALOGUE CLEANUP: PASS")
    print("=" * 60)

    return result


def main():
    clean_dialogue()


if __name__ == "__main__":
    main()
