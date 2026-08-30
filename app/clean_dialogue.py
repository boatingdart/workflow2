import json
from pathlib import Path


INPUT_PATH = Path("/work/dialogue.json")
OUTPUT_PATH = Path("/work/clean_dialogue.json")


# ------------------------------------------------------------------
# Cleanup tuning
# ------------------------------------------------------------------

# Only merge a tiny fragment when the same speaker continues almost
# immediately afterwards.
TINY_FRAGMENT_MAX_CHARS = 2
TINY_FRAGMENT_MAX_GAP = 0.30

# Do not merge dialogue units that have a substantial pause.
MAX_MERGE_GAP = 0.30


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


def normalize_segments(segments):
    """
    Validate and normalize dialogue segments.

    No linguistic rewriting is performed.
    """

    result = []

    for segment in segments:
        text = str(
            segment.get(
                "text",
                "",
            )
        ).strip()

        if not text:
            continue

        speaker = segment.get(
            "speaker"
        )

        try:
            start = float(
                segment["start"]
            )
            end = float(
                segment["end"]
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        if end <= start:
            continue

        result.append(
            {
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text,
            }
        )

    result.sort(
        key=lambda item: (
            item["start"],
            item["end"],
        )
    )

    return result


def merge_tiny_fragments(
    segments
):
    """
    Merge only genuinely tiny fragments.

    Example:

        SPEAKER_00  "あ"
        SPEAKER_00  "あの"

    with a 0.1 second gap

    can become:

        SPEAKER_00  "ああの"

    But two normal dialogue lines are never merged merely because
    they happen to have the same speaker.
    """

    if not segments:
        return []

    result = []

    for current in segments:
        if not result:
            result.append(
                dict(current)
            )
            continue

        previous = result[-1]

        same_speaker = (
            previous["speaker"]
            == current["speaker"]
        )

        gap = (
            current["start"]
            - previous["end"]
        )

        previous_is_tiny = (
            len(previous["text"])
            <= TINY_FRAGMENT_MAX_CHARS
        )

        current_is_tiny = (
            len(current["text"])
            <= TINY_FRAGMENT_MAX_CHARS
        )

        # Only merge if one side is clearly a tiny fragment.
        tiny_fragment = (
            previous_is_tiny
            or current_is_tiny
        )

        if (
            same_speaker
            and tiny_fragment
            and gap >= 0
            and gap <= MAX_MERGE_GAP
        ):
            previous["end"] = (
                current["end"]
            )

            previous["text"] = (
                previous["text"]
                + current["text"]
            )

        else:
            result.append(
                dict(current)
            )

    return result


def remove_duplicate_segments(
    segments
):
    """
    Remove exact duplicate entries created by upstream processing.

    Only exact duplicates are removed. Similar-looking dialogue is
    never treated as duplicate.
    """

    if not segments:
        return []

    result = []

    previous_key = None

    for segment in segments:
        key = (
            segment["speaker"],
            segment["start"],
            segment["end"],
            segment["text"],
        )

        if key == previous_key:
            continue

        result.append(segment)
        previous_key = key

    return result


def normalize_overlapping_segments(
    segments
):
    """
    Protect against tiny accidental timestamp overlaps.

    We do not alter normal timing. If two consecutive segments from
    the same speaker overlap by only a few milliseconds, the second
    segment starts at the previous end.

    Larger overlaps are left untouched because changing them could
    hide a genuine diarization problem.
    """

    if not segments:
        return []

    result = [
        dict(segment)
        for segment in segments
    ]

    for index in range(
        1,
        len(result),
    ):
        previous = result[index - 1]
        current = result[index]

        if (
            previous["speaker"]
            != current["speaker"]
        ):
            continue

        overlap = (
            previous["end"]
            - current["start"]
        )

        if (
            overlap > 0
            and overlap <= 0.05
        ):
            current["start"] = (
                previous["end"]
            )

            if (
                current["end"]
                <= current["start"]
            ):
                current["start"] = (
                    previous["start"]
                )

    return result


def add_ids(segments):
    """
    Assign clean sequential IDs.
    """

    result = []

    for index, segment in enumerate(
        segments
    ):
        result.append(
            {
                "id": index,
                "speaker": segment[
                    "speaker"
                ],
                "start": round(
                    segment["start"],
                    3,
                ),
                "end": round(
                    segment["end"],
                    3,
                ),
                "text": segment[
                    "text"
                ],
            }
        )

    return result


def clean_dialogue(
    input_path=INPUT_PATH,
    output_path=OUTPUT_PATH,
):
    print()
    print("=" * 60)
    print("DIALOGUE CLEANUP")
    print("=" * 60)

    print(
        f"Input:  {input_path}"
    )
    print(
        f"Output: {output_path}"
    )
    print()

    data = load_json(
        input_path
    )

    segments = data.get(
        "segments",
        [],
    )

    print(
        f"Input dialogue lines: {len(segments)}"
    )

    # --------------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------------

    segments = normalize_segments(
        segments
    )

    after_normalization = len(
        segments
    )

    # --------------------------------------------------------------
    # Remove exact duplicate entries
    # --------------------------------------------------------------

    segments = remove_duplicate_segments(
        segments
    )

    after_duplicates = len(
        segments
    )

    # --------------------------------------------------------------
    # Repair only tiny same-speaker timestamp overlaps
    # --------------------------------------------------------------

    segments = (
        normalize_overlapping_segments(
            segments
        )
    )

    # --------------------------------------------------------------
    # Merge genuinely tiny fragments
    # --------------------------------------------------------------

    segments = merge_tiny_fragments(
        segments
    )

    after_merge = len(
        segments
    )

    # --------------------------------------------------------------
    # IDs
    # --------------------------------------------------------------

    segments = add_ids(
        segments
    )

    speakers = sorted(
        {
            segment["speaker"]
            for segment in segments
            if segment["speaker"] is not None
        }
    )

    result = {
        "audio": data.get(
            "audio"
        ),
        "transcription_model": data.get(
            "transcription_model"
        ),
        "diarization_model": data.get(
            "diarization_model"
        ),
        "language": data.get(
            "language"
        ),
        "segments": segments,
    }

    save_json(
        output_path,
        result,
    )

    print(
        f"After normalization: {after_normalization}"
    )
    print(
        f"After duplicates:    {after_duplicates}"
    )
    print(
        f"After tiny merging:  {after_merge}"
    )
    print(
        f"Speakers:             {len(speakers)}"
    )

    for speaker in speakers:
        print(
            f"  - {speaker}"
        )

    print()
    print(
        f"Saved: {output_path}"
    )

    print()
    print("=" * 60)
    print("DIALOGUE CLEANUP: PASS")
    print("=" * 60)

    return result


def main():
    clean_dialogue()


if __name__ == "__main__":
    main()