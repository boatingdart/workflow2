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
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
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
        text = str(segment.get("text", "")).strip()

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
    Merge tiny fragments when they are clearly part of the
    same speaker's nearby utterance.

    We deliberately keep this conservative.
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
                    segment["start"], 3
                ),
                "end": round(
                    segment["end"], 3
                ),
                "text": segment["text"],
            }
        )

    return result


def main():
    print()
    print("=" * 60)
    print("DIALOGUE CLEANUP")
    print("=" * 60)

    print(f"Input:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    data = load_json(INPUT_PATH)

    segments = data.get("segments", [])

    print(
        f"Input dialogue lines: {len(segments)}"
    )

    segments = clean_segments(segments)

    before_merge = len(segments)

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
        OUTPUT_PATH,
        result,
    )

    print(
        f"After cleanup:      {before_merge}"
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
    print(f"Saved: {OUTPUT_PATH}")

    print()
    print("=" * 60)
    print("DIALOGUE CLEANUP: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
