import json
from pathlib import Path


INPUT_PATH = Path("/work/clean_dialogue.json")
OUTPUT_PATH = Path("/work/translation_units.json")

# Segments from the same speaker can be combined when the gap
# is short enough that they are likely part of the same thought.
MAX_GAP = 1.50

# Don't let a translation unit become excessively large.
MAX_CHARS = 120


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


def can_merge(current, following):
    """
    Decide whether two dialogue segments should be presented
    to the translation model as one translation unit.

    Important:
    - Never cross speaker boundaries.
    - Never modify the Japanese text.
    - Keep the original segment IDs.
    """

    if current["speaker"] != following["speaker"]:
        return False

    gap = following["start"] - current["end"]

    if gap < 0:
        gap = 0

    if gap > MAX_GAP:
        return False

    combined_length = (
        len(current["text"])
        + len(following["text"])
    )

    if combined_length > MAX_CHARS:
        return False

    return True


def build_translation_units(segments):
    """
    Combine nearby dialogue segments from the same speaker
    into translation units.

    The original dialogue segments are preserved inside
    each translation unit.
    """

    if not segments:
        return []

    units = []

    current = {
        "speaker": segments[0]["speaker"],
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "text": segments[0]["text"],
        "segment_ids": [segments[0]["id"]],
        "segments": [segments[0]],
    }

    for following in segments[1:]:
        if can_merge(current, following):
            current["end"] = following["end"]
            current["text"] += following["text"]
            current["segment_ids"].append(
                following["id"]
            )
            current["segments"].append(following)

        else:
            units.append(current)

            current = {
                "speaker": following["speaker"],
                "start": following["start"],
                "end": following["end"],
                "text": following["text"],
                "segment_ids": [following["id"]],
                "segments": [following],
            }

    units.append(current)

    return units


def main():
    print()
    print("=" * 60)
    print("TRANSLATION UNIT BUILDING")
    print("=" * 60)

    print(f"Input:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    data = load_json(INPUT_PATH)

    segments = data.get("segments", [])

    if not segments:
        raise ValueError(
            "No dialogue segments found."
        )

    print(
        f"Dialogue segments: {len(segments)}"
    )

    units = build_translation_units(
        segments
    )

    result_units = []

    for index, unit in enumerate(units):
        result_units.append(
            {
                "id": index,
                "speaker": unit["speaker"],
                "start": round(
                    unit["start"],
                    3,
                ),
                "end": round(
                    unit["end"],
                    3,
                ),
                "text": unit["text"],
                "segment_ids": unit["segment_ids"],
                "segments": unit["segments"],
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
        "language": data.get(
            "language",
            "ja",
        ),
        "segments": result_units,
    }

    save_json(
        OUTPUT_PATH,
        result,
    )

    print(
        f"Translation units: {len(result_units)}"
    )

    print()
    print("Units:")

    for unit in result_units:
        print(
            f"  [{unit['id']}] "
            f"{unit['speaker']} "
            f"{unit['start']:.2f}-"
            f"{unit['end']:.2f}: "
            f"{unit['text']}"
        )

    print()
    print(f"Saved: {OUTPUT_PATH}")

    print()
    print("=" * 60)
    print("TRANSLATION UNIT BUILDING: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
