import json
from pathlib import Path


INPUT_PATH = Path("/work/aligned_transcript.json")
OUTPUT_PATH = Path("/work/dialogue.json")

# Pause larger than this starts a new dialogue line.
PAUSE_THRESHOLD = 0.70

# Very small gaps can safely be kept inside the same line.
SHORT_GAP = 0.25


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


def normalize_words(words):
    """
    Remove unusable word entries while preserving the original
    recognized text and timestamps.
    """

    result = []

    for word in words:
        text = word.get("word")

        if text is None:
            continue

        text = str(text).strip()

        if not text:
            continue

        start = word.get("start")
        end = word.get("end")

        if start is None or end is None:
            continue

        result.append(
            {
                "start": float(start),
                "end": float(end),
                "word": text,
                "probability": word.get("probability"),
                "speaker": word.get("speaker"),
            }
        )

    return result


def build_dialogue(aligned):
    """
    Convert word-level speaker alignment into dialogue lines.

    Speaker changes are always hard boundaries.
    Long pauses create boundaries.
    """

    dialogue = []
    current = None

    for segment in aligned.get("segments", []):
        words = normalize_words(segment.get("words", []))

        for word in words:
            speaker = word.get("speaker")

            # Keep unknown-speaker words rather than throwing them away.
            if current is None:
                current = {
                    "speaker": speaker,
                    "start": word["start"],
                    "end": word["end"],
                    "words": [word["word"]],
                }
                continue

            previous_end = current["end"]
            gap = word["start"] - previous_end

            speaker_changed = speaker != current["speaker"]
            long_pause = gap >= PAUSE_THRESHOLD

            # Speaker changes are always boundaries.
            if speaker_changed or long_pause:
                dialogue.append(
                    {
                        "speaker": current["speaker"],
                        "start": round(current["start"], 3),
                        "end": round(current["end"], 3),
                        "text": "".join(current["words"]).strip(),
                    }
                )

                current = {
                    "speaker": speaker,
                    "start": word["start"],
                    "end": word["end"],
                    "words": [word["word"]],
                }

                continue

            current["words"].append(word["word"])
            current["end"] = word["end"]

    if current is not None:
        dialogue.append(
            {
                "speaker": current["speaker"],
                "start": round(current["start"], 3),
                "end": round(current["end"], 3),
                "text": "".join(current["words"]).strip(),
            }
        )

    return dialogue


def merge_tiny_fragments(dialogue):
    """
    Merge extremely short fragments when they belong to the same
    speaker and are separated by only a tiny gap.

    This avoids producing unnecessary one-character dialogue lines.
    """

    if not dialogue:
        return dialogue

    result = [dialogue[0]]

    for current in dialogue[1:]:
        previous = result[-1]

        gap = current["start"] - previous["end"]

        same_speaker = (
            current["speaker"] == previous["speaker"]
        )

        previous_is_tiny = (
            len(previous["text"]) <= 2
        )

        if (
            same_speaker
            and previous_is_tiny
            and gap <= SHORT_GAP
        ):
            previous["end"] = current["end"]
            previous["text"] += current["text"]
        else:
            result.append(current)

    return result


def main():
    print()
    print("=" * 60)
    print("DIALOGUE SEGMENTATION")
    print("=" * 60)

    print(f"Input:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    aligned = load_json(INPUT_PATH)

    print("Building dialogue lines...")

    dialogue = build_dialogue(aligned)
    dialogue = merge_tiny_fragments(dialogue)

    result = {
        "audio": aligned.get("audio"),
        "transcription_model": aligned.get(
            "transcription_model"
        ),
        "diarization_model": aligned.get(
            "diarization_model"
        ),
        "language": aligned.get("language"),
        "segments": dialogue,
    }

    save_json(OUTPUT_PATH, result)

    speakers = sorted(
        {
            segment["speaker"]
            for segment in dialogue
            if segment["speaker"] is not None
        }
    )

    print(f"Dialogue lines: {len(dialogue)}")
    print(f"Speakers:       {len(speakers)}")

    for speaker in speakers:
        print(f"  - {speaker}")

    print()
    print(f"Saved: {OUTPUT_PATH}")

    print()
    print("=" * 60)
    print("DIALOGUE SEGMENTATION: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()