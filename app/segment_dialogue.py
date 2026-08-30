import json
from pathlib import Path


INPUT_PATH = Path("/work/aligned_transcript.json")
OUTPUT_PATH = Path("/work/dialogue.json")


# ------------------------------------------------------------------
# Dialogue segmentation tuning
# ------------------------------------------------------------------

# A pause this long is normally enough to create a new dialogue unit.
PAUSE_THRESHOLD = 0.85

# Very small pauses can safely remain inside the same utterance.
SHORT_GAP = 0.25

# Prevent excessively long dubbing units.
MAX_DIALOGUE_DURATION = 7.0

# A speaker run shorter than this can be considered a possible
# accidental diarization fragment.
MIN_SPEAKER_RUN_CHARS = 2


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


def flatten_aligned_words(aligned):
    """
    Recover the word stream from aligned_transcript.json.
    """

    words = []

    for segment in aligned.get(
        "segments",
        [],
    ):
        for word in segment.get(
            "words",
            [],
        ):
            if (
                "start" not in word
                or "end" not in word
            ):
                continue

            text = str(
                word.get(
                    "word",
                    "",
                )
            ).strip()

            if not text:
                continue

            words.append(
                {
                    "start": float(
                        word["start"]
                    ),
                    "end": float(
                        word["end"]
                    ),
                    "word": text,
                    "speaker": word.get(
                        "speaker"
                    ),
                }
            )

    words.sort(
        key=lambda item: (
            item["start"],
            item["end"],
        )
    )

    return words


def merge_word_stream_fragments(words):
    """
    Resolve remaining tiny unknown-speaker fragments.

    We only merge when the evidence is strong. This function does
    not modify Japanese text.
    """

    if not words:
        return []

    result = [
        dict(word)
        for word in words
    ]

    for index, word in enumerate(result):
        if word["speaker"] is not None:
            continue

        previous = (
            result[index - 1]
            if index > 0
            else None
        )

        following = (
            result[index + 1]
            if index + 1 < len(result)
            else None
        )

        previous_speaker = (
            previous["speaker"]
            if previous
            else None
        )

        following_speaker = (
            following["speaker"]
            if following
            else None
        )

        if (
            previous_speaker is not None
            and previous_speaker
            == following_speaker
        ):
            before_gap = (
                word["start"]
                - previous["end"]
            )

            after_gap = (
                following["start"]
                - word["end"]
            )

            if (
                before_gap <= 0.60
                and after_gap <= 0.60
            ):
                word["speaker"] = (
                    previous_speaker
                )

    return result


def append_word(
    current,
    word,
):
    """
    Add a word to the current dialogue unit.
    """

    if current is None:
        return {
            "speaker": word["speaker"],
            "start": word["start"],
            "end": word["end"],
            "words": [word["word"]],
        }

    current["end"] = word["end"]
    current["words"].append(
        word["word"]
    )

    return current


def finalize_current(current):
    """
    Convert internal dialogue representation into JSON output.
    """

    if current is None:
        return None

    text = "".join(
        current["words"]
    ).strip()

    if not text:
        return None

    return {
        "speaker": current["speaker"],
        "start": round(
            current["start"],
            3,
        ),
        "end": round(
            current["end"],
            3,
        ),
        "text": text,
    }


def build_dialogue(words):
    """
    Build natural dialogue units from the smoothed word stream.

    Boundaries are caused by:

    1. Strong speaker changes.
    2. Long pauses.
    3. Maximum dialogue duration.

    We deliberately do NOT use every Whisper segment boundary.
    """

    dialogue = []

    current = None

    for word in words:
        if current is None:
            current = append_word(
                None,
                word,
            )
            continue

        gap = (
            word["start"]
            - current["end"]
        )

        speaker_changed = (
            word["speaker"]
            != current["speaker"]
        )

        duration = (
            word["end"]
            - current["start"]
        )

        # ----------------------------------------------------------
        # Speaker change
        # ----------------------------------------------------------

        if speaker_changed:
            finished = finalize_current(
                current
            )

            if finished is not None:
                dialogue.append(
                    finished
                )

            current = append_word(
                None,
                word,
            )

            continue

        # ----------------------------------------------------------
        # Long pause
        # ----------------------------------------------------------

        if gap >= PAUSE_THRESHOLD:
            finished = finalize_current(
                current
            )

            if finished is not None:
                dialogue.append(
                    finished
                )

            current = append_word(
                None,
                word,
            )

            continue

        # ----------------------------------------------------------
        # Maximum dialogue duration
        # ----------------------------------------------------------

        if duration >= MAX_DIALOGUE_DURATION:
            finished = finalize_current(
                current
            )

            if finished is not None:
                dialogue.append(
                    finished
                )

            current = append_word(
                None,
                word,
            )

            continue

        # ----------------------------------------------------------
        # Normal continuation
        # ----------------------------------------------------------

        current = append_word(
            current,
            word,
        )

    finished = finalize_current(
        current
    )

    if finished is not None:
        dialogue.append(
            finished
        )

    return dialogue


def merge_tiny_same_speaker_lines(
    dialogue
):
    """
    Merge tiny same-speaker dialogue lines when there is only
    a short gap between them.

    This is intentionally much more conservative than the old
    implementation.
    """

    if not dialogue:
        return []

    result = [
        dict(segment)
        for segment in dialogue
    ]

    changed = True

    while changed:
        changed = False

        i = 0

        while i + 1 < len(result):
            current = result[i]
            following = result[i + 1]

            same_speaker = (
                current["speaker"]
                == following["speaker"]
            )

            gap = (
                following["start"]
                - current["end"]
            )

            tiny = (
                len(current["text"])
                <= MIN_SPEAKER_RUN_CHARS
            )

            if (
                same_speaker
                and tiny
                and gap <= SHORT_GAP
            ):
                current["end"] = (
                    following["end"]
                )

                current["text"] = (
                    current["text"]
                    + following["text"]
                )

                result.pop(i + 1)

                changed = True
                continue

            i += 1

    return result


def main():
    print()
    print("=" * 60)
    print("DIALOGUE SEGMENTATION")
    print("=" * 60)

    print(
        f"Input:  {INPUT_PATH}"
    )
    print(
        f"Output: {OUTPUT_PATH}"
    )
    print()

    aligned = load_json(
        INPUT_PATH
    )

    words = flatten_aligned_words(
        aligned
    )

    if not words:
        raise ValueError(
            "No aligned words found."
        )

    print(
        f"Aligned words: {len(words)}"
    )

    words = merge_word_stream_fragments(
        words
    )

    dialogue = build_dialogue(
        words
    )

    dialogue = (
        merge_tiny_same_speaker_lines(
            dialogue
        )
    )

    result = {
        "audio": aligned.get(
            "audio"
        ),
        "transcription_model": aligned.get(
            "transcription_model"
        ),
        "diarization_model": aligned.get(
            "diarization_model"
        ),
        "language": aligned.get(
            "language"
        ),
        "segments": dialogue,
    }

    save_json(
        OUTPUT_PATH,
        result,
    )

    speakers = sorted(
        {
            segment["speaker"]
            for segment in dialogue
            if segment["speaker"] is not None
        }
    )

    print(
        f"Dialogue lines: {len(dialogue)}"
    )
    print(
        f"Speakers:       {len(speakers)}"
    )

    for speaker in speakers:
        print(
            f"  - {speaker}"
        )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print()
    print("=" * 60)
    print("DIALOGUE SEGMENTATION: PASS")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()