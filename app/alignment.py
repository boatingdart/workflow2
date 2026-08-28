import json
from pathlib import Path


TRANSCRIPTION_PATH = Path("/work/transcript.json")
DIARIZATION_PATH = Path("/work/diarization.json")
OUTPUT_PATH = Path("/work/aligned_transcript.json")


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def overlap(start_a, end_a, start_b, end_b):
    """
    Return the duration of overlap between two time ranges.
    """
    return max(
        0.0,
        min(end_a, end_b) - max(start_a, start_b),
    )


def find_speaker(word_start, word_end, diarization_segments):
    """
    Assign a word to the speaker with the greatest temporal overlap.

    Also return overlap information so we can inspect borderline
    assignments and improve the alignment algorithm later.
    """

    word_duration = max(
        0.0,
        word_end - word_start,
    )

    best_speaker = None
    best_overlap = 0.0

    for segment in diarization_segments:
        segment_start = segment["start"]
        segment_end = segment["end"]

        current_overlap = overlap(
            word_start,
            word_end,
            segment_start,
            segment_end,
        )

        if current_overlap > best_overlap:
            best_overlap = current_overlap
            best_speaker = segment["speaker"]

    overlap_ratio = (
        best_overlap / word_duration
        if word_duration > 0
        else 0.0
    )

    return (
        best_speaker,
        round(best_overlap, 3),
        round(overlap_ratio, 3),
    )

def align():
    print()
    print("=" * 60)
    print("TRANSCRIPTION / DIARIZATION ALIGNMENT")
    print("=" * 60)

    print(f"Transcription: {TRANSCRIPTION_PATH}")
    print(f"Diarization:   {DIARIZATION_PATH}")
    print(f"Output:        {OUTPUT_PATH}")
    print()

    transcription = load_json(TRANSCRIPTION_PATH)
    diarization = load_json(DIARIZATION_PATH)

    diarization_segments = diarization.get("segments", [])

    if not diarization_segments:
        raise ValueError(
            "No diarization segments found."
        )

    transcription_segments = transcription.get(
        "segments",
        [],
    )

    if not transcription_segments:
        raise ValueError(
            "No transcription segments found."
        )

    aligned_segments = []

    for segment_id, segment in enumerate(
        transcription_segments
    ):
        words = []

        for word in segment.get("words", []):
            word_start = word["start"]
            word_end = word["end"]

            (
                speaker,
                speaker_overlap,
                speaker_overlap_ratio,
            ) = find_speaker(
                word_start,
                word_end,
                diarization_segments,
            )

            words.append(
                {
                    "start": word_start,
                    "end": word_end,
                    "word": word["word"],
                    "probability": word.get(
                        "probability"
                    ),
                    "speaker": speaker,
                    "speaker_overlap": speaker_overlap,
                    "speaker_overlap_ratio": (
                        speaker_overlap_ratio
                    ),
                }
            )

        if not words:
            continue

        aligned_segments.append(
            {
                "id": segment.get(
                    "id",
                    segment_id,
                ),
                "start": segment.get(
                    "start",
                    words[0]["start"],
                ),
                "end": segment.get(
                    "end",
                    words[-1]["end"],
                ),
                "text": "".join(
                    word["word"]
                    for word in words
                ),
                "words": words,
            }
        )

    result = {
        "audio": transcription.get("audio"),
        "transcription_model": transcription.get(
            "model"
        ),
        "diarization_model": diarization.get(
            "model"
        ),
        "language": transcription.get(
            "language"
        ),
        "segments": aligned_segments,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    aligned_words = sum(
        len(segment["words"])
        for segment in aligned_segments
    )

    speakers = sorted(
        {
            word["speaker"]
            for segment in aligned_segments
            for word in segment["words"]
            if word["speaker"] is not None
        }
    )

    print(
        f"Words aligned:    {aligned_words}"
    )
    print(
        f"Output segments:  {len(aligned_segments)}"
    )
    print(
        f"Speakers:         {len(speakers)}"
    )

    for speaker in speakers:
        print(f"  - {speaker}")

    print()
    print(f"Saved: {OUTPUT_PATH}")

    print()
    print("=" * 60)
    print("ALIGNMENT: PASS")
    print("=" * 60)

    return result


if __name__ == "__main__":
    align()
