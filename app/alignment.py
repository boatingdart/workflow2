import json
from pathlib import Path


TRANSCRIPTION_PATH = Path("/work/transcript.json")
DIARIZATION_PATH = Path("/work/diarization.json")
OUTPUT_PATH = Path("/work/alignment_diagnostic.json")


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


def overlap(start_a, end_a, start_b, end_b):
    return max(
        0.0,
        min(end_a, end_b) - max(start_a, start_b),
    )


def flatten_words(transcription):
    words = []

    for segment in transcription.get("segments", []):
        for word in segment.get("words", []):
            if "start" not in word or "end" not in word:
                continue

            text = str(word.get("word", "")).strip()

            if not text:
                continue

            try:
                start = float(word["start"])
                end = float(word["end"])
            except (TypeError, ValueError):
                continue

            if end <= start:
                continue

            words.append(
                {
                    "start": start,
                    "end": end,
                    "word": text,
                    "probability": word.get("probability"),
                }
            )

    words.sort(key=lambda x: (x["start"], x["end"]))

    return words


def normalize_diarization(diarization):
    segments = []

    for segment in diarization.get("segments", []):
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue

        speaker = segment.get("speaker")

        if not speaker:
            continue

        if end <= start:
            continue

        segments.append(
            {
                "speaker": speaker,
                "start": start,
                "end": end,
            }
        )

    segments.sort(key=lambda x: (x["start"], x["end"]))

    return segments


def find_speakers_for_word(word, diarization_segments):
    word_start = word["start"]
    word_end = word["end"]
    word_duration = word_end - word_start

    evidence = []

    for segment in diarization_segments:
        amount = overlap(
            word_start,
            word_end,
            segment["start"],
            segment["end"],
        )

        if amount <= 0:
            continue

        ratio = amount / word_duration

        evidence.append(
            {
                "speaker": segment["speaker"],
                "overlap": round(amount, 3),
                "ratio": round(ratio, 3),
                "diarization_start": segment["start"],
                "diarization_end": segment["end"],
            }
        )

    evidence.sort(
        key=lambda x: x["overlap"],
        reverse=True,
    )

    return evidence


def build_diagnostic_words(words, diarization_segments):
    result = []

    for word in words:
        evidence = find_speakers_for_word(
            word,
            diarization_segments,
        )

        best_speaker = None
        best_overlap = 0.0
        best_ratio = 0.0

        if evidence:
            best = evidence[0]

            best_speaker = best["speaker"]
            best_overlap = best["overlap"]
            best_ratio = best["ratio"]

        result.append(
            {
                "start": word["start"],
                "end": word["end"],
                "word": word["word"],
                "probability": word["probability"],
                "best_speaker": best_speaker,
                "best_overlap": best_overlap,
                "best_overlap_ratio": best_ratio,
                "speaker_evidence": evidence,
            }
        )

    return result


def build_diagnostic_segments(words):
    """
    Group consecutive words according to the raw best speaker.

    IMPORTANT:
    There is NO smoothing here.
    """

    segments = []

    for word in words:
        speaker = word["best_speaker"]

        if (
            segments
            and segments[-1]["speaker"] == speaker
        ):
            segments[-1]["end"] = word["end"]
            segments[-1]["text"] += word["word"]
            segments[-1]["words"].append(word)

        else:
            segments.append(
                {
                    "speaker": speaker,
                    "start": word["start"],
                    "end": word["end"],
                    "text": word["word"],
                    "words": [word],
                }
            )

    return segments


def print_diagnostic(words, diarization_segments):
    print()
    print("=" * 80)
    print("RAW WHISPER / PYANNOTE ALIGNMENT")
    print("=" * 80)
    print()

    print(
        f"{'TIME':<15}"
        f"{'WORD':<12}"
        f"{'SPEAKER':<15}"
        f"{'OVERLAP':<10}"
        f"{'RATIO':<8}"
    )

    print("-" * 80)

    for word in words:
        time_text = (
            f"{word['start']:.2f}-"
            f"{word['end']:.2f}"
        )

        speaker = (
            word["best_speaker"]
            if word["best_speaker"] is not None
            else "UNKNOWN"
        )

        print(
            f"{time_text:<15}"
            f"{word['word']:<12}"
            f"{speaker:<15}"
            f"{word['best_overlap']:<10.3f}"
            f"{word['best_overlap_ratio']:<8.3f}"
        )

    print()
    print("=" * 80)
    print("PYANNOTE SEGMENTS")
    print("=" * 80)
    print()

    for segment in diarization_segments:
        print(
            f"{segment['start']:6.3f} - "
            f"{segment['end']:6.3f}  "
            f"{segment['speaker']}"
        )

    print()


def main():
    print()
    print("=" * 60)
    print("SPEAKER ALIGNMENT DIAGNOSTIC")
    print("=" * 60)

    print(f"Transcription: {TRANSCRIPTION_PATH}")
    print(f"Diarization:   {DIARIZATION_PATH}")
    print(f"Output:        {OUTPUT_PATH}")

    print()

    transcription = load_json(
        TRANSCRIPTION_PATH
    )

    diarization = load_json(
        DIARIZATION_PATH
    )

    words = flatten_words(
        transcription
    )

    diarization_segments = normalize_diarization(
        diarization
    )

    if not words:
        raise RuntimeError(
            "No Whisper words found."
        )

    if not diarization_segments:
        raise RuntimeError(
            "No diarization segments found."
        )

    print(
        f"Whisper words:       {len(words)}"
    )

    print(
        f"Diarization segments: {len(diarization_segments)}"
    )

    speakers = sorted(
        {
            segment["speaker"]
            for segment in diarization_segments
        }
    )

    print(
        f"Diarization speakers: {len(speakers)}"
    )

    print()

    diagnostic_words = build_diagnostic_words(
        words,
        diarization_segments,
    )

    diagnostic_segments = build_diagnostic_segments(
        diagnostic_words
    )

    result = {
        "audio": transcription.get("audio"),
        "transcription_model": transcription.get("model"),
        "diarization_model": diarization.get("model"),
        "language": transcription.get("language"),
        "speaker_count_mode": diarization.get(
            "speaker_count_mode"
        ),
        "speakers_detected": diarization.get(
            "speakers_detected"
        ),
        "diarization_segments": diarization_segments,
        "words": diagnostic_words,
        "segments": diagnostic_segments,
    }

    save_json(
        OUTPUT_PATH,
        result,
    )

    print_diagnostic(
        diagnostic_words,
        diarization_segments,
    )

    print("=" * 80)
    print("RAW DIALOGUE SEGMENTS")
    print("=" * 80)
    print()

    for segment in diagnostic_segments:
        speaker = (
            segment["speaker"]
            if segment["speaker"]
            else "UNKNOWN"
        )

        print(
            f"{segment['start']:6.2f} - "
            f"{segment['end']:6.2f} "
            f"{speaker:<15} "
            f"{segment['text']}"
        )

    print()

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print()
    print("=" * 60)
    print("DIAGNOSTIC: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()