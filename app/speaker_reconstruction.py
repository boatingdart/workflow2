import json
from pathlib import Path
from typing import List, Dict, Any, Optional


DIARIZATION_FILE = Path("/work/refined_diarization.json")
CHANGES_FILE = Path("/work/speaker_change_candidates.json")
TRANSCRIPT_FILE = Path("/work/transcript.json")
OUTPUT_FILE = Path("/work/reconstructed_speakers.json")

# Small gaps below this value are treated as part of the same dialogue turn.
MAX_INTERNAL_GAP = 1.20

# Very small diarization fragments can be absorbed into a neighboring turn.
MIN_FRAGMENT_DURATION = 0.20


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def overlap(a_start: float, a_end: float,
            b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def duration(start: float, end: float) -> float:
    return max(0.0, end - start)


def get_words(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    words = []

    for segment in transcript.get("segments", []):
        for word in segment.get("words", []):
            if "start" not in word or "end" not in word:
                continue

            words.append({
                "start": float(word["start"]),
                "end": float(word["end"]),
                "word": word.get("word", ""),
                "probability": word.get("probability"),
                "speaker": word.get("speaker"),
            })

    words.sort(key=lambda x: x["start"])
    return words


def speaker_at_time(
    start: float,
    end: float,
    diarization: List[Dict[str, Any]]
) -> Optional[str]:

    scores = {}

    for seg in diarization:
        ov = overlap(
            start,
            end,
            float(seg["start"]),
            float(seg["end"])
        )

        if ov <= 0:
            continue

        speaker = seg["speaker"]
        scores[speaker] = scores.get(speaker, 0.0) + ov

    if not scores:
        return None

    return max(scores, key=scores.get)


def strongest_speaker(
    start: float,
    end: float,
    diarization: List[Dict[str, Any]]
) -> Optional[str]:

    scores = {}

    for seg in diarization:
        ov = overlap(
            start,
            end,
            float(seg["start"]),
            float(seg["end"])
        )

        if ov <= 0:
            continue

        speaker = seg["speaker"]

        # Weight longer overlaps more strongly.
        ratio = ov / max(
            duration(float(seg["start"]), float(seg["end"])),
            0.001
        )

        scores[speaker] = scores.get(speaker, 0.0) + ov * (0.5 + ratio)

    if not scores:
        return None

    return max(scores, key=scores.get)


def assign_words_to_speakers(
    words: List[Dict[str, Any]],
    diarization: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    result = []

    for word in words:

        start = word["start"]
        end = word["end"]

        speaker = strongest_speaker(
            start,
            end,
            diarization
        )

        item = dict(word)
        item["assigned_speaker"] = speaker

        result.append(item)

    return result


def make_word_groups(
    words: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    groups = []

    current = None

    for word in words:

        speaker = word["assigned_speaker"]

        if current is None:
            current = {
                "speaker": speaker,
                "start": word["start"],
                "end": word["end"],
                "words": [word],
            }
            continue

        gap = word["start"] - current["end"]

        # Unknown words are kept separate initially.
        if speaker is None:
            groups.append(current)
            current = {
                "speaker": None,
                "start": word["start"],
                "end": word["end"],
                "words": [word],
            }
            continue

        # Speaker changed.
        if speaker != current["speaker"]:

            groups.append(current)

            current = {
                "speaker": speaker,
                "start": word["start"],
                "end": word["end"],
                "words": [word],
            }

            continue

        # Same speaker, normal continuation.
        if gap <= MAX_INTERNAL_GAP:
            current["end"] = word["end"]
            current["words"].append(word)
        else:
            groups.append(current)

            current = {
                "speaker": speaker,
                "start": word["start"],
                "end": word["end"],
                "words": [word],
            }

    if current is not None:
        groups.append(current)

    return groups


def text_from_words(words: List[Dict[str, Any]]) -> str:
    return "".join(w.get("word", "") for w in words)


def refine_boundaries(
    groups: List[Dict[str, Any]],
    diarization: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    result = []

    for group in groups:

        start = group["start"]
        end = group["end"]
        speaker = group["speaker"]

        # If Whisper speaker assignment is unavailable,
        # fall back to pyannote.
        if speaker is None:
            speaker = strongest_speaker(
                start,
                end,
                diarization
            )

        result.append({
            "speaker": speaker,
            "start": start,
            "end": end,
            "duration": duration(start, end),
            "text": text_from_words(group["words"]),
            "words": group["words"],
        })

    return result


def merge_only_safe_continuations(
    turns: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    if not turns:
        return []

    result = [turns[0]]

    for turn in turns[1:]:

        previous = result[-1]

        same_speaker = (
            previous["speaker"] is not None
            and turn["speaker"] is not None
            and previous["speaker"] == turn["speaker"]
        )

        gap = turn["start"] - previous["end"]

        # Only merge if:
        # 1. same speaker
        # 2. reasonably small gap
        # 3. neither side is a tiny isolated fragment
        if (
            same_speaker
            and gap <= MAX_INTERNAL_GAP
            and previous["duration"] >= MIN_FRAGMENT_DURATION
            and turn["duration"] >= MIN_FRAGMENT_DURATION
        ):
            previous["end"] = turn["end"]
            previous["duration"] = duration(
                previous["start"],
                previous["end"]
            )
            previous["text"] += turn["text"]
            previous["words"].extend(turn["words"])

        else:
            result.append(turn)

    return result


def build_boundaries(
    turns: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    boundaries = []

    for i in range(1, len(turns)):

        previous = turns[i - 1]
        current = turns[i]

        boundary = current["start"]

        gap = max(
            0.0,
            current["start"] - previous["end"]
        )

        speaker_change = (
            previous["speaker"] != current["speaker"]
        )

        if gap >= 0.75:
            boundary_type = "STRONG_SILENCE"
        elif speaker_change:
            boundary_type = "SPEAKER_CHANGE"
        else:
            boundary_type = "CONTINUATION"

        boundaries.append({
            "time": round(boundary, 3),
            "from": previous["speaker"],
            "to": current["speaker"],
            "gap": round(gap, 3),
            "type": boundary_type,
        })

    return boundaries


def main():

    print("=" * 60)
    print("SPEAKER RECONSTRUCTION - BOUNDARY FIRST")
    print("=" * 60)

    print(f"Diarization: {DIARIZATION_FILE}")
    print(f"Changes:     {CHANGES_FILE}")
    print(f"Transcript:  {TRANSCRIPT_FILE}")
    print(f"Output:      {OUTPUT_FILE}")
    print()

    diarization_data = load_json(DIARIZATION_FILE)

    # Changes are optional for this stage.
    if CHANGES_FILE.exists():
        changes_data = load_json(CHANGES_FILE)
    else:
        changes_data = {"changes": []}

    transcript_data = load_json(TRANSCRIPT_FILE)

    diarization = diarization_data.get("segments", [])
    words = get_words(transcript_data)

    print(f"Diarization segments: {len(diarization)}")
    print(f"Whisper words:        {len(words)}")
    print(
        f"Change candidates:    "
        f"{len(changes_data.get('changes', []))}"
    )
    print()

    print("Assigning transcript words to diarization speakers...")

    assigned_words = assign_words_to_speakers(
        words,
        diarization
    )

    print("Building speaker turns...")

    groups = make_word_groups(assigned_words)

    turns = refine_boundaries(
        groups,
        diarization
    )

    turns = merge_only_safe_continuations(turns)

    boundaries = build_boundaries(turns)

    # Remove internal diagnostic fields from words.
    clean_turns = []

    for turn in turns:

        clean_words = []

        for word in turn["words"]:

            clean_word = {
                "start": word["start"],
                "end": word["end"],
                "word": word["word"],
            }

            if word.get("probability") is not None:
                clean_word["probability"] = word["probability"]

            clean_words.append(clean_word)

        clean_turns.append({
            "speaker": turn["speaker"],
            "start": round(turn["start"], 3),
            "end": round(turn["end"], 3),
            "duration": round(turn["duration"], 3),
            "text": turn["text"],
            "words": clean_words,
        })

    output = {
        "audio": diarization_data.get("audio"),
        "transcription_model": transcript_data.get(
            "transcription_model",
            "large-v3"
        ),
        "diarization_model": diarization_data.get(
            "model"
        ),
        "method": "boundary_first_reconstruction",
        "speaker_count": len(
            set(
                t["speaker"]
                for t in clean_turns
                if t["speaker"] is not None
            )
        ),
        "turns": clean_turns,
        "boundaries": boundaries,
    }

    print()
    print("=" * 100)
    print("RECONSTRUCTED SPEAKER TURNS")
    print("=" * 100)

    print(
        f"{'TIME':<15}"
        f"{'DURATION':<11}"
        f"{'SPEAKER':<18}"
        f"TEXT"
    )

    print("-" * 100)

    for turn in clean_turns:

        print(
            f"{turn['start']:6.2f} - "
            f"{turn['end']:6.2f}    "
            f"{turn['duration']:5.2f}s     "
            f"{str(turn['speaker']):<18}"
            f"{turn['text']}"
        )

    print()
    print("BOUNDARY ANALYSIS")
    print("-" * 100)

    for boundary in boundaries:

        print(
            f"{boundary['time']:6.2f}s  "
            f"{str(boundary['from']):<14} -> "
            f"{str(boundary['to']):<14} "
            f"gap={boundary['gap']:.2f}s  "
            f"{boundary['type']}"
        )

    print()
    print("STATISTICS")
    print("-" * 60)

    print(f"turns               : {len(clean_turns)}")
    print(f"boundaries          : {len(boundaries)}")
    print(
        "speaker changes     : "
        f"{sum(1 for b in boundaries if b['type'] == 'SPEAKER_CHANGE')}"
    )
    print(
        "strong silence      : "
        f"{sum(1 for b in boundaries if b['type'] == 'STRONG_SILENCE')}"
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print()
    print("=" * 60)
    print("SPEAKER RECONSTRUCTION: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()