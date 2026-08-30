import json
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio

from pyannote.audio import Inference, Model


# ================================================================
# CONFIGURATION
# ================================================================

AUDIO_FILE = Path("/work/speech_audio.wav")

DIARIZATION_FILE = Path(
    "/work/refined_diarization.json"
)

OUTPUT_FILE = Path(
    "/work/speaker_change_candidates.json"
)

# Hugging Face model used to create speaker embeddings.
#
# This is NOT the diarization model.
#
# It is used to compare voice characteristics between nearby
# pieces of audio.
EMBEDDING_MODEL = (
    "pyannote/wespeaker-voxceleb-resnet34-LM"
)

# Length of the analysis window in seconds.
#
# Smaller windows detect shorter speaker changes but produce
# less reliable embeddings.
WINDOW_SIZE = 1.5

# Distance between consecutive windows.
WINDOW_STEP = 0.5

# Minimum duration of audio required for an embedding.
MIN_AUDIO_DURATION = 0.7

# A cosine distance above this value is considered a possible
# speaker change.
#
# This is intentionally conservative. We don't want the first
# version to split every emotional change, breath, or noise event.
SPEAKER_CHANGE_THRESHOLD = 0.32

# Do not report changes closer together than this.
MIN_CHANGE_DISTANCE = 1.0


# ================================================================
# JSON
# ================================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ================================================================
# AUDIO
# ================================================================

def load_audio():
    print("Loading audio...")

    waveform, sample_rate = torchaudio.load(
        str(AUDIO_FILE)
    )

    # Convert stereo to mono.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(
            dim=0,
            keepdim=True,
        )

    # Speaker embedding models generally expect 16 kHz.
    if sample_rate != 16000:
        print(
            f"Resampling audio: "
            f"{sample_rate} Hz -> 16000 Hz"
        )

        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            16000,
        )

        sample_rate = 16000

    waveform = waveform.squeeze(0)

    duration = (
        waveform.shape[0]
        / sample_rate
    )

    print(
        f"Audio duration: {duration:.3f}s"
    )

    print(
        f"Sample rate:    {sample_rate} Hz"
    )

    print()

    return waveform, sample_rate


# ================================================================
# DIARIZATION
# ================================================================

def load_diarization():
    data = load_json(
        DIARIZATION_FILE
    )

    segments = []

    for segment in data.get(
        "segments",
        [],
    ):
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

        segments.append(
            {
                "speaker": segment.get(
                    "speaker"
                ),
                "start": start,
                "end": end,
            }
        )

    segments.sort(
        key=lambda x: x["start"]
    )

    return segments


# ================================================================
# EMBEDDING MODEL
# ================================================================

def load_embedding_model():
    token = os.environ.get(
        "HF_TOKEN"
    )

    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set."
        )

    print(
        "Loading speaker embedding model..."
    )

    model = Model.from_pretrained(
        EMBEDDING_MODEL,
        token=token,
    )

    inference = Inference(
        model,
        window="whole",
    )

    print(
        "Speaker embedding model loaded."
    )

    print()

    return inference


# ================================================================
# AUDIO WINDOW
# ================================================================

def extract_audio(
    waveform,
    sample_rate,
    start,
    end,
):
    start = max(
        0.0,
        start,
    )

    end = min(
        end,
        waveform.shape[0]
        / sample_rate,
    )

    if end <= start:
        return None

    start_sample = int(
        start * sample_rate
    )

    end_sample = int(
        end * sample_rate
    )

    audio = waveform[
        start_sample:end_sample
    ]

    if (
        audio.numel()
        < int(
            MIN_AUDIO_DURATION
            * sample_rate
        )
    ):
        return None

    return audio


# ================================================================
# TEMPORARY WAV
# ================================================================

def create_temp_wav(
    audio,
    sample_rate,
    path,
):
    torchaudio.save(
        str(path),
        audio.unsqueeze(0),
        sample_rate,
    )


# ================================================================
# COSINE DISTANCE
# ================================================================

def cosine_distance(
    embedding_a,
    embedding_b,
):
    a = np.asarray(
        embedding_a,
        dtype=np.float32,
    ).reshape(-1)

    b = np.asarray(
        embedding_b,
        dtype=np.float32,
    ).reshape(-1)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 1.0

    similarity = (
        np.dot(a, b)
        / denominator
    )

    similarity = float(
        np.clip(
            similarity,
            -1.0,
            1.0,
        )
    )

    return 1.0 - similarity


# ================================================================
# EMBEDDING
# ================================================================

def get_embedding(
    inference,
    audio,
    sample_rate,
    temp_path,
):
    create_temp_wav(
        audio,
        sample_rate,
        temp_path,
    )

    try:
        embedding = inference(
            str(temp_path)
        )

        if hasattr(
            embedding,
            "data",
        ):
            embedding = embedding.data

        return np.asarray(
            embedding,
            dtype=np.float32,
        )

    finally:
        if temp_path.exists():
            temp_path.unlink()


# ================================================================
# ANALYSIS WINDOWS
# ================================================================

def build_windows(
    segments
):
    """
    Build fine-grained analysis windows.

    We analyze only audio inside pyannote speech regions.
    """

    windows = []

    for segment in segments:

        start = segment["start"]
        end = segment["end"]

        duration = end - start

        if duration < MIN_AUDIO_DURATION:
            continue

        current = start

        while current < end:

            window_end = min(
                current + WINDOW_SIZE,
                end,
            )

            if (
                window_end - current
                >= MIN_AUDIO_DURATION
            ):
                windows.append(
                    {
                        "start": current,
                        "end": window_end,
                        "pyannote_speaker":
                            segment["speaker"],
                    }
                )

            current += WINDOW_STEP

    return windows


# ================================================================
# WINDOW EMBEDDINGS
# ================================================================

def calculate_window_embeddings(
    windows,
    waveform,
    sample_rate,
    inference,
):
    print(
        f"Analysis windows: {len(windows)}"
    )

    print()

    results = []

    temp_path = Path(
        "/tmp/speaker_embedding.wav"
    )

    for index, window in enumerate(
        windows,
        start=1,
    ):

        print(
            f"Embedding "
            f"{index}/{len(windows)} "
            f"{window['start']:.2f}-"
            f"{window['end']:.2f}s",
            end="\r",
            flush=True,
        )

        audio = extract_audio(
            waveform,
            sample_rate,
            window["start"],
            window["end"],
        )

        if audio is None:
            continue

        embedding = get_embedding(
            inference,
            audio,
            sample_rate,
            temp_path,
        )

        results.append(
            {
                **window,
                "embedding": embedding,
            }
        )

    print()
    print()

    return results


# ================================================================
# FIND SPEAKER CHANGES
# ================================================================

def find_changes(
    windows
):
    """
    Compare adjacent voice windows.

    A large embedding distance is a candidate speaker change.

    IMPORTANT:
    This does not declare the change as definitely correct.
    It produces evidence for the next refinement stage.
    """

    candidates = []

    for index in range(
        1,
        len(windows),
    ):

        previous = windows[
            index - 1
        ]

        current = windows[
            index
        ]

        # Only compare temporally adjacent windows.
        gap = (
            current["start"]
            - previous["end"]
        )

        if gap > 1.0:
            continue

        distance = cosine_distance(
            previous["embedding"],
            current["embedding"],
        )

        # Pyannote says the same speaker but the voice embedding
        # says they are substantially different.
        pyannote_same = (
            previous["pyannote_speaker"]
            == current["pyannote_speaker"]
        )

        if (
            distance
            >= SPEAKER_CHANGE_THRESHOLD
        ):
            change_time = (
                previous["end"]
                + current["start"]
            ) / 2.0

            candidates.append(
                {
                    "time": round(
                        change_time,
                        3,
                    ),
                    "distance": round(
                        distance,
                        4,
                    ),
                    "previous_window": {
                        "start": round(
                            previous["start"],
                            3,
                        ),
                        "end": round(
                            previous["end"],
                            3,
                        ),
                    },
                    "current_window": {
                        "start": round(
                            current["start"],
                            3,
                        ),
                        "end": round(
                            current["end"],
                            3,
                        ),
                    },
                    "pyannote_previous":
                        previous[
                            "pyannote_speaker"
                        ],
                    "pyannote_current":
                        current[
                            "pyannote_speaker"
                        ],
                    "pyannote_same_speaker":
                        pyannote_same,
                }
            )

    # ------------------------------------------------------------
    # Merge nearby candidates.
    # ------------------------------------------------------------

    candidates.sort(
        key=lambda x: x["time"]
    )

    filtered = []

    for candidate in candidates:

        if not filtered:
            filtered.append(
                candidate
            )
            continue

        previous = filtered[-1]

        if (
            candidate["time"]
            - previous["time"]
            < MIN_CHANGE_DISTANCE
        ):
            # Keep the stronger candidate.
            if (
                candidate["distance"]
                > previous["distance"]
            ):
                filtered[-1] = candidate

        else:
            filtered.append(
                candidate
            )

    return filtered


# ================================================================
# PRINT CANDIDATES
# ================================================================

def print_candidates(
    candidates
):
    print()
    print("=" * 90)
    print("SPEAKER CHANGE CANDIDATES")
    print("=" * 90)

    if not candidates:
        print(
            "No strong speaker-change candidates found."
        )
        print()

        return

    print(
        f"{'TIME':<10}"
        f"{'DISTANCE':<12}"
        f"{'PYANNOTE':<25}"
        f"{'TYPE'}"
    )

    print("-" * 90)

    for candidate in candidates:

        previous = (
            candidate[
                "pyannote_previous"
            ]
        )

        current = (
            candidate[
                "pyannote_current"
            ]
        )

        if (
            previous == current
        ):
            change_type = (
                "HIDDEN CHANGE"
            )
        else:
            change_type = (
                "NORMAL CHANGE"
            )

        print(
            f"{candidate['time']:7.2f}s  "
            f"{candidate['distance']:<12.4f}"
            f"{previous} -> {current:<10}"
            f"{change_type}"
        )

    print()


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 60)
    print("FINE SPEAKER CHANGE DETECTION")
    print("=" * 60)

    print(
        f"Audio:       {AUDIO_FILE}"
    )

    print(
        f"Diarization: {DIARIZATION_FILE}"
    )

    print(
        f"Output:      {OUTPUT_FILE}"
    )

    print(
        f"Embedding:   {EMBEDDING_MODEL}"
    )

    print(
        f"Window:      {WINDOW_SIZE}s"
    )

    print(
        f"Step:        {WINDOW_STEP}s"
    )

    print(
        f"Threshold:   {SPEAKER_CHANGE_THRESHOLD}"
    )

    print()

    # ------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------

    if not AUDIO_FILE.exists():
        raise FileNotFoundError(
            f"Audio file not found: "
            f"{AUDIO_FILE}"
        )

    if not DIARIZATION_FILE.exists():
        raise FileNotFoundError(
            f"Diarization file not found: "
            f"{DIARIZATION_FILE}"
        )

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------

    waveform, sample_rate = load_audio()

    segments = load_diarization()

    if not segments:
        raise RuntimeError(
            "No diarization segments found."
        )

    print(
        f"Diarization segments: "
        f"{len(segments)}"
    )

    print()

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    inference = load_embedding_model()

    # ------------------------------------------------------------
    # Windows
    # ------------------------------------------------------------

    windows = build_windows(
        segments
    )

    if not windows:
        raise RuntimeError(
            "No suitable audio windows found."
        )

    # ------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------

    windows = calculate_window_embeddings(
        windows,
        waveform,
        sample_rate,
        inference,
    )

    if len(windows) < 2:
        raise RuntimeError(
            "Not enough embeddings for "
            "speaker-change detection."
        )

    # ------------------------------------------------------------
    # Detect changes
    # ------------------------------------------------------------

    candidates = find_changes(
        windows
    )

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    print_candidates(
        candidates
    )

    # ------------------------------------------------------------
    # Remove embeddings from JSON.
    #
    # Embeddings are large and are not needed downstream.
    # ------------------------------------------------------------

    output_windows = []

    for window in windows:

        output_windows.append(
            {
                "start": round(
                    window["start"],
                    3,
                ),
                "end": round(
                    window["end"],
                    3,
                ),
                "pyannote_speaker":
                    window[
                        "pyannote_speaker"
                    ],
            }
        )

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    result = {
        "audio": str(AUDIO_FILE),
        "diarization": str(
            DIARIZATION_FILE
        ),
        "embedding_model":
            EMBEDDING_MODEL,
        "window_size":
            WINDOW_SIZE,
        "window_step":
            WINDOW_STEP,
        "speaker_change_threshold":
            SPEAKER_CHANGE_THRESHOLD,
        "analysis_windows":
            output_windows,
        "speaker_change_candidates":
            candidates,
    }

    save_json(
        OUTPUT_FILE,
        result,
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print()
    print("=" * 60)
    print("SPEAKER CHANGE DETECTION: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
