import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio
from pyannote.audio import Model, Inference


# ============================================================
# CONFIGURATION
# ============================================================

AUDIO_FILE = Path("/work/speech_audio.wav")
INPUT_FILE = Path("/work/reconstructed_speakers.json")
OUTPUT_FILE = Path("/work/identified_speakers.json")

EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"

# Cosine similarity threshold.
#
# Higher = more conservative.
# Lower  = more likely to merge speakers.
#
# Start conservative and adjust after seeing the output.
SIMILARITY_THRESHOLD = 0.72

# Very short pieces of audio are unreliable for speaker identity.
MIN_SAMPLE_DURATION = 0.50

# Short unidentified utterances can inherit the surrounding
# speaker when they are sufficiently close in time.
MAX_UNKNOWN_GAP = 1.00


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()

    a = a / (torch.norm(a) + 1e-8)
    b = b / (torch.norm(b) + 1e-8)

    return float(torch.dot(a, b))


def load_audio(path: Path):
    waveform, sample_rate = torchaudio.load(str(path))

    # Convert stereo -> mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Model expects 16 kHz.
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            sample_rate,
            16000
        )
        waveform = resampler(waveform)
        sample_rate = 16000

    return waveform, sample_rate


def extract_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    start: float,
    end: float
) -> torch.Tensor:

    start_sample = max(
        0,
        int(start * sample_rate)
    )

    end_sample = min(
        waveform.shape[1],
        int(end * sample_rate)
    )

    if end_sample <= start_sample:
        return torch.empty((1, 0))

    return waveform[:, start_sample:end_sample]


# ============================================================
# EMBEDDING
# ============================================================

def create_embedding(
    inference: Inference,
    audio: torch.Tensor,
    sample_rate: int
) -> Optional[torch.Tensor]:

    if audio.shape[1] == 0:
        return None

    duration = audio.shape[1] / sample_rate

    if duration < MIN_SAMPLE_DURATION:
        return None

    # pyannote Inference accepts a dictionary containing
    # waveform and sample rate.
    try:
        embedding = inference({
            "waveform": audio,
            "sample_rate": sample_rate
        })

        if hasattr(embedding, "data"):
            embedding = torch.tensor(
                embedding.data,
                dtype=torch.float32
            )

        elif not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(
                embedding,
                dtype=torch.float32
            )

        return embedding.flatten()

    except Exception as e:
        print(
            f"WARNING: embedding failed: {e}"
        )
        return None


# ============================================================
# SPEAKER PROFILE
# ============================================================

def build_speaker_profiles(
    turns: List[Dict],
    waveform: torch.Tensor,
    sample_rate: int,
    inference: Inference
):

    profiles = {}

    for turn in turns:

        speaker = turn.get("speaker")

        if speaker is None:
            continue

        start = float(turn["start"])
        end = float(turn["end"])

        audio = extract_audio(
            waveform,
            sample_rate,
            start,
            end
        )

        embedding = create_embedding(
            inference,
            audio,
            sample_rate
        )

        if embedding is None:
            continue

        if speaker not in profiles:
            profiles[speaker] = []

        profiles[speaker].append({
            "start": start,
            "end": end,
            "duration": end - start,
            "embedding": embedding
        })

    return profiles


# ============================================================
# PROFILE CENTROIDS
# ============================================================

def calculate_centroids(profiles):

    centroids = {}

    for speaker, samples in profiles.items():

        if not samples:
            continue

        embeddings = [
            sample["embedding"]
            for sample in samples
        ]

        matrix = torch.stack(embeddings)

        centroid = matrix.mean(dim=0)

        centroid = centroid / (
            torch.norm(centroid) + 1e-8
        )

        centroids[speaker] = centroid

    return centroids


# ============================================================
# SPEAKER COMPARISON
# ============================================================

def compare_speakers(centroids):

    speakers = list(centroids.keys())

    comparisons = []

    for i in range(len(speakers)):

        for j in range(i + 1, len(speakers)):

            speaker_a = speakers[i]
            speaker_b = speakers[j]

            similarity = cosine_similarity(
                centroids[speaker_a],
                centroids[speaker_b]
            )

            comparisons.append({
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "similarity": round(similarity, 4)
            })

    comparisons.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return comparisons


# ============================================================
# UNION-FIND CLUSTERING
# ============================================================

class UnionFind:

    def __init__(self, items):
        self.parent = {
            item: item
            for item in items
        }

    def find(self, item):

        if self.parent[item] != item:
            self.parent[item] = self.find(
                self.parent[item]
            )

        return self.parent[item]

    def union(self, a, b):

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:
            self.parent[root_b] = root_a


def cluster_speakers(
    centroids,
    threshold
):

    speakers = list(centroids.keys())

    uf = UnionFind(speakers)

    comparisons = compare_speakers(
        centroids
    )

    for comparison in comparisons:

        if comparison["similarity"] >= threshold:

            uf.union(
                comparison["speaker_a"],
                comparison["speaker_b"]
            )

    clusters = {}

    for speaker in speakers:

        root = uf.find(speaker)

        clusters.setdefault(
            root,
            []
        ).append(speaker)

    return list(clusters.values()), comparisons


# ============================================================
# CREATE PERSISTENT NAMES
# ============================================================

def create_identity_map(clusters):

    identity_map = {}

    # Sort clusters by earliest speaker label.
    clusters = sorted(
        clusters,
        key=lambda c: sorted(c)[0]
    )

    for index, cluster in enumerate(clusters):

        identity = f"SPEAKER_{chr(ord('A') + index)}"

        for diarization_speaker in cluster:

            identity_map[
                diarization_speaker
            ] = identity

    return identity_map


# ============================================================
# UNKNOWN SPEAKER ASSIGNMENT
# ============================================================

def assign_unknown_turns(
    turns,
    identity_map
):

    result = []

    for index, turn in enumerate(turns):

        speaker = turn.get("speaker")

        if speaker is not None:

            new_turn = dict(turn)

            new_turn["identity"] = identity_map.get(
                speaker,
                speaker
            )

            result.append(new_turn)

            continue

        # ----------------------------------------------------
        # UNKNOWN TURN
        # ----------------------------------------------------

        best_identity = None
        best_gap = float("inf")

        # Previous turn
        if index > 0:

            previous = turns[index - 1]

            if previous.get("speaker") is not None:

                gap = (
                    turn["start"]
                    - previous["end"]
                )

                if 0 <= gap <= MAX_UNKNOWN_GAP:

                    best_identity = identity_map.get(
                        previous["speaker"]
                    )

                    best_gap = gap

        # Next turn
        if index + 1 < len(turns):

            next_turn = turns[index + 1]

            if next_turn.get("speaker") is not None:

                gap = (
                    next_turn["start"]
                    - turn["end"]
                )

                if 0 <= gap <= MAX_UNKNOWN_GAP:

                    if gap < best_gap:

                        best_identity = identity_map.get(
                            next_turn["speaker"]
                        )

                        best_gap = gap

        new_turn = dict(turn)

        new_turn["identity"] = best_identity

        if best_identity is not None:
            new_turn["identity_inferred"] = True
        else:
            new_turn["identity_inferred"] = False

        result.append(new_turn)

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SPEAKER IDENTITY RECONSTRUCTION")
    print("=" * 70)

    print(f"Audio:       {AUDIO_FILE}")
    print(f"Input:       {INPUT_FILE}")
    print(f"Output:      {OUTPUT_FILE}")
    print(f"Embedding:   {EMBEDDING_MODEL}")
    print(f"Threshold:   {SIMILARITY_THRESHOLD}")
    print()

    data = load_json(INPUT_FILE)

    turns = data.get("turns", [])

    print(
        f"Reconstructed turns: {len(turns)}"
    )

    print()
    print("Loading audio...")

    waveform, sample_rate = load_audio(
        AUDIO_FILE
    )

    duration = waveform.shape[1] / sample_rate

    print(
        f"Audio duration: {duration:.3f}s"
    )

    print(
        f"Sample rate:    {sample_rate} Hz"
    )

    print()
    print("Loading speaker embedding model...")

    model = Model.from_pretrained(
        EMBEDDING_MODEL
    )

    inference = Inference(
        model,
        window="whole"
    )

    print("Speaker embedding model loaded.")

    print()
    print("Building speaker profiles...")

    profiles = build_speaker_profiles(
        turns,
        waveform,
        sample_rate,
        inference
    )

    print()
    print("Speaker profiles:")

    for speaker, samples in profiles.items():

        total_duration = sum(
            sample["duration"]
            for sample in samples
        )

        print(
            f"  {speaker}: "
            f"{len(samples)} samples, "
            f"{total_duration:.2f}s"
        )

    print()
    print("Calculating speaker centroids...")

    centroids = calculate_centroids(
        profiles
    )

    print(
        f"Usable speaker profiles: "
        f"{len(centroids)}"
    )

    print()
    print("=" * 80)
    print("SPEAKER SIMILARITY")
    print("=" * 80)

    clusters, comparisons = cluster_speakers(
        centroids,
        SIMILARITY_THRESHOLD
    )

    for comparison in comparisons:

        print(
            f"{comparison['speaker_a']:<15} "
            f"<-> "
            f"{comparison['speaker_b']:<15} "
            f"{comparison['similarity']:.4f}"
        )

    print()
    print("=" * 80)
    print("SPEAKER CLUSTERS")
    print("=" * 80)

    for index, cluster in enumerate(clusters):

        identity = (
            f"SPEAKER_"
            f"{chr(ord('A') + index)}"
        )

        print(
            f"{identity:<12}: "
            f"{', '.join(sorted(cluster))}"
        )

    identity_map = create_identity_map(
        clusters
    )

    print()
    print("=" * 80)
    print("IDENTITY MAP")
    print("=" * 80)

    for diarization_speaker, identity in sorted(
        identity_map.items()
    ):

        print(
            f"{diarization_speaker:<15} "
            f"-> {identity}"
        )

    print()
    print("Assigning identities to dialogue turns...")

    identified_turns = assign_unknown_turns(
        turns,
        identity_map
    )

    output = dict(data)

    output["identity_model"] = EMBEDDING_MODEL
    output["identity_similarity_threshold"] = (
        SIMILARITY_THRESHOLD
    )

    output["speaker_identity_map"] = identity_map

    output["speaker_clusters"] = [
        {
            "identity": (
                f"SPEAKER_"
                f"{chr(ord('A') + index)}"
            ),
            "pyannote_speakers": sorted(cluster)
        }
        for index, cluster in enumerate(clusters)
    ]

    output["similarity_comparisons"] = comparisons

    output["turns"] = identified_turns

    print()
    print("=" * 100)
    print("IDENTIFIED SPEAKER TURNS")
    print("=" * 100)

    print(
        f"{'TIME':<17}"
        f"{'PYANNOTE':<15}"
        f"{'IDENTITY':<12}"
        f"TEXT"
    )

    print("-" * 100)

    for turn in identified_turns:

        print(
            f"{turn['start']:6.2f} - "
            f"{turn['end']:6.2f}    "
            f"{str(turn.get('speaker')):<15}"
            f"{str(turn.get('identity')):<12}"
            f"{turn.get('text', '')}"
        )

    save_json(
        OUTPUT_FILE,
        output
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")

    print()
    print("=" * 70)
    print("SPEAKER IDENTITY RECONSTRUCTION: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
