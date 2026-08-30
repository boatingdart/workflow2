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

# Persistent speaker database
PROFILE_DIR = Path("/work/speaker_profiles")

EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"

# Conservative matching threshold.
#
# A new speaker is assigned to an existing identity ONLY when
# the similarity reaches this threshold.
#
SIMILARITY_THRESHOLD = 0.72

MIN_SAMPLE_DURATION = 0.50

# Maximum number of embedding samples retained per identity.
#
# Keeping several samples makes the identity more robust to
# changes in emotion, volume, microphone position, etc.
MAX_PROFILE_SAMPLES = 10

# Unknown short utterances can inherit identity from a nearby
# known turn.
MAX_UNKNOWN_GAP = 1.00


# ============================================================
# JSON
# ============================================================

def load_json(path: Path) -> Dict:

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "w", encoding="utf-8") as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# AUDIO
# ============================================================

def load_audio(path: Path):

    waveform, sample_rate = torchaudio.load(
        str(path)
    )

    # Stereo -> mono
    if waveform.shape[0] > 1:

        waveform = waveform.mean(
            dim=0,
            keepdim=True
        )

    # Resample -> 16 kHz
    if sample_rate != 16000:

        resampler = torchaudio.transforms.Resample(
            sample_rate,
            16000
        )

        waveform = resampler(
            waveform
        )

        sample_rate = 16000

    return waveform, sample_rate


def extract_audio(
    waveform,
    sample_rate,
    start,
    end
):

    start_sample = max(
        0,
        int(start * sample_rate)
    )

    end_sample = min(
        waveform.shape[1],
        int(end * sample_rate)
    )

    if end_sample <= start_sample:

        return torch.empty(
            (1, 0)
        )

    return waveform[
        :,
        start_sample:end_sample
    ]


# ============================================================
# EMBEDDING
# ============================================================

def create_embedding(
    inference,
    audio,
    sample_rate
):

    if audio.shape[1] == 0:

        return None

    duration = (
        audio.shape[1]
        /
        sample_rate
    )

    if duration < MIN_SAMPLE_DURATION:

        return None

    try:

        embedding = inference({

            "waveform": audio,

            "sample_rate": sample_rate
        })

        if hasattr(
            embedding,
            "data"
        ):

            embedding = torch.tensor(
                embedding.data,
                dtype=torch.float32
            )

        elif not isinstance(
            embedding,
            torch.Tensor
        ):

            embedding = torch.tensor(
                embedding,
                dtype=torch.float32
            )

        embedding = embedding.flatten()

        # Normalize
        embedding = embedding / (
            torch.norm(embedding)
            +
            1e-8
        )

        return embedding

    except Exception as e:

        print(
            f"WARNING: embedding failed: {e}"
        )

        return None


# ============================================================
# SPEAKER PROFILES FROM CURRENT VIDEO
# ============================================================

def build_profiles(
    turns,
    waveform,
    sample_rate,
    inference
):

    profiles = {}

    for turn in turns:

        speaker = turn.get(
            "speaker"
        )

        if speaker is None:

            continue

        start = float(
            turn["start"]
        )

        end = float(
            turn["end"]
        )

        duration = end - start

        if duration < MIN_SAMPLE_DURATION:

            continue

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

        profiles.setdefault(
            speaker,
            []
        ).append({

            "start": start,

            "end": end,

            "duration": duration,

            "embedding": embedding
        })

    return profiles


# ============================================================
# CENTROIDS
# ============================================================

def calculate_centroids(
    profiles
):

    centroids = {}

    for speaker, samples in profiles.items():

        embeddings = [
            x["embedding"]
            for x in samples
        ]

        if not embeddings:

            continue

        matrix = torch.stack(
            embeddings
        )

        centroid = matrix.mean(
            dim=0
        )

        centroid = centroid / (
            torch.norm(centroid)
            +
            1e-8
        )

        centroids[
            speaker
        ] = centroid

    return centroids


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    a,
    b
):

    a = a.flatten().float()

    b = b.flatten().float()

    a = a / (
        torch.norm(a)
        +
        1e-8
    )

    b = b / (
        torch.norm(b)
        +
        1e-8
    )

    return float(
        torch.dot(a, b)
    )


# ============================================================
# CURRENT VIDEO SPEAKER COMPARISON
# ============================================================

def compare_speakers(
    centroids
):

    speakers = list(
        centroids.keys()
    )

    comparisons = []

    for i in range(
        len(speakers)
    ):

        for j in range(
            i + 1,
            len(speakers)
        ):

            a = speakers[i]

            b = speakers[j]

            similarity = cosine_similarity(
                centroids[a],
                centroids[b]
            )

            comparisons.append({

                "speaker_a": a,

                "speaker_b": b,

                "similarity": round(
                    similarity,
                    4
                ),

                "same_speaker": (
                    similarity
                    >=
                    SIMILARITY_THRESHOLD
                )
            })

    comparisons.sort(
        key=lambda x:
        x["similarity"],
        reverse=True
    )

    return comparisons


# ============================================================
# TENSOR SERIALIZATION
# ============================================================

def embedding_to_list(
    embedding
):

    return [
        float(x)
        for x in embedding
        .detach()
        .cpu()
        .flatten()
        .tolist()
    ]


def list_to_embedding(
    values
):

    tensor = torch.tensor(
        values,
        dtype=torch.float32
    )

    tensor = tensor / (
        torch.norm(tensor)
        +
        1e-8
    )

    return tensor


# ============================================================
# PERSISTENT PROFILE DIRECTORY
# ============================================================

def ensure_profile_directory():

    PROFILE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# FIND NEXT IDENTITY NUMBER
# ============================================================

def get_next_identity_number():

    ensure_profile_directory()

    highest = 0

    for path in PROFILE_DIR.glob(
        "IDENTITY_*.json"
    ):

        try:

            number = int(
                path.stem.split("_")[-1]
            )

            highest = max(
                highest,
                number
            )

        except Exception:

            continue

    return highest + 1


# ============================================================
# LOAD PERSISTENT PROFILES
# ============================================================

def load_persistent_profiles():

    ensure_profile_directory()

    profiles = {}

    for path in sorted(
        PROFILE_DIR.glob(
            "IDENTITY_*.json"
        )
    ):

        try:

            data = load_json(
                path
            )

            identity = data.get(
                "identity"
            )

            embeddings = data.get(
                "embeddings",
                []
            )

            if not identity:

                continue

            usable_embeddings = []

            for values in embeddings:

                if not values:

                    continue

                usable_embeddings.append(
                    list_to_embedding(
                        values
                    )
                )

            if not usable_embeddings:

                continue

            profiles[
                identity
            ] = {

                "identity": identity,

                "embedding_model":
                    data.get(
                        "embedding_model",
                        EMBEDDING_MODEL
                    ),

                "samples":
                    data.get(
                        "samples",
                        len(
                            usable_embeddings
                        )
                    ),

                "total_duration":
                    data.get(
                        "total_duration",
                        0.0
                    ),

                "embeddings":
                    usable_embeddings,

                "created":
                    data.get(
                        "created"
                    ),

                "updated":
                    data.get(
                        "updated"
                    )
            }

        except Exception as e:

            print(
                f"WARNING: failed to load "
                f"profile {path}: {e}"
            )

    return profiles


# ============================================================
# PERSISTENT PROFILE CENTROID
# ============================================================

def profile_centroid(
    profile
):

    embeddings = profile.get(
        "embeddings",
        []
    )

    if not embeddings:

        return None

    matrix = torch.stack(
        embeddings
    )

    centroid = matrix.mean(
        dim=0
    )

    centroid = centroid / (
        torch.norm(centroid)
        +
        1e-8
    )

    return centroid


# ============================================================
# MATCH CURRENT SPEAKER AGAINST DATABASE
# ============================================================

def match_existing_identity(
    embedding,
    persistent_profiles
):

    best_identity = None

    best_similarity = -1.0

    for identity, profile in (
        persistent_profiles.items()
    ):

        centroid = profile_centroid(
            profile
        )

        if centroid is None:

            continue

        similarity = cosine_similarity(
            embedding,
            centroid
        )

        if similarity > best_similarity:

            best_similarity = similarity

            best_identity = identity

    if (
        best_identity is not None
        and
        best_similarity
        >=
        SIMILARITY_THRESHOLD
    ):

        return (
            best_identity,
            best_similarity
        )

    return (
        None,
        best_similarity
    )


# ============================================================
# CREATE NEW PROFILE
# ============================================================

def create_new_profile(
    embedding,
    duration
):

    number = get_next_identity_number()

    identity = (
        f"IDENTITY_{number:03d}"
    )

    profile = {

        "identity": identity,

        "embedding_model":
            EMBEDDING_MODEL,

        "samples": 1,

        "total_duration":
            round(
                duration,
                3
            ),

        "embeddings": [
            embedding_to_list(
                embedding
            )
        ]
    }

    save_json(
        PROFILE_DIR /
        f"{identity}.json",
        profile
    )

    return (
        identity,
        profile
    )


# ============================================================
# UPDATE EXISTING PROFILE
# ============================================================

def update_profile(
    profile,
    embedding,
    duration
):

    embeddings = profile.setdefault(
        "embeddings",
        []
    )

    embeddings.append(
        embedding
    )

    # Keep the profile bounded.
    #
    # We keep the most recent samples.
    #
    if len(embeddings) > MAX_PROFILE_SAMPLES:

        embeddings = embeddings[
            -MAX_PROFILE_SAMPLES:
        ]

        profile[
            "embeddings"
        ] = embeddings

    profile[
        "samples"
    ] = len(
        profile["embeddings"]
    )

    profile[
        "total_duration"
    ] = round(
        float(
            profile.get(
                "total_duration",
                0.0
            )
        )
        +
        duration,
        3
    )

    profile[
        "embedding_model"
    ] = EMBEDDING_MODEL

    return profile


# ============================================================
# SAVE PROFILE
# ============================================================

def save_profile(
    identity,
    profile
):

    serializable = {

        "identity":
            identity,

        "embedding_model":
            profile.get(
                "embedding_model",
                EMBEDDING_MODEL
            ),

        "samples":
            profile.get(
                "samples",
                0
            ),

        "total_duration":
            profile.get(
                "total_duration",
                0.0
            ),

        "embeddings": [

            embedding_to_list(
                x
            )
            if isinstance(
                x,
                torch.Tensor
            )
            else x

            for x in profile.get(
                "embeddings",
                []
            )
        ]
    }

    save_json(
        PROFILE_DIR /
        f"{identity}.json",
        serializable
    )


# ============================================================
# ASSIGN PERSISTENT IDENTITIES
# ============================================================

def assign_persistent_identities(
    profiles,
    centroids
):

    persistent_profiles = (
        load_persistent_profiles()
    )

    identity_map = {}

    identity_matches = []

    # --------------------------------------------------------
    # Process strongest / longest speakers first.
    #
    # Longer samples normally provide better embeddings.
    # --------------------------------------------------------

    speaker_order = sorted(

        centroids.keys(),

        key=lambda speaker:
        sum(
            x["duration"]
            for x in profiles.get(
                speaker,
                []
            )
        ),

        reverse=True
    )

    # --------------------------------------------------------
    # Prevent two NEW speakers in the same video from being
    # assigned to the same persistent identity.
    # --------------------------------------------------------

    identities_used_this_video = set()

    for speaker in speaker_order:

        embedding = centroids[
            speaker
        ]

        duration = sum(
            x["duration"]
            for x in profiles.get(
                speaker,
                []
            )
        )

        best_identity = None

        best_similarity = -1.0

        # ----------------------------------------------------
        # Compare against existing persistent identities.
        # ----------------------------------------------------

        for identity, profile in (
            persistent_profiles.items()
        ):

            if identity in (
                identities_used_this_video
            ):

                continue

            centroid = profile_centroid(
                profile
            )

            if centroid is None:

                continue

            similarity = cosine_similarity(
                embedding,
                centroid
            )

            if similarity > best_similarity:

                best_similarity = similarity

                best_identity = identity

        # ----------------------------------------------------
        # Existing identity match
        # ----------------------------------------------------

        if (
            best_identity is not None
            and
            best_similarity
            >=
            SIMILARITY_THRESHOLD
        ):

            identity = best_identity

            profile = (
                persistent_profiles[
                    identity
                ]
            )

            profile = update_profile(
                profile,
                embedding,
                duration
            )

            persistent_profiles[
                identity
            ] = profile

            save_profile(
                identity,
                profile
            )

            identity_matches.append({

                "pyannote_speaker":
                    speaker,

                "identity":
                    identity,

                "similarity":
                    round(
                        best_similarity,
                        4
                    ),

                "match":
                    "EXISTING"
            })

        # ----------------------------------------------------
        # No match -> create new identity
        # ----------------------------------------------------

        else:

            identity, profile = (
                create_new_profile(
                    embedding,
                    duration
                )
            )

            persistent_profiles[
                identity
            ] = profile

            identity_matches.append({

                "pyannote_speaker":
                    speaker,

                "identity":
                    identity,

                "similarity":
                    (
                        round(
                            best_similarity,
                            4
                        )
                        if best_similarity >= 0
                        else None
                    ),

                "match":
                    "NEW"
            })

        identity_map[
            speaker
        ] = identity

        identities_used_this_video.add(
            identity
        )

    return (
        identity_map,
        identity_matches,
        persistent_profiles
    )


# ============================================================
# UNKNOWN TURN ASSIGNMENT
# ============================================================

def assign_unknown_turns(
    turns,
    identity_map
):

    output = []

    for index, turn in enumerate(
        turns
    ):

        speaker = turn.get(
            "speaker"
        )

        new_turn = dict(
            turn
        )

        # ----------------------------------------------------
        # Known speaker
        # ----------------------------------------------------

        if speaker is not None:

            new_turn[
                "identity"
            ] = identity_map.get(
                speaker
            )

            new_turn[
                "identity_inferred"
            ] = False

            output.append(
                new_turn
            )

            continue

        # ----------------------------------------------------
        # Unknown speaker
        # ----------------------------------------------------

        previous_identity = None

        next_identity = None

        # Previous turn
        if index > 0:

            previous = turns[
                index - 1
            ]

            previous_speaker = (
                previous.get(
                    "speaker"
                )
            )

            if previous_speaker is not None:

                gap = (
                    turn["start"]
                    -
                    previous["end"]
                )

                if (
                    gap >= 0
                    and
                    gap <= MAX_UNKNOWN_GAP
                ):

                    previous_identity = (
                        identity_map.get(
                            previous_speaker
                        )
                    )

        # Next turn
        if index + 1 < len(turns):

            following = turns[
                index + 1
            ]

            following_speaker = (
                following.get(
                    "speaker"
                )
            )

            if following_speaker is not None:

                gap = (
                    following["start"]
                    -
                    turn["end"]
                )

                if (
                    gap >= 0
                    and
                    gap <= MAX_UNKNOWN_GAP
                ):

                    next_identity = (
                        identity_map.get(
                            following_speaker
                        )
                    )

        # ----------------------------------------------------
        # Infer only from nearby known speaker.
        # If both sides exist they must agree.
        # ----------------------------------------------------

        identity = None

        inferred = False

        if (
            previous_identity is not None
            and
            next_identity is not None
            and
            previous_identity
            ==
            next_identity
        ):

            identity = previous_identity

            inferred = True

        elif (
            previous_identity is not None
            and
            next_identity is None
        ):

            identity = previous_identity

            inferred = True

        elif (
            next_identity is not None
            and
            previous_identity is None
        ):

            identity = next_identity

            inferred = True

        new_turn[
            "identity"
        ] = identity

        new_turn[
            "identity_inferred"
        ] = inferred

        output.append(
            new_turn
        )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "SPEAKER IDENTITY RECONSTRUCTION"
    )
    print("=" * 70)

    print(
        f"Audio:       {AUDIO_FILE}"
    )

    print(
        f"Input:       {INPUT_FILE}"
    )

    print(
        f"Output:      {OUTPUT_FILE}"
    )

    print(
        f"Profiles:    {PROFILE_DIR}"
    )

    print(
        f"Embedding:   {EMBEDDING_MODEL}"
    )

    print(
        f"Threshold:   {SIMILARITY_THRESHOLD}"
    )

    print()

    ensure_profile_directory()

    data = load_json(
        INPUT_FILE
    )

    turns = data.get(
        "turns",
        []
    )

    print(
        f"Reconstructed turns: "
        f"{len(turns)}"
    )

    print()
    print(
        "Loading audio..."
    )

    waveform, sample_rate = (
        load_audio(
            AUDIO_FILE
        )
    )

    duration = (
        waveform.shape[1]
        /
        sample_rate
    )

    print(
        f"Audio duration: "
        f"{duration:.3f}s"
    )

    print(
        f"Sample rate:    "
        f"{sample_rate} Hz"
    )

    print()
    print(
        "Loading speaker embedding model..."
    )

    model = Model.from_pretrained(
        EMBEDDING_MODEL
    )

    inference = Inference(
        model,
        window="whole"
    )

    print(
        "Speaker embedding model loaded."
    )

    # --------------------------------------------------------
    # Current video profiles
    # --------------------------------------------------------

    print()
    print(
        "Building speaker profiles..."
    )

    profiles = build_profiles(
        turns,
        waveform,
        sample_rate,
        inference
    )

    print()

    for speaker, samples in sorted(
        profiles.items()
    ):

        total_duration = sum(
            x["duration"]
            for x in samples
        )

        print(
            f"  {speaker}: "
            f"{len(samples)} samples, "
            f"{total_duration:.2f}s"
        )

    print()
    print(
        "Calculating speaker centroids..."
    )

    centroids = calculate_centroids(
        profiles
    )

    print(
        f"Usable speaker profiles: "
        f"{len(centroids)}"
    )

    # --------------------------------------------------------
    # Current-video similarity
    # --------------------------------------------------------

    comparisons = compare_speakers(
        centroids
    )

    print()
    print("=" * 80)
    print(
        "SPEAKER SIMILARITY"
    )
    print("=" * 80)

    if not comparisons:

        print(
            "No speaker pairs to compare."
        )

    for comparison in comparisons:

        status = (
            "SAME"
            if comparison[
                "same_speaker"
            ]
            else "DIFFERENT"
        )

        print(
            f"{comparison['speaker_a']:<15}"
            f"<-> "
            f"{comparison['speaker_b']:<15}"
            f"{comparison['similarity']:>7.4f}"
            f"  {status}"
        )

    # --------------------------------------------------------
    # Persistent identities
    # --------------------------------------------------------

    print()
    print(
        "Loading persistent speaker profiles..."
    )

    existing_profiles = (
        load_persistent_profiles()
    )

    print(
        f"Existing persistent identities: "
        f"{len(existing_profiles)}"
    )

    print()
    print(
        "Matching speakers against persistent identities..."
    )

    (
        identity_map,
        identity_matches,
        persistent_profiles
    ) = assign_persistent_identities(
        profiles,
        centroids
    )

    print()
    print("=" * 80)
    print(
        "PERSISTENT SPEAKER IDENTITIES"
    )
    print("=" * 80)

    for match in identity_matches:

        similarity = match[
            "similarity"
        ]

        similarity_text = (
            f"{similarity:.4f}"
            if similarity is not None
            else "-"
        )

        print(
            f"{match['pyannote_speaker']:<15}"
            f"-> "
            f"{match['identity']:<15}"
            f"{match['match']:<10}"
            f"similarity={similarity_text}"
        )

    # --------------------------------------------------------
    # Assign identities to turns
    # --------------------------------------------------------

    print()
    print(
        "Assigning identities to dialogue turns..."
    )

    identified_turns = (
        assign_unknown_turns(
            turns,
            identity_map
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = dict(
        data
    )

    output[
        "identity_model"
    ] = EMBEDDING_MODEL

    output[
        "identity_similarity_threshold"
    ] = SIMILARITY_THRESHOLD

    output[
        "persistent_profile_directory"
    ] = str(
        PROFILE_DIR
    )

    output[
        "speaker_identity_map"
    ] = identity_map

    output[
        "speaker_identity_matches"
    ] = identity_matches

    output[
        "speaker_similarity"
    ] = comparisons

    output[
        "turns"
    ] = identified_turns

    print()
    print("=" * 100)
    print(
        "IDENTIFIED SPEAKER TURNS"
    )
    print("=" * 100)

    print(
        f"{'TIME':<17}"
        f"{'PYANNOTE':<15}"
        f"{'IDENTITY':<15}"
        f"TEXT"
    )

    print(
        "-" * 100
    )

    for turn in identified_turns:

        print(
            f"{turn['start']:6.2f} - "
            f"{turn['end']:6.2f}    "
            f"{str(turn.get('speaker')):<15}"
            f"{str(turn.get('identity')):<15}"
            f"{turn.get('text', '')}"
        )

    save_json(
        OUTPUT_FILE,
        output
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print()
    print(
        f"Persistent profiles: "
        f"{PROFILE_DIR}"
    )

    print()
    print("=" * 70)
    print(
        "SPEAKER IDENTITY RECONSTRUCTION: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
