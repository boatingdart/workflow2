import json
import os
import time
from pathlib import Path

from faster_whisper import WhisperModel


def transcribe_audio(audio_path, output_path):
    """
    Transcribe speech audio with Faster-Whisper.

    Produces both segment-level and word-level timestamps.
    """

    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    model_name = os.getenv(
        "WHISPER_MODEL",
        "large-v3",
    )

    compute_type = os.getenv(
        "WHISPER_COMPUTE_TYPE",
        "int8_float16",
    )

    device = os.getenv(
        "WHISPER_DEVICE",
        "cuda",
    )

    beam_size = int(
        os.getenv(
            "WHISPER_BEAM_SIZE",
            "5",
        )
    )

    print()
    print("=" * 60)
    print("WHISPER INITIALIZATION")
    print("=" * 60)

    print(f"Model:         {model_name}")
    print(f"Device:        {device}")
    print(f"Compute type:  {compute_type}")
    print(f"Beam size:     {beam_size}")

    model_start = time.perf_counter()

    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root="/models/whisper",
    )

    model_time = time.perf_counter() - model_start

    print(
        f"Model loaded in "
        f"{model_time:.2f} seconds"
    )

    print()
    print("=" * 60)
    print("WHISPER TRANSCRIPTION")
    print("=" * 60)

    print(f"Audio: {audio_path}")

    transcription_start = time.perf_counter()
    
    segments, info = model.transcribe(
        str(audio_path),
        language="ja",
        beam_size=beam_size,
        best_of=5,
        temperature=0.0,
        vad_filter=True,
        word_timestamps=True,
    )

    segments = list(segments)

    transcription_time = (
        time.perf_counter()
        - transcription_start
    )

    result_segments = []

    for segment in segments:
        words = []

        if segment.words:
            for word in segment.words:
                words.append(
                    {
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": word.probability,
                    }
                )

        result_segments.append(
            {
                "id": len(result_segments),
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": words,
            }
        )

    result = {
        "audio": str(audio_path),
        "model": model_name,
        "compute_type": compute_type,
        "device": device,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "transcription_time": transcription_time,
        "segments": result_segments,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Language: {info.language}")
    print(
        "Language probability: "
        f"{info.language_probability:.3f}"
    )
    print(f"Segments: {len(result_segments)}")
    print(
        f"Transcription time: "
        f"{transcription_time:.2f} seconds"
    )
    print(f"Output: {output_path}")

    print()
    print("TRANSCRIPTION: PASS")

    return result

if __name__ == "__main__":
    audio_path = Path("/work/speech_audio.wav")
    output_path = Path("/work/transcript.json")

    transcribe_audio(
        audio_path,
        output_path,
    )
