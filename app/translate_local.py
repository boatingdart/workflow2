import json
import os
import time
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


INPUT_PATH = Path("/work/clean_dialogue.json")
OUTPUT_PATH = Path("/work/translated_dialogue.json")

MODEL_NAME = os.getenv(
    "TRANSLATION_MODEL",
    "Helsinki-NLP/opus-mt-ja-en",
)

MODEL_DIR = Path(
    os.getenv(
        "TRANSLATION_MODEL_DIR",
        "/models/translation",
    )
)

DEVICE = os.getenv(
    "TRANSLATION_DEVICE",
    "auto",
)

MAX_LENGTH = int(
    os.getenv(
        "TRANSLATION_MAX_LENGTH",
        "256",
    )
)


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


def get_device():
    if DEVICE != "auto":
        return DEVICE

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"

    except Exception:
        pass

    return "cpu"


def load_model():
    print()
    print("=" * 60)
    print("LOCAL TRANSLATION INITIALIZATION")
    print("=" * 60)

    print(f"Model:       {MODEL_NAME}")
    print(f"Model cache: {MODEL_DIR}")

    device = get_device()

    print(f"Device:      {device}")
    print()

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=str(MODEL_DIR),
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        cache_dir=str(MODEL_DIR),
    )

    model.to(device)
    model.eval()

    load_time = (
        time.perf_counter()
        - model_start
    )

    print(
        f"Model loaded in {load_time:.2f} seconds"
    )

    return tokenizer, model, device


def translate_text(
    text,
    tokenizer,
    model,
    device,
):
    if not text.strip():
        return ""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    import torch

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            num_beams=4,
        )

    translation = tokenizer.decode(
        generated[0],
        skip_special_tokens=True,
    )

    return translation.strip()


def main():
    print()
    print("=" * 60)
    print("LOCAL JAPANESE → ENGLISH TRANSLATION")
    print("=" * 60)

    print(f"Input:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    data = load_json(INPUT_PATH)

    segments = data.get(
        "segments",
        [],
    )

    print(
        f"Dialogue lines: {len(segments)}"
    )

    tokenizer, model, device = load_model()

    print()
    print("=" * 60)
    print("TRANSLATING DIALOGUE")
    print("=" * 60)

    translation_start = time.perf_counter()

    translated_segments = []

    for index, segment in enumerate(segments):
        text_ja = segment.get(
            "text",
            "",
        ).strip()

        print(
            f"[{index + 1}/{len(segments)}] "
            f"{segment.get('speaker')}: "
            f"{text_ja}"
        )

        text_en = translate_text(
            text_ja,
            tokenizer,
            model,
            device,
        )

        print(
            f"    → {text_en}"
        )

        translated_segments.append(
            {
                "id": segment.get(
                    "id",
                    index,
                ),
                "speaker": segment.get(
                    "speaker"
                ),
                "start": segment.get(
                    "start"
                ),
                "end": segment.get(
                    "end"
                ),
                "text_ja": text_ja,
                "text_en": text_en,
            }
        )

    translation_time = (
        time.perf_counter()
        - translation_start
    )

    result = {
        "audio": data.get("audio"),
        "transcription_model": data.get(
            "transcription_model"
        ),
        "diarization_model": data.get(
            "diarization_model"
        ),
        "translation_model": MODEL_NAME,
        "language": data.get(
            "language",
            "ja",
        ),
        "target_language": "en",
        "translation_time": translation_time,
        "segments": translated_segments,
    }

    save_json(
        OUTPUT_PATH,
        result,
    )

    print()
    print("=" * 60)
    print("LOCAL TRANSLATION: PASS")
    print("=" * 60)

    print(
        f"Translated:        {len(translated_segments)}"
    )
    print(
        f"Translation time:  "
        f"{translation_time:.2f} seconds"
    )
    print(
        f"Saved:             {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
