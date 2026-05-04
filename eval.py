from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
from transformers import SpeechT5ForTextToSpeech

from tts_common import (
    DEFAULT_BASE_MODEL,
    TARGET_SAMPLE_RATE,
    ensure_directory,
    find_latest_checkpoint,
    load_processor,
    load_vocoder,
    normalize_text,
    sanitize_filename,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate speech from text using the latest trained checkpoint.")
    parser.add_argument("--text", type=str, default="While this kind of generalization has always been thought of as a key strength of robotic foundation models, actual models demonstrated to date have not shown the kind of broad compositional generalization that we’ve seen, for example, from LLMs.", help="Text to synthesize. If omitted, you will be prompted.")
    parser.add_argument("--checkpoint", type=Path, default=Path("output/model"), help="Checkpoint directory or model root.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/eval"), help="Directory for synthesized audio.")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL, help="Fallback base checkpoint if no trained checkpoint is found.")
    parser.add_argument("--maxlenratio", type=float, default=60.0, help="Maximum generation length ratio. Larger values allow longer output.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Stop threshold used by SpeechT5 generation.")
    parser.add_argument("--chunk-token-limit", type=int, default=480, help="Maximum tokens per generated chunk. Keep below the model limit for long text.")
    parser.add_argument("--chunk-gap-seconds", type=float, default=0.05, help="Silence inserted between generated chunks when stitching long text.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override, for example cpu or cuda.")
    return parser.parse_args()


def resolve_checkpoint(path: Path, base_model: str) -> Path | str:
    if path.is_dir():
        try:
            return find_latest_checkpoint(path)
        except FileNotFoundError:
            return path if any(path.iterdir()) else base_model
    return path


def compute_safe_chunk_token_limit(
    model,
    tokenizer,
    requested_limit: int,
    maxlenratio: float,
) -> int:
    max_text_positions = int(getattr(model.config, "max_text_positions", getattr(tokenizer, "model_max_length", requested_limit)))
    max_speech_positions = int(getattr(model.config, "max_speech_positions", 0))
    reduction_factor = int(getattr(model.config, "reduction_factor", 1))

    safe_limit = requested_limit
    if max_speech_positions > 0 and maxlenratio > 0:
        safe_limit = min(safe_limit, max(1, int(((max_speech_positions - 1) * reduction_factor) / maxlenratio)))

    safe_limit = min(safe_limit, max_text_positions)
    tokenizer_limit = int(getattr(tokenizer, "model_max_length", safe_limit))
    if tokenizer_limit > 0:
        safe_limit = min(safe_limit, tokenizer_limit)

    return max(1, safe_limit)


def split_text_into_token_chunks(text: str, tokenizer, max_tokens: int) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    token_ids = tokenizer(normalized, add_special_tokens=False).input_ids
    if not token_ids:
        return []

    chunks: list[str] = []
    for start in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[start : start + max_tokens]
        if chunk_ids:
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            chunk_text = normalize_text(chunk_text)
            if chunk_text:
                chunks.append(chunk_text)
    return chunks


def generate_waveform_chunks(
    model,
    tokenizer,
    vocoder,
    text: str,
    device: torch.device,
    threshold: float,
    maxlenratio: float,
    chunk_token_limit: int,
) -> list[np.ndarray]:
    chunk_texts = split_text_into_token_chunks(text, tokenizer, chunk_token_limit)
    if not chunk_texts:
        return []

    speaker_embeddings = torch.zeros((1, model.config.speaker_embedding_dim), device=device)
    waveforms: list[np.ndarray] = []

    with torch.inference_mode():
        for chunk_text in chunk_texts:
            chunk_tokens = tokenizer(chunk_text, return_tensors="pt")
            chunk_tokens = {name: tensor.to(device) for name, tensor in chunk_tokens.items()}
            waveform = model.generate_speech(
                input_ids=chunk_tokens["input_ids"],
                attention_mask=chunk_tokens.get("attention_mask"),
                speaker_embeddings=speaker_embeddings,
                threshold=threshold,
                maxlenratio=maxlenratio,
                vocoder=vocoder,
            )

            if isinstance(waveform, tuple):
                waveform = waveform[0]
            if waveform.ndim > 1:
                waveform = waveform.squeeze(0)

            waveforms.append(waveform.detach().cpu().numpy().astype(np.float32))

    return waveforms


def stitch_waveforms(waveforms: Iterable[np.ndarray], gap_seconds: float) -> np.ndarray:
    chunks = [np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in waveforms if chunk is not None and len(chunk) > 0]
    if not chunks:
        return np.zeros(0, dtype=np.float32)

    gap = np.zeros(int(TARGET_SAMPLE_RATE * max(gap_seconds, 0.0)), dtype=np.float32)
    stitched: list[np.ndarray] = []
    for index, chunk in enumerate(chunks):
        if index > 0 and gap.size > 0:
            stitched.append(gap)
        stitched.append(chunk)
    return np.concatenate(stitched) if stitched else np.zeros(0, dtype=np.float32)


def main() -> None:
    args = parse_args()

    if args.text is None:
        text = input("Enter text to synthesize: ").strip()
    else:
        text = args.text.strip()

    if not text:
        raise ValueError("Text input cannot be empty.")

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    checkpoint = resolve_checkpoint(args.checkpoint, args.base_model)
    processor = load_processor(str(checkpoint))
    model = SpeechT5ForTextToSpeech.from_pretrained(str(checkpoint))
    vocoder = load_vocoder()

    model.to(device)
    vocoder.to(device)
    model.eval()
    vocoder.eval()

    waveform_chunks = generate_waveform_chunks(
        model=model,
        tokenizer=processor.tokenizer,
        vocoder=vocoder,
        text=text,
        device=device,
        threshold=args.threshold,
        maxlenratio=args.maxlenratio,
        chunk_token_limit=compute_safe_chunk_token_limit(
            model=model,
            tokenizer=processor.tokenizer,
            requested_limit=args.chunk_token_limit,
            maxlenratio=args.maxlenratio,
        ),
    )
    waveform = stitch_waveforms(waveform_chunks, args.chunk_gap_seconds)

    output_dir = ensure_directory(args.output_dir)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    stem = f"{stamp}_{sanitize_filename(text)}"
    wav_path = output_dir / f"{stem}.wav"
    txt_path = output_dir / f"{stem}.txt"
    meta_path = output_dir / f"{stem}.json"

    sf.write(wav_path, waveform.astype(np.float32), TARGET_SAMPLE_RATE)
    txt_path.write_text(text, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "chunk_count": len(waveform_chunks),
                "chunk_token_limit": compute_safe_chunk_token_limit(
                    model=model,
                    tokenizer=processor.tokenizer,
                    requested_limit=args.chunk_token_limit,
                    maxlenratio=args.maxlenratio,
                ),
                "chunk_gap_seconds": args.chunk_gap_seconds,
                "checkpoint": str(checkpoint),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Saved audio to {wav_path}")
    print(f"Text: {text}")
    print(f"Checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
