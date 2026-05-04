from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from transformers import SpeechT5FeatureExtractor, SpeechT5HifiGan, SpeechT5Processor, SpeechT5Tokenizer


TARGET_SAMPLE_RATE = 16000
DEFAULT_BASE_MODEL = "microsoft/speecht5_tts"
DEFAULT_VOCODER = "microsoft/speecht5_hifigan"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def sanitize_filename(text: str, max_length: int = 72) -> str:
    cleaned = normalize_text(text)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    if not cleaned:
        cleaned = "sample"
    return cleaned[:max_length]


def load_speech_dataset(data_dir: Path, seed: int = 42, max_examples: int | None = None):
    parquet_files = sorted(str(path) for path in data_dir.glob("train-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet shards found in {data_dir}")

    frames = [pd.read_parquet(path, columns=["file_name", "transcription", "transcription_normalised", "audio"]) for path in parquet_files]
    frame = pd.concat(frames, ignore_index=True)

    if max_examples is not None:
        max_examples = min(max_examples, len(frame))
        frame = frame.iloc[:max_examples].copy()

    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return frame


def decode_audio_bytes(audio_dict: dict, target_sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    raw_bytes = audio_dict["bytes"]
    waveform, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    waveform = np.asarray(waveform, dtype=np.float32)
    if sample_rate != target_sample_rate:
        tensor_waveform = torch.from_numpy(waveform)
        waveform = torchaudio.functional.resample(tensor_waveform, sample_rate, target_sample_rate).numpy()

    return trim_silence(waveform)


def trim_silence(waveform: np.ndarray, threshold: float = 0.01, pad_seconds: float = 0.05) -> np.ndarray:
    if waveform.size == 0:
        return waveform.astype(np.float32, copy=False)

    peak = float(np.max(np.abs(waveform)))
    if peak <= 0.0:
        return waveform.astype(np.float32, copy=False)

    active = np.flatnonzero(np.abs(waveform) >= peak * threshold)
    if active.size == 0:
        return waveform.astype(np.float32, copy=False)

    pad = int(TARGET_SAMPLE_RATE * pad_seconds)
    start = max(int(active[0]) - pad, 0)
    end = min(int(active[-1]) + pad + 1, waveform.shape[0])
    return np.asarray(waveform[start:end], dtype=np.float32)


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Path, seed: int = 42, max_examples: int | None = None):
        self.dataset = load_speech_dataset(data_dir=data_dir, seed=seed, max_examples=max_examples)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        row = self.dataset.iloc[int(index)]
        text = row.get("transcription_normalised") or row.get("transcription") or ""
        return {
            "text": normalize_text(text),
            "audio": row["audio"],
            "file_name": row.get("file_name", ""),
        }


class TTSCollator:
    def __init__(
        self,
        tokenizer: SpeechT5Tokenizer,
        feature_extractor: SpeechT5FeatureExtractor,
        speaker_embedding_dim: int,
        reduction_factor: int = 1,
        max_text_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.speaker_embedding_dim = speaker_embedding_dim
        self.reduction_factor = max(1, int(reduction_factor))
        self.max_text_length = max_text_length

    def _pad_spectrograms_for_reduction_factor(
        self,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_length = labels.shape[1]
        pad_frames = (-current_length) % self.reduction_factor
        if pad_frames == 0:
            return labels, attention_mask

        pad_labels = labels.new_full((labels.shape[0], pad_frames, labels.shape[2]), -100.0)
        pad_mask = attention_mask.new_zeros((attention_mask.shape[0], pad_frames))
        labels = torch.cat([labels, pad_labels], dim=1)
        attention_mask = torch.cat([attention_mask, pad_mask], dim=1)
        return labels, attention_mask

    def __call__(self, batch: Sequence[dict]) -> dict[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        waveforms = [decode_audio_bytes(item["audio"]) for item in batch]

        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        spectrograms = self.feature_extractor(
            audio_target=waveforms,
            sampling_rate=TARGET_SAMPLE_RATE,
            padding=True,
            return_tensors="pt",
        )

        labels = spectrograms["input_values"].to(torch.float32)
        attention_mask = spectrograms["attention_mask"].to(torch.long)
        labels, attention_mask = self._pad_spectrograms_for_reduction_factor(labels, attention_mask)
        labels = labels.masked_fill(attention_mask.unsqueeze(-1) == 0, -100.0)

        speaker_embeddings = torch.zeros((len(batch), self.speaker_embedding_dim), dtype=torch.float32)

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
            "speaker_embeddings": speaker_embeddings,
        }


def load_processor(model_name: str = DEFAULT_BASE_MODEL) -> SpeechT5Processor:
    return SpeechT5Processor.from_pretrained(model_name)


def load_tokenizer(model_name: str = DEFAULT_BASE_MODEL) -> SpeechT5Tokenizer:
    return SpeechT5Tokenizer.from_pretrained(model_name)


def load_vocoder(model_name: str = DEFAULT_VOCODER) -> SpeechT5HifiGan:
    return SpeechT5HifiGan.from_pretrained(model_name)


def write_loss_artifacts(records: list[dict], csv_path: Path, plot_path: Path) -> None:
    ensure_directory(csv_path.parent)
    ensure_directory(plot_path.parent)

    frame = pd.DataFrame(records)
    frame.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(frame["epoch"], frame["loss"], linewidth=2.0)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def find_latest_checkpoint(model_root: Path) -> Path:
    latest_marker = model_root / "last_checkpoint.txt"
    if latest_marker.exists():
        stored = Path(latest_marker.read_text(encoding="utf-8").strip())
        if stored.exists():
            return stored

    candidates = []
    for path in model_root.glob("epoch_*"):
        if path.is_dir():
            match = re.search(r"epoch_(\d+)", path.name)
            if match:
                candidates.append((int(match.group(1)), path))

    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {model_root}")

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def save_checkpoint_marker(model_root: Path, checkpoint_path: Path) -> None:
    ensure_directory(model_root)
    marker = model_root / "last_checkpoint.txt"
    marker.write_text(str(checkpoint_path.resolve()), encoding="utf-8")


def save_rollout_samples(
    model,
    processor: SpeechT5Processor,
    vocoder,
    sample_texts: Sequence[str],
    output_dir: Path,
    device: torch.device,
    maxlenratio: float,
    threshold: float,
) -> None:
    ensure_directory(output_dir)
    model.eval()
    vocoder.eval()

    metadata = []
    with torch.inference_mode():
        for index, text in enumerate(sample_texts, start=1):
            prompt = normalize_text(text)
            inputs = processor.tokenizer(prompt, return_tensors="pt")
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            speaker_embeddings = torch.zeros((1, model.config.speaker_embedding_dim), device=device)

            waveform = model.generate_speech(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                speaker_embeddings=speaker_embeddings,
                threshold=threshold,
                maxlenratio=maxlenratio,
                vocoder=vocoder,
            )

            if isinstance(waveform, tuple):
                waveform = waveform[0]
            if waveform.ndim > 1:
                waveform = waveform.squeeze(0)

            sample_name = f"sample_{index:02d}.wav"
            sample_path = output_dir / sample_name
            sf.write(sample_path, waveform.detach().cpu().numpy().astype(np.float32), TARGET_SAMPLE_RATE)
            (output_dir / f"sample_{index:02d}.txt").write_text(prompt, encoding="utf-8")
            metadata.append({"file": sample_name, "text": prompt})

    (output_dir / "samples.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def build_default_sample_texts() -> list[str]:
    return [
        "Hello, this is an epoch sample.",
        "The model should become clearer and more stable as training continues.",
        "This longer sentence checks whether the synthesizer can keep going without cutting off too early.",
    ]
