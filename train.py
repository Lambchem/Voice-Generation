from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import SpeechT5ForTextToSpeech, get_linear_schedule_with_warmup

from tts_common import (
    DEFAULT_BASE_MODEL,
    ensure_directory,
    build_default_sample_texts,
    find_latest_checkpoint,
    load_processor,
    load_vocoder,
    save_checkpoint_marker,
    save_rollout_samples,
    SpeechDataset,
    TTSCollator,
    write_loss_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a text-to-speech model on the Radio dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing train-*.parquet shards.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory for checkpoints and rollouts.")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL, help="Base SpeechT5 checkpoint to fine-tune.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-6, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max-text-length", type=int, default=512, help="Maximum number of text tokens per sample.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional cap on the number of dataset examples.")
    parser.add_argument("--seed", type=int, default=random.randint(0, 2**32 - 1), help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. Keep 0 for maximum compatibility.")
    parser.add_argument("--rollout-maxlenratio", type=float, default=20.0, help="Maximum output length ratio used for rollout samples.")
    parser.add_argument("--rollout-threshold", type=float, default=0.5, help="Stop threshold used for rollout samples.")
    parser.add_argument("--sample-text", action="append", default=None, help="Optional rollout text prompt. May be passed multiple times.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start a fresh run instead of resuming from the latest checkpoint.")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def load_loss_history(loss_csv_path: Path) -> list[dict]:
    if not loss_csv_path.exists():
        return []

    frame = pd.read_csv(loss_csv_path)
    if frame.empty:
        return []

    records: list[dict] = []
    for row in frame.itertuples(index=False):
        records.append({"epoch": int(row.epoch), "loss": float(row.loss)})
    return records


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    root_dir = args.output_dir
    model_dir = ensure_directory(root_dir / "model")
    rollout_root = ensure_directory(root_dir / "rollout")
    ensure_directory(root_dir)

    loss_csv_path = root_dir / "loss.csv"
    loss_plot_path = root_dir / "loss.png"
    records: list[dict] = []

    resume_checkpoint = None
    resume_state = None
    start_epoch = 1

    if args.resume:
        try:
            resume_checkpoint = find_latest_checkpoint(model_dir)
            state_path = resume_checkpoint / "training_state.pt"
            if state_path.exists():
                resume_state = torch.load(state_path, map_location="cpu")
                start_epoch = int(resume_state.get("epoch", 0)) + 1
        except FileNotFoundError:
            resume_checkpoint = None
            resume_state = None

        if resume_state is not None:
            records = [record for record in load_loss_history(loss_csv_path) if int(record["epoch"]) < start_epoch]

    if args.resume and resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        processor = load_processor(str(resume_checkpoint))
        model = SpeechT5ForTextToSpeech.from_pretrained(str(resume_checkpoint))
    else:
        processor = load_processor(args.base_model)
        model = SpeechT5ForTextToSpeech.from_pretrained(args.base_model)

    vocoder = load_vocoder()

    model.to(device)
    vocoder.to(device)

    dataset = SpeechDataset(args.data_dir, seed=args.seed, max_examples=args.max_examples)
    collator = TTSCollator(
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        speaker_embedding_dim=model.config.speaker_embedding_dim,
        reduction_factor=model.config.reduction_factor,
        max_text_length=args.max_text_length,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )

    planned_update_steps = max(1, math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.epochs)
    total_update_steps = planned_update_steps

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    warmup_steps = max(1, int(total_update_steps * 0.05))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    sample_texts = args.sample_text if args.sample_text else build_default_sample_texts()
    started_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    end_epoch = start_epoch + args.epochs - 1
    print(f"Training on {len(dataset)} examples for epochs {start_epoch}..{end_epoch}.")
    print(f"Device: {device}")
    print(f"Output root: {root_dir.resolve()}")
    print(f"Started: {started_at}")

    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            batch_count += 1
            batch = {name: tensor.to(device) for name, tensor in batch.items()}

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("SpeechT5 did not return a loss. Check the batch preparation.")
                scaled_loss = loss / args.gradient_accumulation_steps

            scaler.scale(scaled_loss).backward()
            running_loss += float(loss.detach().cpu())

            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            if step % 25 == 0 or step == len(train_loader):
                print(f"Epoch {epoch}/{args.epochs} Step {step}/{len(train_loader)} Loss {running_loss / batch_count:.4f}")

        avg_loss = running_loss / max(1, batch_count)
        records.append({"epoch": epoch, "loss": avg_loss})

        checkpoint_dir = ensure_directory(model_dir / f"epoch_{epoch:04d}")
        model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        torch.save(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "total_update_steps": total_update_steps,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "base_model": args.base_model,
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            },
            checkpoint_dir / "training_state.pt",
        )
        save_checkpoint_marker(model_dir, checkpoint_dir)

        rollout_dir = ensure_directory(rollout_root / f"epoch_{epoch:04d}")
        save_rollout_samples(
            model=model,
            processor=processor,
            vocoder=vocoder,
            sample_texts=sample_texts,
            output_dir=rollout_dir,
            device=device,
            maxlenratio=args.rollout_maxlenratio,
            threshold=args.rollout_threshold,
        )

        write_loss_artifacts(records, loss_csv_path, loss_plot_path)

        print(f"Epoch {epoch} complete. Avg loss: {avg_loss:.4f}")
        print(f"Saved checkpoint: {checkpoint_dir}")
        print(f"Saved rollout: {rollout_dir}")

    metadata = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "resume": args.resume,
        "resumed_from": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_text_length": args.max_text_length,
        "max_examples": args.max_examples,
        "seed": args.seed,
        "started_at": started_at,
    }
    (root_dir / "training_run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
