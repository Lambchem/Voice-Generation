# Radio TTS Quick Start

This workspace fine-tunes a text-to-speech model on the dataset in `data/` and writes all outputs to `output/`.

## What you get

- `train.py` for training and automatic checkpoint saving
- `eval.py` for synthesizing one WAV file from text
- `output/model/` for saved model checkpoints
- `output/rollout/` for per-epoch audio samples
- `loss.csv` and `loss.png` in the project root for loss tracking

## Requirements

You need Python with these packages available:

- `torch`
- `torchaudio`
- `transformers`
- `datasets`
- `pandas`
- `soundfile`
- `matplotlib`

The first run will also download the pretrained SpeechT5 model and vocoder from Hugging Face.

## Data layout

The dataset should stay in the provided structure:

```text
data/
  train-00000-of-00010.parquet
  train-00001-of-00010.parquet
  ...
```

The dataset metadata is described in `dataset_README.md`.

## Train

Start training with the default settings:

```bash
python train.py
```

Recommended first run on Windows:

```bash
python train.py --epochs 1 --batch-size 2
```

Useful options:

- `--max-examples` limits the dataset for a quick smoke test
- `--sample-text` adds custom rollout prompts and may be passed multiple times
- `--output-dir` changes where checkpoints and samples are written

## Resume training

Training resumes automatically from the latest checkpoint in `output/model/` when one exists.

To force a fresh run, use:

```bash
python train.py --no-resume
```

## Outputs after each epoch

Each epoch writes:

- `output/model/epoch_XXXX/` for the checkpoint
- `output/rollout/epoch_XXXX/` for generated audio samples
- `loss.csv` and `loss.png` updated in the project root
- `output/model/last_checkpoint.txt` pointing to the most recent checkpoint

## Evaluate

Generate a WAV file from text:

```bash
python eval.py --text "Hello, this is a test."
```

The output is written to `output/eval/`.

## Long text

`eval.py` supports long text by splitting the input into chunks, re-encoding each chunk through the tokenizer, generating audio chunk by chunk, and stitching the result into one WAV file.

If you want to tune long-form generation, try:

```bash
python eval.py --text "your long text" --chunk-token-limit 480 --chunk-gap-seconds 0.05 --maxlenratio 60
```

## Notes

- The scripts use the pretrained `microsoft/speecht5_tts` model and `microsoft/speecht5_hifigan` vocoder by default.
- The dataset README says attribution is required when using the generated voice in interfaces that generate audio in response to user action. Refer to the voice as Jenny, and where practical, Jenny (Dioco).
