"""Kaggle runner for fresh sequential V3 chat training.

The input dataset is flat. Files are copied into stage directories, then the
existing train.py CLI runs DailyDialog from random weights and resumes through
Persona-Chat and filtered OpenAssistant.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working/v3")
CHECKPOINTS = WORK / "checkpoints"

STAGES = (
    ("daily", "daily", 5, None),
    ("persona", "persona", 8, "daily"),
    ("openassistant", "openassistant", 11, "persona"),
)


def find_file(name: str) -> Path:
    matches = [Path(path) for path in glob.glob(str(INPUT_ROOT / "**" / name), recursive=True)]
    if not matches:
        raise FileNotFoundError(f"Could not find {name} below {INPUT_ROOT}")
    return matches[0]


def prepare_stage(name: str) -> Path:
    stage = WORK / name
    stage.mkdir(parents=True, exist_ok=True)
    for suffix in ("formatted_chat.txt", "chat_train_tokens.npy", "chat_val_tokens.npy"):
        source = find_file(f"{name}_{suffix}")
        shutil.copy2(source, stage / suffix)
    return stage


def run_stage(name: str, data_name: str, epochs: int, previous: str | None) -> None:
    data_dir = prepare_stage(data_name)
    tokenizer = find_file("v3_tokenizer_chat.json")
    command = [
        "python", "train.py",
        "--data_dir", str(data_dir),
        "--tokenizer", str(tokenizer),
        "--vocab_size", "4099",
        "--n_layers", "5",
        "--context_length", "512",
        "--batch_size", "32",
        "--grad_accum_steps", "2",
        "--epochs", str(epochs),
        "--lr", "3e-4" if previous is None else "1e-4",
        "--warmup_steps", "200",
        "--eval_every", "200",
        "--eval_batches", "100",
        "--checkpoint_dir", str(CHECKPOINTS / name),
        "--device", "cuda",
        "--seed", "42",
    ]
    if previous is not None:
        command.extend(["--resume", str(CHECKPOINTS / previous / "final.pt")])
    print("RUNNING:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    shutil.copy2(find_file("train.py"), WORK / "train.py")
    shutil.copy2(find_file("model.py"), WORK / "model.py")
    shutil.copy2(find_file("dataset.py"), WORK / "dataset.py")
    shutil.copy2(find_file("tokenizer.py"), WORK / "tokenizer.py")
    os.chdir(WORK)
    for stage in STAGES:
        run_stage(*stage)
    print("V3 COMPLETE:", CHECKPOINTS / "openassistant" / "final.pt")
