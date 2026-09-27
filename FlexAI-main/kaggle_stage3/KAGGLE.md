# Kaggle training — the proven workflow

Chat fine-tunes run on Kaggle's free T4 instead of ~6 h on the local CPU.
This document reflects the workflow as actually used for stages 3 → 3d
(Sep 22, 2026), including every gotcha we hit.

## Current state

- **Dataset:** `ishantdahiya1501/stage3` — flat files: `train.py`, `model.py`,
  `dataset.py`, `tokenizer.py`, `train_tokens.npy`, `validation_tokens.npy`,
  `tokenizer_chat.json`, `final.pt` (the resume checkpoint)
- **Kernel:** `ishantdahiya1501/stage3-kaggle` — notebook from
  `kaggle_stage3/stage3_kaggle.ipynb`, GPU T4, output = `stage3_checkpoints.zip`
- **Result:** `checkpoints/s3d/final.pt` — the current default chat model

## One-time setup

1. **Phone verification is MANDATORY for GPU.** Without it, API-pushed kernels
   *silently run on CPU* (no error!). Settings → Phone Verification.
2. **API token:** Settings → API → Create New Token → save as
   `~/.kaggle/access_token` (chmod 600). The classic `~/.kaggle/kaggle.json`
   (`{"username": ..., "key": ...}`) also works.
3. `pip install kaggle` (in `.venv`).
4. Export the token per command (or rely on the file):
   `export KAGGLE_API_TOKEN=KGAT_...`

## Per-run workflow (all from the terminal)

```bash
# 1. Stage the upload dir FLAT (Kaggle datasets SKIP subfolders — this bit us):
kaggle_upload/
  dataset-metadata.json      # {"title": "stage3", "id": "<user>/stage3", "licenses": [{"name": "CC0-1.0"}]}
  train.py model.py dataset.py tokenizer.py
  train_tokens.npy validation_tokens.npy tokenizer_chat.json
  final.pt                   # the checkpoint to RESUME from (73 MB)

# 2. Push dataset (creates a new version), wait for processing:
.venv/bin/kaggle datasets version -p kaggle_upload -m 'msg' --dir-mode skip
.venv/bin/kaggle datasets status <user>/stage3        # wait for 'ready'

# 3. Push the kernel (kaggle_kernel/kernel-metadata.json has enable_gpu: true,
#    dataset_sources: ["<user>/stage3"]):
.venv/bin/kaggle kernels push -p kaggle_kernel

# 4. Poll; ~3-5 min for a chat stage:
.venv/bin/kaggle kernels status <user>/stage3-kaggle   # RUNNING -> COMPLETE

# 5. Pull results (checkpoints zip + the executed notebook + log):
.venv/bin/kaggle kernels output <user>/stage3-kaggle -p kernel_out
unzip -o -j kernel_out/stage3_checkpoints.zip 'checkpoints/s3/final.pt' \
    -d checkpoints/<new_name>/
```

## The notebook's safety rails (learned the hard way)

- **Hard GPU assert** in cell 1: `assert torch.cuda.is_available()` — if Kaggle
  gives a CPU box, the run fails loudly in ~1 min instead of training wrong.
- **Dataset discovery by glob**, not hardcoded path: Kaggle has mounted our
  dataset at `/kaggle/input/stage3` *and* `/kaggle/input/datasets/...` on
  different runs. Cell 1 searches `/kaggle/input/**/final.pt` recursively.
- **Copy out of /kaggle/input** (`shutil.copytree` to /kaggle/working) — the
  input mount is read-only and slow; also lets `train.py` write checkpoints.
- **Checkpoint selection:** the resumed `final.pt`'s epoch counter matters —
  `--epochs` is the *target total*. Resume from a ckpt at epoch N with
  `--epochs N+K` to train exactly K new passes. Set it ≤ N and train.py trains
  **zero** steps (silently "Done").

## Gotchas we hit (so you don't again)

| Gotcha | Symptom | Fix |
|---|---|---|
| No phone verification | `nvidia-smi` missing, session is CPU | verify phone in Settings |
| Pushed kernel before dataset processed | inputs empty / stale mount | poll `datasets status` until `ready` first |
| Subfolders in dataset | CLI: "Skipping folder: ..." | keep the upload dir flat |
| Nested zip in dataset | unzip double-path confusion | raw files, no zip |
| `--epochs` ≤ checkpoint epoch | trains 0 steps, "Done" instantly | epochs = ckpt_epoch + passes |
| Quota view | `kaggle kernels push --help` shows `--accelerator` | 30 GPU-h/week, check `quota_view()` |

## Useful API snippets

```python
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
r = api.quota_view()                       # weekly GPU quota
q = r.gpu_quota
print(q.time_used, "/", q.total_time_allowed)
st = api.kernels_status("<user>/stage3-kaggle")   # st.status, st.failureMessage
```

## After pulling a new model

```bash
.venv/bin/python sample.py --chat                       # if it's now the default
PYTHONPATH=. .venv/bin/python junk/buftest/chat_probe.py  # quality probe (edit CKPT)
```

And the key diagnostic for reply stability: measure
`logit("Hi") − logit("What")` after `<|user|>hi<|assistant|>` —
near 0 means sampling will scatter; ≥ ~1 nat means stable.
