# Tuff potato — a ~6M-param built from scratch (stories → chat) + phase roadmap

> **🤖 AGENT HANDOFF — READ THIS FIRST**
> The owner (Ishant) builds and trains this LLM with an AI agent (Codebuff/Buffy)
> as a pair-programming partner. This file is the source of truth. If you are a
> new agent session: read this whole file, then continue naturally — the owner
> speaks casual English, wants honest engineering answers (never oversell),
> likes understanding *why*, and explicitly rejects hardcoded reply logic.
> Verify claims with measurements (probes, logit gaps), not vibes.
>
> **📜 STANDING RULE (owner-mandated):** EVERY change made and EVERY new idea
> proposed — no matter how small — MUST be written into this README
> ("Changelog" + "Ideas parking lot" sections below) in the same session it
> happens. The README is the project's memory; if it's not here, it didn't
> happen. Update it BEFORE ending the session.

## CURRENT STATE (updated 2026-09-22 ~17:45) — PHASE 2 COMPLETE

- **Stage 4 trained + verified + installed as default** (`checkpoints/s4/final.pt`,
  epoch-17 weights + 4 intent/memory passes on corpus v6). Head-to-head vs
  ep17: sampled 13 vs 10/21, knowledge 4 vs 2/8, greedy tied 14/14, counting
  dipped 1. Full-stack suite 34/35 (only "ok so…" prefix-paraphrase fails).
- **🎯 RAW-NN MEMORY ACHIEVED** — the network itself (mind.py OFF) now does
  disclosure → ack → recall: **6/6 greedy, 24/24 across temp 0.3/0.5/0.7** on
  unseen names. This was the phase-2 headline goal.
- **mind.py = 13 intents** (GREETING/FAREWELL/SELF_DISCLOSURE/MEMORY_QUERY/
  THANKS/APOLOGY/EMOTION_SHARE/IDENTITY/CAPABILITY/AFFIRMATION/SMALLTALK/
  QUESTION/OTHER), 35-case eval suite, corpus v6 with in-weights memory data.
- **Next: phase 2B proper** — LoRA rank-8 adapters on attention (or paraphrase-
  augmentation corpus v7) targeting the two stubborn fails: "ok so…"-prefix
  paraphrases and typo robustness ("farance"). See Ideas parking lot.

- **THE LONG KAGGLE RUN IS COMPLETE** (epoch 17, step 6399) and a graded
  model-vs-model comparison was run (`compare_models.py`, 21 prompts, simple →
  hard, greedy + sampled, auto-checked). Verdict: the new model **knows more**
  (knowledge cases 0→2/8 raw) but **samples more loosely** (raw sampled
  13→10/21 at temp 0.5; 12/21 at temp 0.3). Through the full stack (mind.py +
  NN) it **wins: 21/22 (95%) vs s3d's 20/22**, including the memory case.
- **Default chat model switched to `checkpoints/longrun/final.pt`** (epoch 17);
  chat temperature default lowered to **0.3** (measured: 10/21 at 0.5, 12/21 at
  0.3 raw — the flatter distribution is a known cost of 6× more diverse data).
  Old default `checkpoints/s3d/final.pt` is kept and still wins raw greedy
  consistency drills; nothing is deleted.
- **PHASE 2A + 2C ARE DONE** (see Changelog): `mind.py` cognition layer is live
  in `sample.py --chat` (memory works: "my name is Ravi" → "Nice to meet you,
  Ravi!" → "what is my name?" → "Your name is Ravi!", persisted in memory.json).
  `eval_suite.py` scores the stack. Run it after every change.
- **Agreed next work (phase 2B, on Kaggle):** LoRA adapters on attention layers
  (freeze base, train rank-8 inserts) on intent/memory data; optional scale-up
  d_model 256→512, layers 5→8, FFN→2048 (~25M params, full retrain).
- **Explicit owner decisions:** memory *training data* is deferred until after
  2A/2B — first we do the code-based memory. No hardcoded replies ever
  (templates live only in `mind.py` intent layer, never in inference of the NN).
- What "hi" getting answered correctly taught us: data composition (150× social
  oversample + 8K greeting-opener convs) fixed it; metric = logit gap
  ("Hi"−"What" after `<|user|>hi<|assistant|>`: +0.02 → +0.92 nat).

## What the model is

A from-scratch GPT-style decoder-only Transformer in PyTorch (no HF
`transformers`). Hand-written in `model.py`: multi-head self-attention, RoPE,
RMSNorm, SwiGLU, tied embeddings. Pretrained on TinyStories (fluent child-level
English), then fine-tuned for chat with `<|user|>/<|assistant|>/<|end|>` tokens
on a curated chat-dominant corpus.

| Component | Value | | Component | Value |
|---|---|---|---|---|
| Layers | 5 (grown 4→5 in v3) | | Vocab | 4096 BPE + 3 chat = **4099** |
| d_model | 256 | | Context | 512 (grown 256→512) |
| Heads | 8 (head_dim 32) | | FFN | SwiGLU 1024 |
| Pos | RoPE θ=10000 | | Norm | RMSNorm, no biases |
| **Params** | **6,295,040** | | Tied | tok_emb == lm_head |

## Project structure (every file, what it does)

```
Ai_v2/
├── model.py               # the NN: TinyGPT, attention, RoPE, generate(), save/load ckpt
├── train.py               # training loop: AdamW, cosine+warmup, AMP, resume logic
│                          #   (grow-layers / resize-embedding / fresh-schedule fine-tune branch)
├── dataset.py             # prepare_tokens(): caches {split}_tokens.npy (chat_* legacy fallback)
├── tokenizer.py           # BPE tokenizer train/load helpers
├── sample.py              # CLI: --chat (default ckpt s3d, temp 0.5) or --prompt stories
│                          #   --no-mind bypasses the cognition layer; --show-intent debugs it
├── mind.py                # ⭐ PHASE 2A: cognition layer — intent classifier (GREETING /
│                          #   FAREWELL / SELF_DISCLOSURE / MEMORY_QUERY / QUESTION / OTHER),
│                          #   MemoryStore (memory.json), fact extractor (name/age/likes/
│                          #   dislikes/location/favorite X), memory injection, best-of-3
│                          #   reranking, intent-aware temperature. ONLY scripted fallbacks live here.
├── eval_suite.py          # ⭐ PHASE 2C: 22-case behavior suite (greeting/memory/facts/
│                          #   robust/hygiene tags), checker fns, exit code = pass/fail
├── make_overnight_data.py # builds data/overnight corpus; flags: --social_mult 150,
│                          #   --n_greeting_openers 8000, --include_all_dolly (full 66.9K)
├── tokenize_chat.py       # conversations -> packed uint16 train/validation_tokens.npy (95/5)
├── add_chat_tokens.py     # adds <|user|>/<|assistant|>/<|end|> to tokenizer (4096->4099)
├── download_chat_data.py  # dolly/alpaca + TinyStoriesInstruct download
├── format_chat.py         # formats raw convs into the chat template, one per line
├── run_all.py             # legacy: runs the chat data prep chain in order
├── run_v3.sh              # local CPU chain: stage1 (grow model) + stage2 (chat FT)  [superseded]
├── run_v3_stage3.sh       # local CPU stage-3 (kept for reference; Kaggle is the way now)
├── kaggle_stage3/         # THE KAGGLE BUNDLE
│   ├── stage3_kaggle.ipynb    # the kernel notebook (hard GPU assert, recursive dataset glob)
│   └── KAGGLE.md              # full Kaggle playbook incl. all gotchas we hit
├── kaggle_upload/         # flat staging dir pushed as dataset "stage3" (code+data+final.pt)
├── kaggle_kernel/         # kernel-metadata.json + notebook copy for `kaggle kernels push`
├── junk/
│   ├── buftest/chat_probe.py   # quality probe (greedy replies, memorization, story test)
│   └── backups/, debug_scripts/
├── data/
│   ├── tinystories/       # base corpus + 4096-vocab tokenizer.json + 485M-token cache
│   ├── chat/              # legacy 995K-token chat data + tokenizer_chat.json (4099)
│   ├── chat_v3/           # 9.2M-token dolly+alpaca (stage-2 data; source for --include_all_dolly)
│   └── overnight/         # CURRENT corpus: train/validation_tokens.npy + formatted txt
├── checkpoints/
│   ├── latest.pt          # base TinyStories model (4L, ctx256, vocab4096, epoch2/step27000)
│   ├── chat_run_v3/       # stage-2 chat model (val 2.95) — historical
│   ├── s3/                # stage-3 (val 0.887) — historical
│   ├── s3c/               # social-50x iteration — historical
│   ├── s3d/               # ⭐ CURRENT DEFAULT (epoch 10) — also uploaded to Kaggle as resume ckpt
│   └── s3b/               # (may be deleted; was intermediate)
├── prompt.txt             # earlier debug-session notes (stuck-loss bug history)
├── TRAINING_GUIDE.md      # CPU / 840M / Kaggle how-to (Option K = Kaggle, recommended)
└── kaggle_stage3/KAGGLE.md# the battle-tested Kaggle workflow + gotcha table
```

## Commands the owner uses daily

```bash
.venv/bin/python sample.py --chat                 # chat (defaults: longrun epoch-17 model, temp 0.3)
.venv/bin/python sample.py --temperature 0 --chat # deterministic
.venv/bin/python sample.py --temperature 0.8 --prompt "Once upon a time"  # story mode
.venv/bin/python eval_suite.py                    # full-stack score (now 21/22 on the default)
.venv/bin/python eval_suite.py --checkpoint checkpoints/s3d/final.pt  # score the old model
SAMPLE_TEMP=0.5 .venv/bin/python compare_models.py    # head-to-head: s3d vs longrun

# Kaggle loop (full playbook: kaggle_stage3/KAGGLE.md)
export KAGGLE_API_TOKEN=KGAT_...
.venv/bin/kaggle datasets version -p kaggle_upload -m 'msg' --dir-mode skip
.venv/bin/kaggle datasets status ishantdahiya1501/stage3      # wait: ready
.venv/bin/kaggle kernels push -p kaggle_kernel
.venv/bin/kaggle kernels status ishantdahiya1501/stage3-kaggle
.venv/bin/kaggle kernels output ishantdahiya1501/stage3-kaggle -p kernel_out
```

Venv: `.venv` (torch 2.14+cpu, tokenizers, kaggle). System python has NO torch.
Kaggle auth lives at `~/.kaggle/access_token`. Disk is TIGHT (~500MB free) —
clean `kernel_out/` and old zips before big pulls.

## Training history (what produced the current model)

| Stage | Data | Passes | Result |
|---|---|---|---|
| Base pretrain | TinyStories (5.7M tok seen) | 2 ep / 27k steps | val 1.654; fluent English |
| v3 stage 1 (local) | TinyStories | 1 ep | grew 4→5 layers, ctx 256→512 |
| v3 stage 2 (local) | dolly+alpaca 9.2M tok | 1 ep / 1126 steps | val 2.95; English ✓ chat ✗ |
| stage 3 (Kaggle 3.4 min!) | chat corpus 1.72M tok | 3 ep / 396 steps | val 0.887; facts work |
| 3b/3c (Kaggle) | + social 15×/50× | +4 ep | gap +0.02→+0.12; still coin-flip |
| 3d (Kaggle) | + social 150× + 8K openers | +2 ep / 1100 steps | val 0.77; gap +0.92 ✅ default |
| long run (Kaggle ~30 min) | corpus v5 13.08M tok (+full dolly 66.9K convs) | 7 ep (epoch 11→17, step 6399) | val 2.30 (on the NEW 13M-token val set — harder, not comparable to s3d's 0.77); ⭐ CURRENT default |

> Note: eval checkpoints are every 250 steps: `checkpoints/` from the run has
> epoch11..17 + final. Only `final.pt` (epoch 17) was pulled locally to
> `checkpoints/longrun/`; the rest stayed in the kernel output (disk budget).

## Hard-won lessons (do not rediscover these)

1. **Resuming with `--epochs` ≤ checkpoint epoch trains ZERO steps.** `--epochs`
   is the target *total*. Resume from epoch N with `--epochs N+K` for K passes.
2. **lr=0 freeze**: fine-tuning a long-pretrained ckpt on a short schedule used
   to restore a dead cosine schedule. train.py now starts fresh when the run's
   total steps ≤ ckpt step. (Was a real multi-day bug — see prompt.txt.)
3. **Kaggle without phone verification silently gives CPU** (no GPU, no error).
   Assert `torch.cuda.is_available()` in the notebook (it does).
4. **Kaggle datasets skip subfolders** — the upload dir must be flat. The
   dataset mount path varies (`/kaggle/input/stage3` vs `/kaggle/input/datasets/…`)
   — the notebook globs recursively for `final.pt`.
5. **Greedy-correct ≠ sampling-stable.** Always measure the first-token logit
   gap for critical intents; ±0.02 nat = coin flip under temp 0.7.
6. **Small models memorize; data composition is the steering wheel.** Oversample
   the behavior you want (150× for social), include the *context shape* you want
   (greeting-opener conversations for first-turn greetings).
7. **The model loads fully at once** (strict load, 38/38 tensors, all finite).
   If a reply looks random, it's the sampling distribution, not a load problem.
8. **TinyStories pretraining is why English is fluent.** Don't skip base
   pretraining when rebuilding; don't over-index on chat data at its expense.
9. **More/diverse data flattens the sampling distribution.** After the 6× data
   bump the raw sampled score DROPPED (13→10/21) even as knowledge rose — more
   knowledge ≠ sharper replies. Re-measure the temperature after any big data
   change (0.3 is the new chat default), and remember `mind.py` reranking
   carries real score: full-stack 21/22 even though raw sampled is 12/21.

## Changelog (STANDING RULE: append every change here, same session)

### 2026-09-22
- **[V3 DATA]** Built a fresh chat-first corpus pipeline for sequential training:
  DailyDialog (13,118 conversations), Persona-Chat (139,239 examples), and
  filtered English OpenAssistant (20,423 examples). A shared BPE tokenizer has
  4,096 base tokens plus `<|user|>`, `<|assistant|>`, and `<|end|>` (4,099 total).
- **[V3 KAGGLE]** Uploaded the 216 MB flat dataset as
  `ishantdahiya1501/chat-v3-from-zero-data` and launched the private GPU kernel
  `ishantdahiya1501/chat-v3-from-zero-kaggle`. It trains fresh DailyDialog first,
  then resumes through Persona-Chat and OpenAssistant into separate checkpoint
  directories; no V2 checkpoint is included. Epoch targets are 5 -> 8 -> 11,
  because `train.py --epochs` means total target epochs when resuming.
- **[2A] Created `mind.py`** — cognition layer: `classify_intent()` (priority:
  facts > memory-queries > farewell > greeting > question > other),
  `extract_facts()` (name/age/likes/dislikes/location/favorite-X regexes with
  state-word filter so "i am fine" ≠ a name), `MemoryStore` (memory.json
  persistence, `--file -` = RAM only), intent acks ("Nice to meet you, Ravi!"),
  memory-injection fallback for unknown queries, best-of-3 reranking with
  per-intent scoring (greeting must start with a greeting word; repetition
  penalized), `_pick()` deterministic template rotation.
- **[2A] Wired into `sample.py`** — `--chat` now routes through `Mind.handle()`;
  new flags `--no-mind` (raw model) and `--show-intent` (debug prints);
  graceful fallback if mind.py is missing.
- **[2C] Created `eval_suite.py`** — 22 cases / 5 tags (greeting, memory, facts,
  robust, hygiene), checker combinators (`contains`, `contains_any`,
  `starts_with_any`, `not_matches`, `word_count_between`, `ends_properly`),
  `-k <tag>` filter, `-v` verbose, `--no-mind` raw-model mode, exit code 0/1.
- **[BUGFIX] disclosure-vs-query priority** — "my name is Ravi" was classified
  MEMORY_QUERY (its own "my name" matched the query regex!) so it was never
  stored. Fix: `_has_fact()` checked FIRST in `classify_intent()`. Caught by
  our own chat test, confirmed fixed by eval suite (memory 6/6).
- **[BUGFIX] None-history crash** — memory-query fallback path passed
  `history=None` into the encoder. Fix: `history = history or []`.
- **[BUGFIX] rerank threshold** — a "...?" bonus let "Ankara is the capital of
  which country?" win the GREETING rerank for "yo!". Fix: GREETING/FAREWELL
  require score ≥ 2 (an actual greeting-word start). "yo!" now passes.
- **[BASELINE] eval_suite on s3d: 20/22 (91%)**. Remaining fails = model-level:
  (a) "Count to five." → "1, 2, 3, 4! That is how you count to 4" (memorization
  slip), (b) "ok so what is the capital of france??" → "capital of frantic
  ishi" (paraphrase fragility). Both are 2B targets, not wrapper gaps.
- **[TRAINING] Kaggle long run launched 13:43 (kernel v9)**: corpus v5 13.08M
  tokens (added `--include_all_dolly`: full 66,926-conv dolly+alpaca set),
  resume s3d@epoch10, --epochs 17 (7 passes), eval+ckpt every 250 steps.
  Owner's Colab anecdote (5h → 2.7/3 epochs on TinyStories) prompted the
  resize: old 2.1M-token corpus would have been ~200 passes in 3h = overfit.
- **[TRAINING] Long run COMPLETE** — pulled from kernel (after two disk-full
  download failures; owner freed 3 GB and the full 747 MB output landed).
  Real trained model identified as `checkpoints/s3/final.pt` inside the output
  (epoch 17, step 6399); the top-level `final.pt` in the output was just the
  s3d resume input (epoch 10) — first grab was the wrong file, caught by
  checking epoch metadata before comparing.
- **[EVAL] Created `compare_models.py`** — graded side-by-side battery (21
  prompts: facts/greeting/social/counting/knowledge/paraphrase/typo/memory/
  longform), each model answering greedy (temp 0) + sampled (temp
  `$SAMPLE_TEMP`, default 0.5, seed 42), auto-checkers (any/all/min_words) + a
  6-gram loop detector. Used for the s3d-vs-longrun verdict below.
- **[VERDICT] s3d vs longrun (epoch 17)**: raw greedy 13 vs 14/21; raw sampled
  13 vs 10/21 @0.5 (12 @0.3). Knowledge 0→2/8, counting scattered, memory 0/4
  raw for BOTH (raw NN never learned name recall — that's mind.py's job),
  stories fine. Full stack: **longrun 21/22 (95%) vs s3d 20/22** — longrun
  becomes default.
- **[DEFAULT] `sample.py` default checkpoint → `checkpoints/longrun/final.pt`,
  chat temp default → 0.3** (flatter sampling after 6× data; measured, see
  verdict). s3d kept on disk.
- **[2A] mind.py expanded to 13 intents** — added THANKS / APOLOGY /
  EMOTION_SHARE / IDENTITY / CAPABILITY / AFFIRMATION / SMALLTALK, with
  `EXPANDED_SOCIAL` table + `SOCIAL_PREFIXES` fallbacks and a new
  `_social_reply()` router (scripted surface first, NN best-of-3 fallback),
  plus per-intent rerank scoring (empathy must be about the USER; identity
  must self-describe; affirmation must lead with a yes/no word). Capability
  pattern deliberately excludes capability-phrased content questions
  ("can you tell me a story" stays QUESTION → NN tells a story). Classifier
  unit test: 20/20 phrases → correct intents.
- **[2C] eval_suite: 22 → 35 cases** — new `social` tag covers all added
  intents; default checkpoint repointed to longrun. New baseline: **34/35
  (97%)** on longrun ep17 — the only fail is the known "ok so…" paraphrase
  (NN-level, target of the stage-4 fine-tune).
- **[DATA] corpus v6** (135,300 convs / 14.09M tok): `NNSOCIAL` expanded
  social pairs synced with mind.py's surface, and **7 in-weights memory
  generators** (name/likes/age/location disclosure → ack → recall 1–3 turns
  later, incl. recall-opener convs and social-mix convs; `--n_memory 12000`),
  so the RAW network learns entity persistence, not just the code layer.
- **[TRAINING] Stage 4 launched 16:42 (kernel v10)**: resume longrun ep17 →
  4 fresh passes over corpus v6, LR 2.5e-5 fresh cosine, eval every 250 steps.
  Fine-tune mode auto-triggers (ckpt step 6399 > 4-epoch run total). Notebook
  probe cell extended with the new intents + raw memory recall checks.
- **[BUGFIX] notebook cp direction error** — a careless `cp` overwrote the
  edited kernel with a stale copy; caught by the verification print
  (`--epochs 4 in kernel: False`), re-applied, and both copies re-synced.
- **[MONITOR] Stage 4 COMPLETE at 17:11** (29 min on the T4): correct resume
  verified from log (epoch 17, step 6399, fresh-schedule fine-tune mode),
  val_loss 2.05 → **2.0253** on corpus v6's val split, kernel probe cell green
  on all 9 checks incl. raw memory recall.
- **[BUGFIX] eval_suite --no-mind crash** — the raw-mode branch imported the
  deleted helper `_model_replies`; replaced with a local `raw_reply()`.
- **[EVAL] Raw-mode suite caveat recorded**: memory-tag cases FAIL in --no-mind
  mode by construction (each case runs with empty history, but recall needs the
  disclosure turn first). Real raw-memory test = the multi-turn recall probe:
  6/6 greedy, 24/24 sampled.
- **[EVAL] compare_models.py head-to-head (longrun ep17 vs s4)**: greedy
  14/14 tie; sampled@0.5 **13 vs 10**; knowledge **4 vs 2**; memory **2 vs 0**;
  counting 2 vs 3 (one dip); greeting/social/facts/longform all held.
- **[DEFAULT] sample.py default → `checkpoints/s4/final.pt`**. Live smoke test
  (show-intent): GREETING→greet, SELF_DISCLOSURE→ack+store, MEMORY_QUERY→
  recall, THANKS/IDENTITY/CAPABILITY→correct scripted replies. s4 is the new
  baseline; compare_models.py OLD/NEW labels now read longrun vs s4.
- **[TEMP RAW MODE]** Backed up the cognition layer as `mind.py.bak` and changed
  active `Mind.handle()` to bypass intent routing, memory, scripted responses,
  reranking, and fallbacks. Normal `sample.py --chat` now generates directly
  from the model; the original cognition behavior remains recoverable from the
  backup. Smoke-tested with s4 using `hello` and `--temperature 0`.

### 2026-09-21
- v3 chain ran locally (stage 1 grow + stage 2 chat FT); run_v3.sh was edited
  mid-run by a previous session → transient `d_accum_steps` bash error after
  final.pt saved (model itself finished fine).
- Evening session built make_overnight_data.py + first overnight corpus but
  never launched the follow-up training (the "missing overnight run" mystery).

## Ideas parking lot (write ideas here the moment they pop)

- **From the owner:** make the NN "10x better" and a proper memory system;
  model must *feel* like it distinguishes greeting / question / self-intro /
  something-to-remember. (2A code layer does this now; 2B LoRA teaches the NN.)
- **Paraphrase augmentation** (2B data): same question, 5–10 phrasings + casing
  /punct/typo variants — kills the "capital of farance"-class fragility.
- **Conversations that start with arbitrary user facts** ("hi I'm Sam, I like
  dinosaurs") so the NN itself learns disclosure→ack+store patterns (2B data).
- **memory.json → per-user profiles** if the chat ever has multiple users.
- **Scale-up (2B option):** d_model 512 / 8 layers / FFN 2048 ≈ 25M params.
- **Persistent sessions:** `--resume-chat` flag to reload history+memory.json.
- **max-reply-length guard** from prompt.txt (reject replies < N tokens before
  accepting <|end|>) — still not implemented.

## Phase roadmap (agreed with the owner)

- **Phase 2A — Cognition Layer (code only, NEXT):** `mind.py` with intent router
  (regex/keyword classifier), memory store (name/likes/age/…, session + optional
  file persistence), memory injection into prompts, best-of-3 reranking for
  greeting/memory intents, per-intent temperature. Edit `sample.py` to call it.
- **Phase 2C — Eval Suite (code only, with 2A):** `eval_suite.py`, ~30 cases
  across intents × paraphrases, assertions on behavior (contains "Ravi",
  starts with greeting…), exit code = pass/fail. Run after every change.
- **Phase 2B — model-level (Kaggle):** LoRA (rank-8 on attention) trained on
  intent/memory data so the NN itself learns the categories; then optional
  scale-up (~25M params, d_model 512, 8 layers) with full retrain.
- **Phase 3 (ideas):** broader pretrain mix (simple Wikipedia), bigger context,
  RLHF/DPO experiments, persistence across sessions.

## Notes for agents continuing this project

- The owner tests in `sample.py --chat` and pastes transcripts back; treat his
  reports as ground truth and reproduce before fixing.
- Never claim something is "hardcoded" or "loaded wrong" without checking —
  several times the real cause was data composition or schedule math.
- Checkpoints are precious; don't delete any without asking. Free `kernel_out/`
  and zips instead (disk is the binding constraint).
- After any corpus change: regenerate → tokenize → push dataset → *wait for
  `ready`* → push kernel → poll → pull → probe. ~5 min of mechanism, 3 min of
  compute. The probe + logit-gap check decides whether a model becomes default.
