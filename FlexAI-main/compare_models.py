"""Compare two TinyGPT chat checkpoints on the same graded prompt battery.

Old model:  checkpoints/s3d/final.pt   (epoch 10 — the current default)
New model:  checkpoints/longrun/final.pt (epoch 17 — the 7-pass long run)

Every prompt is answered twice per model:
  * greedy  (temperature 0          — the model's best possible answer)
  * sampled (temperature 0.5, seed 42 — the real chat-loop condition)

Each case carries check rules so scoring is automatic, not vibes:
  {"any": [...]}  -> reply must contain at least one needle (lowercased)
  {"all": [...]}  -> reply must contain every needle
  {"min_words": N} -> reply must be at least N words long

Run:  .venv/bin/python compare_models.py
"""

import os
import re

import torch

from model import ModelConfig, TinyGPT, generate, resolve_device
from tokenizer import load_tokenizer

OLD_CKPT = "checkpoints/longrun/final.pt"   # ep17 (pre-stage-4)
NEW_CKPT = "checkpoints/s4/final.pt"        # ep21-equiv (stage-4 intent+memory FT)
TOKENIZER = "data/chat/tokenizer_chat.json"

# ---------------------------------------------------------------- prompt battery
# Ordered simple -> hard so the verdict shows *where* the new model gains.
CASES = [
    # -- level 1: facts drilled into the corpus -------------------------------
    ("facts",      "what is the capital of France?",                       {"any": ["paris"]}),
    ("facts",      "what is the opposite of hot?",                         {"any": ["cold"]}),
    ("facts",      "what sound does a dog make?",                          {"any": ["woof", "bark"]}),
    ("greeting",   "hi",                                                   {"any": ["hi", "hello", "hey"]}),
    ("greeting",   "good morning",                                         {"any": ["morning"]}),
    ("social",     "how are you?",                                         {"any": ["good", "great", "well", "nice", "fine"]}),
    ("social",     "what is your name?",                                   {"any": ["buddy"]}),
    # -- level 2: arithmetic / counting ---------------------------------------
    ("counting",   "what is 6 plus 3?",                                    {"any": ["9", "nine"]}),
    ("counting",   "count to five",                                        {"all": ["1", "2", "3", "4", "5"]}),
    ("counting",   "if i have 3 apples and eat 1, how many are left?",     {"any": ["2", "two"]}),
    # -- level 3: real dolly-style knowledge (rarely drilled before) ----------
    ("knowledge",  "who wrote romeo and juliet?",                          {"any": ["shakespeare"]}),
    ("knowledge",  "what is photosynthesis?",                              {"any": ["light", "plant", "sun", "energy", "oxygen"]}),
    ("knowledge",  "how do i make a paper airplane?",                      {"any": ["fold", "paper"]}),
    ("knowledge",  "what is the largest planet?",                          {"any": ["jupiter"]}),
    # -- level 4: robustness — phrasings the corpus never showed --------------
    ("paraphrase", "ok so um what's the capital of france??",              {"any": ["paris"]}),
    ("paraphrase", "hey so like what sound does a dog make",               {"any": ["woof", "bark"]}),
    ("typo",       "what is the capital of farance",                       {"any": ["paris"]}),
    # -- level 5: memory inside the raw network (no mind.py wrapper) ----------
    ("memory",     ("my name is ravi", "what is my name?"),            {"any": ["ravi"]}),
    ("memory",     ("i like cricket", "what do i like?"),              {"any": ["cricket"]}),
    # -- level 6: long-form generation quality --------------------------------
    ("longform",   "tell me a story about a dragon",                       {"min_words": 30}),
    ("longform",   "why is the sky blue?",                                 {"any": ["light", "sun", "air", "scatter", "blue", "sky"]}),
]


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ModelConfig(**ckpt["config"])
    model = TinyGPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    meta = {k: ckpt.get(k) for k in ("epoch", "global_step", "val_loss") if ckpt.get(k) is not None}
    return model, meta


@torch.no_grad()
def _gen(model, tokenizer, end_id, history, user_input, device, temperature, max_tokens=120):
    """One chat completion over a rendered history (exact sample.py format)."""
    parts = [f"<|user|>{u}<|assistant|>{a}<|end|>" for u, a in history]
    parts.append(f"<|user|>{user_input}<|assistant|>")
    ids = tokenizer.encode("".join(parts)).ids[-model.cfg.context_length:]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(model, idx, max_new_tokens=max_tokens, temperature=temperature,
                   top_k=50, eos_id=end_id, repetition_penalty=1.15)
    new_ids = out[0].tolist()[len(ids):]
    if end_id in new_ids:
        new_ids = new_ids[:new_ids.index(end_id)]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def reply(model, tokenizer, end_id, turns, device, temperature, max_tokens=120):
    """Walk the conversation: real generated replies for earlier turns, like the chat loop."""
    history = []
    for t in turns[:-1]:
        history.append((t, _gen(model, tokenizer, end_id, history, t, device,
                                temperature, max_tokens)))
    return _gen(model, tokenizer, end_id, history, turns[-1], device, temperature,
                max_tokens)


def check(case, reply_text):
    """True if reply passes the case's rule; loops count as failure."""
    text = reply_text.lower()
    words = re.findall(r"[a-z0-9']+", text)
    # loop detector: any 6-gram repeated 3+ times = degenerate output
    grams = [" ".join(words[i:i + 6]) for i in range(len(words) - 6)]
    if grams and max(grams.count(g) for g in set(grams)) >= 3:
        return False
    if "any" in case and not any(n in text for n in case["any"]):
        return False
    if "all" in case and not all(n in text for n in case["all"]):
        return False
    if "min_words" in case and len(words) < case["min_words"]:
        return False
    return True


def main():
    device = resolve_device("auto")
    tokenizer = load_tokenizer(TOKENIZER)
    end_id = tokenizer.token_to_id("<|end|>")

    print("Loading models…")
    old_model, old_meta = load_model(OLD_CKPT, device)
    new_model, new_meta = load_model(NEW_CKPT, device)
    print(f"OLD  {OLD_CKPT}  {old_meta}")
    print(f"NEW  {NEW_CKPT}  {new_meta}\n")

    score = {k: {"old": 0, "new": 0} for k in ("greedy", "sampled")}
    cat_score = {}
    detail = []

    for label, prompt, rule in CASES:
        turns = list(prompt) if isinstance(prompt, tuple) else [prompt]
        shown = " >> ".join(turns)
        row = {"cat": label, "prompt": shown, "rule": rule}

        for name, temp, seed in (("greedy", 0.0, None),
                                 ("sampled", float(os.environ.get("SAMPLE_TEMP", "0.5")), 42)):
            for tag, model in (("old", old_model), ("new", new_model)):
                if seed is not None:
                    torch.manual_seed(seed)
                r = reply(model, tokenizer, end_id, turns, device, temp)
                ok = check(rule, r)
                score[name][tag] += int(ok)
                cat_score.setdefault(label, {"old": 0, "new": 0})[tag] += int(ok)
                row.setdefault(name, {})[tag] = (r, ok)

        detail.append(row)

    # ---------------- per-case printout ---------------------------------------
    for row in detail:
        marks = {d: {"old": "✅" if row[d]["old"][1] else "❌",
                     "new": "✅" if row[d]["new"][1] else "❌"} for d in ("greedy", "sampled")}
        print(f"\n[{row['cat']}] {row['prompt']}")
        for d in ("greedy", "sampled"):
            for tag in ("old", "new"):
                r, _ = row[d][tag]
                short = r if len(r) <= 110 else r[:107] + "…"
                print(f"  {marks[d][tag]} {d:7s} {tag}: {short}")

    # ---------------- verdict tables ------------------------------------------
    total = len(CASES)
    print("\n" + "=" * 72)
    import os
    print(f"{'':14s}{os.path.basename(os.path.dirname(OLD_CKPT)):>12s}"
          f"{os.path.basename(os.path.dirname(NEW_CKPT)):>16s}")
    print(f"{'greedy':14s}{score['greedy']['old']:>9d}/{total}{score['greedy']['new']:>13d}/{total}")
    st = float(os.environ.get("SAMPLE_TEMP", "0.5"))
    print(f"{'sampled@'+os.environ.get('SAMPLE_TEMP','0.5'):14s}"
          f"{score['sampled']['old']:>9d}/{total}{score['sampled']['new']:>13d}/{total}")
    print("-" * 72)
    cats = sorted(cat_score)
    n_by_cat = {c: 2 * sum(1 for l, *_ in CASES if l == c) for c in cats}  # greedy + sampled
    for c in cats:
        print(f"{c:14s}{cat_score[c]['old']:>9d}/{n_by_cat[c]}{cat_score[c]['new']:>13d}/{n_by_cat[c]}")


if __name__ == "__main__":
    main()
