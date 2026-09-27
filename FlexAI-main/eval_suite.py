"""Phase 2C — the Eval Suite: objective behavior checks for the chat model.

Tests the FULL stack (mind.py cognition layer + NN) by default; --no-mind tests
the raw model through the same interface. One command -> pass/fail score, so
every future change (LoRA, scale-up, corpus edits) gets an objective number.

Run:
    .venv/bin/python eval_suite.py                 # full suite (mind on)
    .venv/bin/python eval_suite.py -k memory       # only cases tagged 'memory'
    .venv/bin/python eval_suite.py --no-mind       # raw model comparison
    .venv/bin/python eval_suite.py -v              # show every reply
"""

from __future__ import annotations

import argparse
import sys

import torch

from model import ModelConfig, TinyGPT, resolve_device
from tokenizer import load_tokenizer

DEFAULT_CKPT = "checkpoints/longrun/final.pt"
DEFAULT_TOK = "data/chat/tokenizer_chat.json"


# --------------------------------------------------------------------- checks
def contains(*needles, case_insensitive=True):
    def check(reply, ctx):
        hay = reply.lower() if case_insensitive else reply
        return all(n.lower() in hay for n in needles)
    return check


def contains_any(*needles, case_insensitive=True):
    def check(reply, ctx):
        hay = reply.lower() if case_insensitive else reply
        return any(n.lower() in hay for n in needles)
    return check


def starts_with_any(*prefixes):
    def check(reply, ctx):
        low = reply.lower().lstrip()
        return any(low.startswith(p.lower()) for p in prefixes)
    return check


def not_matches(*regexes):
    def check(reply, ctx):
        return not any(r.search(reply) for r in regexes)
    return check


def word_count_between(lo, hi):
    def check(reply, ctx):
        return lo <= len(reply.split()) <= hi
    return check


def nonempty(min_words=1):
    def check(reply, ctx):
        return len(reply.split()) >= min_words
    return check


def ends_properly(reply, ctx):
    return bool(reply) and reply[-1] in ".!?\"')"


# ------------------------------------------------------------------- the suite
# Each case: (tag, user input, [checks]). Checks get (reply, ctx) where ctx
# carries the stored memory BEFORE this case ran (so memory cases can assert
# facts the extractor should have picked up from earlier turns).
import re
RE_WARNING = re.compile(r"\b(Are you|is that) a warning\b", re.I)
RE_QUESTION_WORD = re.compile(r"\b(what|why|how|where|when|who)\b", re.I)

SUITE = [
    # --- greetings (short openers must get greeting-style replies)
    ("greeting", "hi",             [starts_with_any("hi", "hello", "hey", "good")]),
    ("greeting", "hello",          [starts_with_any("hi", "hello", "hey", "good")]),
    ("greeting", "hey there",      [starts_with_any("hi", "hello", "hey", "good")]),
    ("greeting", "good morning",   [starts_with_any("good", "hi", "hello", "morning")]),
    ("greeting", "yo!",            [starts_with_any("hi", "hello", "hey", "yo", "good")]),

    # --- self-disclosure -> ack + stored
    ("memory", "my name is Ravi",
     [contains("ravi"), not_matches(RE_WARNING)]),
    ("memory", "I'm 17 years old",
     [starts_with_any("got", "noted", "cool", "nice")]),
    ("memory", "i like cricket",
     [starts_with_any("got", "noted", "cool", "nice", "that")]),

    # --- memory recall (only meaningful after the disclosures above ran)
    ("memory", "what is my name?",
     [contains("ravi")]),
    ("memory", "how old am i?",
     [contains("17")]),
    ("memory", "what do i like?",
     [contains("cricket")]),

    # --- facts (the NN's memorized knowledge, via mind QUESTION path)
    ("facts", "What is the capital of France?", [contains("paris")]),
    ("facts", "What sound does a dog make?",    [contains_any("woof", "bark")]),
    ("facts", "What is the opposite of hot?",   [contains("cold")]),
    ("facts", "What month comes after June?",   [contains("july")]),
    ("facts", "Count to five.",                 [contains_any("5", "five")]),

    # --- robustness: paraphrases/typos must not derail into nonsense
    ("robust", "what is my nam?",
     [contains("ravi")]),
    ("robust", "tell me my name please",
     [contains("ravi")]),
    ("robust", "ok so what is the capital of france??",
     [contains("paris")]),

    # --- expanded social intents (thanks / apology / emotion / identity /
    #     capability / affirmation / smalltalk)
    ("social", "thank you so much!",
     [contains_any("welcome", "any time", "happy to help")]),
    ("social", "thanks",
     [contains_any("welcome", "any time", "happy to help")]),
    ("social", "sorry",
     [starts_with_any("that is okay", "no need", "it is okay", "no worries",
                      "that's okay")]),
    ("social", "i am sorry for that",
     [contains_any("okay", "no need", "all right", "alright", "fine")]),
    ("social", "i am sad today",
     [contains_any("sorry", "listen", "tell me", "here")]),
    ("social", "i am so happy today!",
     [contains_any("happy", "yay", "wonderful", "glad")]),
    ("social", "who are you",
     [contains_any("buddy", "ai", "robot", "friendly")]),
    ("social", "are you a robot?",
     [contains_any("buddy", "ai", "robot", "friendly", "computer")]),
    ("social", "what can you do",
     [contains_any("chat", "questions", "stories", "remember", "help")]),
    ("social", "can you help me",
     [contains_any("course", "help", "best", "anything", "yes")]),
    ("social", "yes",
     [word_count_between(2, 40)]),
    ("social", "nope",
     [word_count_between(2, 40)]),
    ("social", "the weather is nice today",
     [contains_any("day", "lovely", "weather", "animals", "fun", "yes")]),

    # --- hygiene: every reply terminates and isn't degenerate
    ("hygiene", "hi",
     [nonempty(2), word_count_between(2, 60), ends_properly]),
    ("hygiene", "tell me a joke",
     [nonempty(3), word_count_between(3, 80)]),
    ("hygiene", "how do you make lemonade?",
     [nonempty(5), word_count_between(5, 120)]),
]


def load_model(ckpt_path, tok_path, device):
    tok = load_tokenizer(tok_path)
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = ModelConfig(**ck["config"])
    model = TinyGPT(cfg).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, tok, ck


@torch.no_grad()
def raw_reply(model, tok, device, user_input, temperature=0.5, max_new_tokens=80):
    """One raw-model chat completion (same format as the chat loop, no mind.py)."""
    from mind import _encode_context, END_TOKEN
    from model import generate
    end_id = tok.token_to_id(END_TOKEN)
    ids = _encode_context([], user_input, model.cfg.context_length, tok)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(model, idx, max_new_tokens=max_new_tokens, temperature=temperature,
                   top_k=50, eos_id=end_id, repetition_penalty=1.15)
    new = out[0].tolist()[len(ids):]
    if end_id in new:
        new = new[:new.index(end_id)]
    return tok.decode(new, skip_special_tokens=True).strip()


def run_suite(model, tok, device, use_mind=True, verbose=False):
    from mind import Mind, classify_intent, extract_facts
    mind = Mind(memory_path=None, show_intent=False) if use_mind else None

    results = []
    for tag, user_input, checks in SUITE:
        if use_mind:
            reply, intent = mind.handle(user_input, [], model, tok, device,
                                        temperature=0.5, max_tokens=80)
        else:
            reply = raw_reply(model, tok, device, user_input)
            intent = classify_intent(user_input)
        ctx = {"memory": dict(mind.mem.facts) if use_mind else {},
               "intent": intent}
        passed = all(check(reply, ctx) for check in checks)
        results.append((tag, user_input, reply, passed, intent))
        if verbose:
            mark = "PASS" if passed else "FAIL"
            print(f"[{mark}] ({tag}/{intent}) {user_input!r}\n      -> {reply!r}")

    return results


def main():
    ap = argparse.ArgumentParser(description="TinyGPT behavior eval suite")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--no-mind", action="store_true", help="test the raw model")
    ap.add_argument("-k", dest="keyword", default=None, help="only cases whose tag contains this")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    model, tok, ck = load_model(args.checkpoint, args.tokenizer, device)
    print(f"eval: {args.checkpoint} (epoch {ck.get('epoch', '?')}, "
          f"step {ck.get('global_step', '?')}) | mind {'ON' if not args.no_mind else 'OFF'}\n")

    suite = [c for c in SUITE if args.keyword is None or args.keyword in c[0]]
    results = run_suite(model, tok, device, use_mind=not args.no_mind,
                        verbose=args.verbose)

    by_tag = {}
    for tag, user_input, reply, passed, intent in results:
        if args.keyword is None or args.keyword in tag:
            by_tag.setdefault(tag, []).append(passed)

    print("=" * 62)
    for tag, marks in by_tag.items():
        bar = "".join("✓" if m else "✗" for m in marks)
        print(f"{tag:10} {bar}  {sum(marks)}/{len(marks)}")
    total = sum(len(m) for m in by_tag.values())
    good = sum(sum(m) for m in by_tag.values())
    print("=" * 62)
    print(f"TOTAL: {good}/{total} ({good / total:.0%})")
    sys.exit(0 if good == total else 1)


if __name__ == "__main__":
    main()
