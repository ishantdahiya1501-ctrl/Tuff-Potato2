"""Download Q&A chat data (databricks-dolly-15k and/or Alpaca) to raw_chat.json.

Defaults preserve the old behavior: dolly-15k only, same files as before.
New: pass --add_alpaca to merge in Alpaca (52k general instruction→output pairs),
which broadens coverage beyond dolly's narrow Q&A style.
"""
import argparse
import json
import os

from datasets import load_dataset

OUT_DIR = "data/chat"


def build_dolly():
    print("Downloading databricks-dolly-15k...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    pairs = []
    for ex in ds:
        user = ex["instruction"].strip()
        if ex.get("context"):
            user = f"{user}\n\nContext: {ex['context'].strip()}"
        assistant = ex["response"].strip()
        if user and assistant:
            pairs.append({"user": user, "assistant": assistant})
    print(f"  dolly: {len(pairs)} pairs")
    return pairs


def build_alpaca():
    print("Downloading Alpaca (tatsu-lab/alpaca)...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")  # ~52k
    pairs = []
    for ex in ds:
        inst = " ".join(ex["instruction"].split())
        output = " ".join(ex["output"].split())
        inp = " ".join(ex["input"].split())
        user = f"{inst}\n{inp}" if inp else inst
        if user and output:
            pairs.append({"user": user, "assistant": output})
    print(f"  alpaca: {len(pairs)} pairs")
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--add_alpaca", action="store_true",
                        help="Also download Alpaca and merge it with dolly")
    parser.add_argument("--out", default=os.path.join(OUT_DIR, "raw_chat.json"))
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    pairs = build_dolly()
    if args.add_alpaca:
        pairs += build_alpaca()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(pairs, f)
    print(f"Saved {len(pairs)} pairs to {args.out}")


if __name__ == "__main__":
    main()
