"""Generate text from a trained TinyGPT checkpoint, or chat with it (--chat)."""

import argparse

import torch

from model import ModelConfig, TinyGPT, generate, resolve_device
from tokenizer import EOS_TOKEN, load_tokenizer

USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN = "<|user|>", "<|assistant|>", "<|end|>"
DEFAULT_TOKENIZER = "data/tinystories/tokenizer.json"
DEFAULT_CHAT_TOKENIZER = "data/chat/tokenizer_chat.json"


def build_prompt_text(history, user_input):
    """Render the transcript: <|user|>u<|assistant|>a<|end|> ... <|user|>u<|assistant|>."""
    parts = [f"{USER_TOKEN}{u}{ASSISTANT_TOKEN}{a}{END_TOKEN}" for u, a in history]
    parts.append(f"{USER_TOKEN}{user_input}{ASSISTANT_TOKEN}")
    return "".join(parts)


def encode_chat_context(tokenizer, history, user_input, context_length):
    """Encode the transcript, dropping the oldest turns until it fits the context."""
    ids = tokenizer.encode(build_prompt_text(history, user_input)).ids
    while len(ids) > context_length and history:
        history.pop(0)
        ids = tokenizer.encode(build_prompt_text(history, user_input)).ids
    return ids[-context_length:]  # single oversized turn: keep the most recent tokens


def chat_loop(tokenizer, model, device, args, mind=None):
    """Interactive loop; keeps the full history up to the model's context length."""
    for token in (USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN):
        if tokenizer.token_to_id(token) is None:
            raise SystemExit(f"Tokenizer is missing {token!r}; pass --tokenizer "
                             f"{DEFAULT_CHAT_TOKENIZER} (see add_chat_tokens.py)")
    end_id = tokenizer.token_to_id(END_TOKEN)
    history = []
    print(f"Chat mode — type 'quit' to exit | context {model.cfg.context_length} "
          f"tokens | max reply {args.max_tokens} tokens")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            return

        if mind is not None:
            reply, intent = mind.handle(user_input, history, model, tokenizer,
                                        device, temperature=args.temperature,
                                        top_k=args.top_k, max_tokens=args.max_tokens)
            history.append((user_input, reply))
            print(f"Assistant: {reply}")
            continue

        ids = encode_chat_context(tokenizer, history, user_input,
                                  model.cfg.context_length)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = generate(model, idx, max_new_tokens=args.max_tokens,
                       temperature=args.temperature, top_k=args.top_k, eos_id=end_id,
                       repetition_penalty=args.repetition_penalty)
        new_ids = out[0].tolist()[len(ids):]
        if end_id in new_ids:  # cut the reply at <|end|>
            new_ids = new_ids[:new_ids.index(end_id)]
        reply = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        history.append((user_input, reply))
        print(f"Assistant: {reply}")


def main():
    parser = argparse.ArgumentParser(description="Sample from (or chat with) a TinyGPT checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/s4/final.pt",
                        help="Default: the stage-4 model (epoch-17 + intent/memory fine-tune; "
                             "raw-NN memory recall 24/24, suite 34/35)")
    parser.add_argument("--tokenizer", default=None,
                        help=f"Path to tokenizer.json (default: {DEFAULT_CHAT_TOKENIZER} "
                             f"for --chat, else {DEFAULT_TOKENIZER})")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--chat", action="store_true",
                        help="Interactive chat loop using the "
                             "<|user|>/<|assistant|>/<|end|> format")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Max new tokens per generation (default: 100, or 200 in --chat)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="0 = greedy, sampling default: 0.5 in --chat, 0.8 for stories")
    parser.add_argument("--top_k", type=int, default=50, help="0 disables top-k filtering")
    parser.add_argument("--repetition_penalty", type=float, default=1.15,
                        help="1 = off; >1 down-weights tokens already generated this reply")
    parser.add_argument("--no-mind", action="store_true",
                        help="bypass the cognition layer (raw model, no memory/intents)")
    parser.add_argument("--show-intent", action="store_true",
                        help="print the classified intent of each user message")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    tokenizer_path = args.tokenizer or (DEFAULT_CHAT_TOKENIZER if args.chat
                                        else DEFAULT_TOKENIZER)
    args.max_tokens = args.max_tokens or (200 if args.chat else 100)
    if args.temperature is None:
        # The long-run model's sampling distribution is flatter than s3d's (it saw
        # 6x more diverse data), so raw chat runs cooler: measured 12/21 at 0.3
        # vs 10/21 at 0.5 (compare_models.py). Stories still want heat.
        args.temperature = 0.3 if args.chat else 0.8

    tokenizer = load_tokenizer(tokenizer_path)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = ModelConfig(**checkpoint["config"])
    if cfg.vocab_size != tokenizer.get_vocab_size():
        raise SystemExit(
            f"Checkpoint vocab_size ({cfg.vocab_size}) != tokenizer vocab size "
            f"({tokenizer.get_vocab_size()}); pass the matching --tokenizer."
        )
    model = TinyGPT(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded {args.checkpoint} ({model.num_parameters():,} params, "
          f"epoch {checkpoint.get('epoch', '?')}, "
          f"step {checkpoint.get('global_step', '?')}) on {device.type}")

    if args.chat:
        mind = None
        if not args.no_mind:
            try:
                from mind import Mind
                mind = Mind(memory_path="memory.json", show_intent=args.show_intent)
            except ImportError:
                print("(mind.py unavailable — running raw model)")
        chat_loop(tokenizer, model, device, args, mind=mind)
        return

    ids = tokenizer.encode(args.prompt).ids
    eos = tokenizer.token_to_id(EOS_TOKEN)
    if not ids:  # empty prompt: seed with EOS
        ids = [eos if eos is not None else 0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = generate(model, idx, max_new_tokens=args.max_tokens,
                   temperature=args.temperature, top_k=args.top_k, eos_id=eos,
                   repetition_penalty=args.repetition_penalty)
    text = tokenizer.decode(out[0].tolist()).replace(EOS_TOKEN, "").strip()
    print("-" * 60)
    print(text)
    print("-" * 60)


if __name__ == "__main__":
    main()
