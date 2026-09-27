"""Train the tiny GPT on packed TinyStories tokens (next-token prediction)."""

import argparse
import itertools
import math
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from dataset import prepare_tokens
from model import ModelConfig, TinyGPT, resolve_device, save_checkpoint
from model import TransformerBlock
from tokenizer import load_tokenizer


def lr_at(step: int, warmup_steps: int, total_steps: int) -> float:
    """Cosine schedule with linear warmup, as a multiplicative LR factor."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def batch_starts(n_sequences, batch_size, rng):
    """Yield shuffled arrays of sequence start indices, one array per batch."""
    order = rng.permutation(n_sequences)
    for i in range(0, n_sequences - batch_size + 1, batch_size):
        yield order[i:i + batch_size]


def gather_batch(tokens, starts, context_length):
    idx = starts[:, None] * context_length + np.arange(context_length)[None, :]
    return np.asarray(tokens[idx]), np.asarray(tokens[idx + 1])


def grow_layers(state, cfg, old_layers):
    """Return a copy of `state` with extra transformer blocks appended.

    Existing blocks keep their trained weights; each new block gets TinyGPT's
    init (std=0.02, residual-output projections scaled down by depth), so a
    deeper model can continue from a shallower checkpoint.
    """
    new_state = dict(state)
    residual_std = 0.02 / math.sqrt(2 * cfg.n_layers)
    for layer in range(old_layers, cfg.n_layers):
        ref = TransformerBlock(cfg)  # fresh block with default init
        torch.nn.init.normal_(ref.attn.proj.weight, std=residual_std)
        torch.nn.init.normal_(ref.ffn.w2.weight, std=residual_std)
        for name, tensor in ref.state_dict().items():
            new_state[f"blocks.{layer}.{name}"] = tensor.clone()
    return new_state


def truncate_conversations(tokens, max_tokens, end_id):
    """Cut every conversation (delimited by end_id) to max_tokens tokens.

    Keeps the opening max_tokens-1 tokens plus the trailing end_id, so every
    conversation still terminates properly. Returns the new packed stream.
    """
    positions = np.flatnonzero(tokens == end_id)
    kept, n_truncated, n_dropped = [], 0, 0
    start = 0
    for pos in positions:
        conv = tokens[start:pos + 1]
        start = pos + 1
        if conv.size > max_tokens:
            conv = np.concatenate([conv[:max_tokens - 1], conv[-1:]])
            n_truncated += 1
            n_dropped += pos + 1 - start_original(conv.size, max_tokens)
        kept.append(conv)
    if start < tokens.size:  # trailing tokens after the last end_id
        kept.append(tokens[start:])
    return np.concatenate(kept) if kept else tokens[:0], n_truncated, n_dropped


def start_original(new_size, max_tokens):
    """Tokens dropped from one conversation truncated to `max_tokens`."""
    return max(0, new_size - max_tokens)


def resize_token_embedding(state, new_vocab_size):
    """Return a copy of `state` with tok_emb/lm_head resized to new_vocab_size.

    Overlapping rows are copied from the old embedding; the new rows are
    randomly initialized (std=0.02, matching TinyGPT's init).
    """
    old_weight = state["tok_emb.weight"]
    old_vocab, d_model = old_weight.shape
    new_weight = old_weight.new_empty((new_vocab_size, d_model))
    torch.nn.init.normal_(new_weight, mean=0.0, std=0.02)
    overlap = min(old_vocab, new_vocab_size)
    new_weight[:overlap] = old_weight[:overlap]
    state = dict(state)
    state["tok_emb.weight"] = new_weight
    state["lm_head.weight"] = new_weight  # tied with tok_emb in TinyGPT
    return state, old_vocab, new_vocab_size


@torch.no_grad()
def evaluate(model, tokens, context_length, batch_size, device, max_batches, autocast_kwargs):
    model.eval()
    n_sequences = (len(tokens) - 1) // context_length
    if n_sequences == 0:
        model.train()
        return float("nan")
    batch_size = min(batch_size, n_sequences)
    losses = []
    for start in range(0, n_sequences - batch_size + 1, batch_size):
        if len(losses) >= max_batches:
            break
        x, y = gather_batch(tokens, np.arange(start, start + batch_size), context_length)
        x = torch.from_numpy(x.astype(np.int64)).to(device)
        y = torch.from_numpy(y.astype(np.int64)).to(device)
        with torch.autocast(**autocast_kwargs):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ~5M param GPT on TinyStories")
    parser.add_argument("--data_dir", default="data/tinystories")
    parser.add_argument("--tokenizer", default=None,
                        help="Path to tokenizer.json (default: <data_dir>/tokenizer.json)")
    parser.add_argument("--resume", default=None,
                        help="Checkpoint to resume / fine-tune from")
    parser.add_argument("--vocab_size", type=int, default=None,
                        help="Override vocab size (e.g. 4099 for the chat tokenizer); "
                             "default: the tokenizer's vocab size")
    parser.add_argument("--n_layers", type=int, default=None,
                        help="Override layer count (e.g. 5 to grow a 4-layer "
                             "checkpoint; new layers are freshly initialized)")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Truncate chat conversations to this many tokens "
                             "(keeps <|end|>); default: no truncation")
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Micro-batch size (use --grad_accum_steps if it doesn't fit)")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Micro-batches per optimizer step (effective batch = batch_size * this)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max_stories", type=int, default=None,
                        help="Cap the number of training stories (default: all)")
    parser.add_argument("--eval_every", type=int, default=500,
                        help="Optimizer steps between validation evals (0 = off)")
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch_threads", type=int, default=0,
                        help="CPU threads for PyTorch (0 = all available CPUs)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Re-tokenize even if cached")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.torch_threads or (os.cpu_count() or 1))
        torch.set_num_interop_threads(1)

    tokenizer_path = args.tokenizer or os.path.join(args.data_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        # Chat fine-tuning data uses the extended chat tokenizer instead.
        alt_path = os.path.join(args.data_dir, "tokenizer_chat.json")
        if os.path.exists(alt_path):
            tokenizer_path = alt_path
    if not os.path.exists(tokenizer_path):
        raise SystemExit(
            f"Tokenizer not found at {tokenizer_path}.\n"
            f"Run first: python tokenizer.py --data_dir {args.data_dir} --vocab_size 4096"
        )
    vocab_size = args.vocab_size or load_tokenizer(tokenizer_path).get_vocab_size()

    train_tokens = prepare_tokens(args.data_dir, tokenizer_path, "train",
                                  args.max_stories, args.force)
    val_tokens = (
        prepare_tokens(args.data_dir, tokenizer_path, "validation", None, args.force)
        if args.eval_every > 0 else None
    )

    cfg = ModelConfig(vocab_size=vocab_size, context_length=args.context_length,
                      dropout=args.dropout,
                      n_layers=args.n_layers if args.n_layers is not None
                      else ModelConfig().n_layers)

    # Optional: truncate overlong conversations instead of dropping them.
    if args.max_tokens and args.max_tokens > 1:
        tok_for_ids = load_tokenizer(tokenizer_path)
        end_id = tok_for_ids.token_to_id("<|end|>")
        if end_id is not None:
            n_before = len(train_tokens)
            train_tokens, n_trunc, n_drop = truncate_conversations(
                train_tokens, args.max_tokens, end_id)
            print(f"Truncated {n_trunc:,} conversations to {args.max_tokens} tokens "
                  f"({n_drop:,} tokens dropped, {n_before - len(train_tokens):,} net)")
    model = TinyGPT(cfg).to(device)

    # Resume / fine-tune from a checkpoint. If the checkpoint's embedding does
    # not match the current vocab size, resize it: copy the overlapping rows,
    # randomly initialize the new ones, and start with a fresh optimizer.
    start_epoch, checkpoint, resized = 0, None, False
    if args.resume:
        if not os.path.exists(args.resume):
            raise SystemExit(f"Checkpoint not found: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        state = checkpoint["model_state"]
        old_layers = sum(1 for k in state
                         if k.startswith("blocks.") and k.endswith(".attn_norm.weight"))
        if old_layers < cfg.n_layers:
            state = grow_layers(state, cfg, old_layers)
            print(f"Grew model: {old_layers} -> {cfg.n_layers} layers "
                  f"(new layers freshly initialized)")
        old_vocab = state["tok_emb.weight"].shape[0]
        if old_vocab != cfg.vocab_size:
            state, old_vocab, new_vocab = resize_token_embedding(state, cfg.vocab_size)
            resized = True
            print(f"Resized embedding: {old_vocab} -> {new_vocab}")
        model.load_state_dict(state)
        print(f"Loaded weights from {args.resume} (checkpoint epoch "
              f"{checkpoint.get('epoch', '?')}, step {checkpoint.get('global_step', '?')})")
    print(f"Total parameters: {model.num_parameters():,}")

    n_sequences = (len(train_tokens) - 1) // args.context_length
    micro_per_epoch = n_sequences // args.batch_size
    if micro_per_epoch == 0:
        raise SystemExit(f"Only {n_sequences} sequences available: too few for one "
                         f"batch of {args.batch_size}. Lower --batch_size.")
    accum = max(1, args.grad_accum_steps)
    opt_per_epoch = max(1, micro_per_epoch // accum)
    total_opt_steps = opt_per_epoch * args.epochs
    print(f"{n_sequences:,} sequences of {args.context_length} tokens | "
          f"effective batch = {args.batch_size} x {accum} = {args.batch_size * accum} | "
          f"{total_opt_steps:,} optimizer steps over {args.epochs} epochs")

    # AdamW: weight decay on 2-D matrices only (not embeddings/norms).
    decay = [p for name, p in model.named_parameters()
             if p.ndim >= 2 and "tok_emb" not in name]
    no_decay = [p for name, p in model.named_parameters()
                if not (p.ndim >= 2 and "tok_emb" not in name)]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_at(step, args.warmup_steps, total_opt_steps)
    )

    # Restore optimizer/schedule/progress when resuming with a matching vocab.
    # (After an embedding resize we deliberately start fresh: the checkpoint's
    # AdamW state and step count no longer match the new model.)
    start_epoch, resumed_step = 0, 0
    if checkpoint is not None:
        if resized:
            print(f"Fine-tune mode: fresh optimizer & schedule, "
                  f"training {args.epochs} new epoch(s).")
        else:
            ckpt_step = checkpoint.get("global_step", 0)
            # A restored schedule is only meaningful if this run actually has
            # more steps than the checkpoint already took. Fine-tuning from a
            # 27000-step checkpoint with a shorter schedule (e.g. 573 x N chat
            # steps) lands past the cosine end -> lr == 0.0 -> frozen loss.
            if ckpt_step >= total_opt_steps:
                print(f"Fine-tune mode: checkpoint is at step {ckpt_step} but this "
                      f"run only has {total_opt_steps:,} steps; fresh optimizer & "
                      f"schedule (training {args.epochs} new epoch(s)).")
            else:
                try:
                    if checkpoint.get("optimizer_state") is not None:
                        optimizer.load_state_dict(checkpoint["optimizer_state"])
                    resumed_step = ckpt_step
                    scheduler = torch.optim.lr_scheduler.LambdaLR(
                        optimizer, lambda step: lr_at(step, args.warmup_steps, total_opt_steps),
                        last_epoch=resumed_step,
                    )
                    start_epoch = checkpoint.get("epoch", 0)
                    print(f"Resuming optimizer/schedule at epoch {start_epoch + 1} "
                          f"(step {resumed_step}); training to epoch {args.epochs}.")
                except (ValueError, KeyError) as exc:
                    print(f"Could not restore optimizer state ({exc}); starting fresh.")
                    resumed_step = 0

    # Mixed precision on CUDA (bf16 if supported, else fp16); plain fp32 on CPU.
    if device.type == "cuda":
        use_amp = True
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        use_amp = False
        amp_dtype = torch.bfloat16  # unused: autocast stays disabled on CPU
    autocast_kwargs = {"device_type": device.type, "dtype": amp_dtype, "enabled": use_amp}
    use_scaler = use_amp and amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)
    except (AttributeError, TypeError):  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    latest_path = os.path.join(args.checkpoint_dir, "latest.pt")

    def run_eval(step, epoch):
        if val_tokens is None:
            return None
        val_loss = evaluate(model, val_tokens, args.context_length, args.batch_size,
                            device, args.eval_batches, autocast_kwargs)
        print(f"[eval] step {step} (epoch {epoch}): val_loss = {val_loss:.4f}")
        return val_loss

    def save(path, epoch, step, val_loss=None):
        save_checkpoint(path, model, optimizer, epoch=epoch,
                        global_step=step, val_loss=val_loss)
        print(f"[ckpt] saved {path}")

    def optimizer_step():
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    print(f"Training on {device.type} | mixed precision: "
          f"{str(amp_dtype).split('.')[-1] if use_amp else 'off'}")
    start_time = time.time()
    global_step, epoch = resumed_step, start_epoch
    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            batches = batch_starts(n_sequences, args.batch_size, rng)
            usable_micro = opt_per_epoch * accum
            progress = tqdm(itertools.islice(batches, usable_micro), total=usable_micro,
                            desc=f"epoch {epoch}/{args.epochs}", unit="batch",
                            dynamic_ncols=True)
            recent, seen = [], 0
            optimizer.zero_grad(set_to_none=True)
            for starts in progress:
                x, y = gather_batch(train_tokens, starts, args.context_length)
                x = torch.from_numpy(x.astype(np.int64)).to(device)
                y = torch.from_numpy(y.astype(np.int64)).to(device)
                with torch.autocast(**autocast_kwargs):
                    _, loss = model(x, y)
                scaler.scale(loss / accum).backward()
                recent.append(loss.item())
                seen += 1
                if seen % accum == 0:
                    optimizer_step()
                    global_step += 1
                    if global_step % args.log_every == 0:
                        progress.set_postfix(
                            loss=f"{np.mean(recent[-args.log_every * accum:]):.4f}",
                            lr=f"{scheduler.get_last_lr()[0]:.2e}")
                    if args.eval_every > 0 and global_step % args.eval_every == 0:
                        val_loss = run_eval(global_step, epoch)
                        save(latest_path, epoch, global_step, val_loss)
            # Flush trailing accumulated grads (when the epoch isn't divisible).
            if seen % accum != 0:
                optimizer_step()
                global_step += 1
            print(f"epoch {epoch}: train_loss (last 100 avg) = {np.mean(recent[-100:]):.4f}")
            val_loss = None
            if val_tokens is not None and (args.eval_every == 0
                                           or global_step % args.eval_every != 0):
                val_loss = run_eval(global_step, epoch)
            save(os.path.join(args.checkpoint_dir, f"epoch{epoch}.pt"),
                 epoch, global_step, val_loss)
            save(latest_path, epoch, global_step, val_loss)

        save(os.path.join(args.checkpoint_dir, "final.pt"), args.epochs, global_step)
        print(f"Done in {(time.time() - start_time) / 60:.1f} min "
              f"({global_step:,} optimizer steps).")
        print("Generate with: python sample.py --checkpoint "
              f"{os.path.join(args.checkpoint_dir, 'final.pt')} --prompt \"Once upon a time\"")
    except KeyboardInterrupt:
        print("\nInterrupted - saving checkpoint ...")
        save(latest_path, max(epoch, 1), global_step)


if __name__ == "__main__":
    main()
