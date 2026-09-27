#!/usr/bin/env bash
# v3 chain: wait for the v2 chat run to finish, then:
#   stage 1 — continue-pretrain the grown model (4->5 layers, ctx 256->512) on TinyStories
#   stage 2 — chat fine-tune the grown model on data/chat_v3 (dolly + alpaca, 9.7M tokens)
# Logs: stage1_continue.log, stage2_chat.log
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
# Use all logical CPUs for the CPU-only PyTorch build.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "[chain] waiting for v2 training to finish..."
while pgrep -f "train.py --data_dir data/chat --resume" > /dev/null; do
  sleep 60
done
echo "[chain] v2 done at $(date). Starting stage 1 (continue-pretrain, 5 layers, ctx 512)..."

"$PY" train.py --data_dir data/tinystories \
    --resume checkpoints/latest.pt --vocab_size 4096 \
    --n_layers 5 --context_length 512 \
    --max_stories 25000 --epochs 1 --lr 2e-4 --batch_size 8 \
    --grad_accum_steps 2 --warmup_steps 100 --eval_every 1000 \
    --checkpoint_dir checkpoints/v3_grown

if [ $? -ne 0 ]; then echo "[chain] stage 1 FAILED"; exit 1; fi
echo "[chain] stage 1 done at $(date). Starting stage 2 (chat fine-tune on dolly+alpaca)..."

"$PY" train.py --data_dir data/chat_v3 \
    --tokenizer data/chat/tokenizer_chat.json \
    --resume checkpoints/v3_grown/final.pt --vocab_size 4099 \
    --n_layers 5 --context_length 512 \
    --epochs 1 --lr 1e-4 --batch_size 8 --grad_accum_steps 2 \
    --warmup_steps 100 --eval_every 200 \
    --checkpoint_dir checkpoints/chat_run_v3

if [ $? -ne 0 ]; then echo "[chain] stage 2 FAILED"; exit 1; fi
echo "[chain] ALL DONE at $(date). Chat with: $PY sample.py --checkpoint checkpoints/chat_run_v3/final.pt --chat"
