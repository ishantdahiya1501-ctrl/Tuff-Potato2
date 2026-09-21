# FRIES — Tiny Multimodal AI

**FRIES** is a lightweight multimodal AI system built from multiple specialized small models, designed to eventually run on embedded hardware and power an autonomous desktop AI companion.

Its core language model is a **~5.25M-parameter decoder-only Transformer**, while the larger FRIES system is designed around specialized models such as **Tuff Potato**, **Tuff Potato Maths**, and future task-specific models.

The goal is simple:

> **Build useful AI capabilities at an extremely small model scale, then give the AI a physical body.**

---

## What Makes FRIES Different?

* **~5.6M parameters across the current FRIES model system**
* Specialized models instead of one giant model
* Designed with embedded hardware in mind
* Target platforms include **ESP32-S3, ESP32-P4, ESP32-C5, and ESP32-C6**
* Focused on simple, human-like interaction
* Deterministic tools for tasks where an LLM should not guess
* Designed to eventually combine **language, vision, voice, memory, emotions, robotics, and home control**

---

# Current AI Model

The core language model is a minimal GPT-style Transformer implemented from scratch in PyTorch.

It does **not** use HuggingFace `transformers`. The Transformer architecture, including attention, RoPE, RMSNorm, and SwiGLU, is implemented directly in the project.

### Architecture

| Component         | Choice                 |
| ----------------- | ---------------------- |
| Type              | Decoder-only GPT       |
| Layers            | 4                      |
| `d_model`         | 256                    |
| Attention heads   | 8                      |
| Head dimension    | 32                     |
| Context length    | 256 tokens             |
| FFN               | SwiGLU                 |
| FFN hidden size   | 1024                   |
| Normalization     | RMSNorm                |
| Position encoding | RoPE                   |
| RoPE theta        | 10000                  |
| Embeddings        | Input/output tied      |
| Bias              | None                   |
| Dropout           | 0.0 configurable       |
| Vocabulary        | 4096 byte-level BPE    |
| Parameters        | **5,245,184 (~5.25M)** |

---

# Training

The base model is trained on:

* [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

The project also includes an optional chat fine-tuning pipeline using:

* [TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct)

The chat format uses three special tokens:

```text
<|user|>
<|assistant|>
<|end|>
```

This allows the same small Transformer to be adapted from story generation into a basic conversational model.

---

# Project Structure

```text
model.py
```

Model definition, Transformer blocks, RoPE, generation, and checkpoint handling.

```text
tokenizer.py
```

Trains and loads the 4096-vocabulary byte-level BPE tokenizer.

```text
dataset.py
```

Downloads, tokenizes, and caches TinyStories as packed token streams.

```text
train.py
```

Training loop using AdamW, cosine learning-rate scheduling, warmup, AMP, evaluation, and checkpoints.

```text
sample.py
```

Text generation and interactive chat.

```text
add_chat_tokens.py
```

Adds the three chat special tokens.

```text
download_chat_data.py
```

Downloads TinyStoriesInstruct data.

```text
format_chat.py
```

Formats and filters conversations.

```text
tokenize_chat.py
```

Creates packed chat token datasets.

```text
run_all.py
```

Runs the complete chat-data preparation pipeline.

```text
requirements.txt
```

Project dependencies.

---

# Setup

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training the Base Model

Train the tokenizer:

```bash
python tokenizer.py --data_dir data/tinystories --vocab_size 4096
```

Train the model:

```bash
python train.py --data_dir data/tinystories \
    --context_length 256 \
    --batch_size 64 \
    --epochs 3 \
    --lr 3e-4 \
    --device cuda
```

For a faster CPU experiment:

```bash
python train.py --max_stories 50000 --device auto
```

---

# Chat Fine-Tuning

Prepare the chat dataset:

```bash
python run_all.py
```

This creates:

```text
data/chat/
├── tokenizer_chat.json
├── raw_chat.json
├── formatted_chat.txt
├── chat_train_tokens.npy
└── chat_val_tokens.npy
```

Fine-tune the pretrained model:

```bash
python train.py --data_dir data/chat \
    --resume checkpoints/latest.pt \
    --vocab_size 4099 \
    --epochs 2 \
    --lr 1e-4 \
    --batch_size 32
```

Run interactive chat:

```bash
python sample.py --checkpoint checkpoints/final.pt --chat
```

---

# Text Generation

```bash
python sample.py \
    --prompt "Once upon a time" \
    --max_tokens 100
```

Useful options include:

```text
--checkpoint
--temperature
--top_k
--seed
--device
```

---

# FRIES Development Roadmap

The current Transformer is only the **core intelligence layer**.

The long-term goal is to combine it with specialized models and physical systems.

---

## Phase 1 — Core Multimodal AI

FRIES becomes a collection of small specialized models.

### Planned models

* **Tuff Potato** — general conversation
* **Tuff Potato Maths** — deterministic mathematical reasoning
* Additional specialized models for future tasks

Instead of forcing one model to perform everything, FRIES can route tasks to the appropriate model or tool.

### Design goals

* Extremely small model footprint
* Independent specialized models
* Embedded-device compatibility
* Simple conversational interaction
* Deterministic tools where reliability matters

---

# Phase 2 — Give FRIES a Body

The long-term goal is to turn FRIES into a physical AI companion.

## 1. Presence & Body

Possible form factors:

* Desk companion
* Two-legged waddling robot
* Small desktop robot

Features:

* Screen-based eyes
* Blinking
* Looking around
* Mood animations
* Head pan/tilt
* Face following
* PIR/mmWave presence detection
* Touch interaction
* Desk-edge detection

### Idle behaviors

FRIES could:

* Look around
* Stretch
* Blink slowly
* Doze
* React to its surroundings

---

# 2. Voice

FRIES will eventually support:

* Wake-word detection such as `"hey bot"`
* Short speech commands
* Speech recognition
* Text-to-speech
* Simple, clear robotic voice
* Supported-language restrictions

The system should avoid pretending to understand capabilities it does not actually support.

---

# 3. Conversation & Memory

Basic conversational capabilities:

* Greetings
* Name
* Feelings
* Likes
* Weather
* Day and time
* Goodbyes

Memory could store:

* User's name
* A small number of user facts
* Previous conversation topics
* Previous mood states

A core design principle:

> **If FRIES does not know something, it should say so instead of inventing an answer.**

Advanced topics outside the supported models can be routed away from the small conversational model rather than forcing it to answer.

---

# 4. Emotion & Personality

FRIES can maintain simple internal mood states:

* Happy
* Sleepy
* Curious
* Bored
* Excited

Mood can affect:

* Facial animations
* Voice tone
* Response style
* Idle behavior

Example:

```text
User: you're doing great!

FRIES:
happy expression
+ excited voice
+ animated response
```

The physical behavior can become more expressive when the user interacts with the robot.

---

# 5. Tuff Potato Maths

Math should be handled by a **symbolic/deterministic engine rather than the language model**.

Supported operations can include:

* Addition
* Subtraction
* Multiplication
* Division
* Percentages
* Basic unit conversion
* Timer calculations

Example:

```text
User:
25% of 200

Math Engine:
50
```

This prevents the language model from hallucinating mathematical results.

---

# 6. Timers & Reminders

Planned features:

* Timers
* Reminders
* Alarms
* Wake-up commands
* Pomodoro mode
* Focus/break states
* Persistent storage

Persistent data can eventually be stored using an SD card or other non-volatile storage.

---

# 7. Home Control

FRIES could interact with home devices using:

* IR
* ESP-NOW
* Relay nodes

Possible controls:

* AC
* TV
* Lights
* Fans
* Smart plugs
* Volume/input
* Temperature
* Light brightness

### Example routines

```text
"good night"
→ lights off
→ AC on
→ sleep mode
```

```text
"movie time"
→ TV on
→ lights dim
```

```text
"focus mode"
→ warm lights
→ focus state
```

### Presence automation

```text
User leaves
→ lights off
→ AC off
```

```text
User arrives
→ greeting
→ lights on
```

---

# 8. Robotic Arm — 2–3 DOF

A future FRIES body could include a small robotic arm.

Possible capabilities:

* Wave hello/goodbye
* Point at objects
* Point at people
* Nod/shake
* Reach for lightweight objects
* Pick up lightweight objects
* Hand objects to the user
* Push/tap objects
* Learn and replay taught poses
* Synchronize gestures with speech

Example:

```text
"I am happy"

→ arms move upward
→ happy eyes
→ excited voice
```

---

# 9. Vision — On-Device Computer Vision

FRIES is intended to use lightweight computer vision instead of requiring a huge vision model for every task.

Possible capabilities:

* Face detection
* Face tracking
* Head following
* Motion detection
* Room presence detection
* Color tracking
* Object/color tracking
* Line following
* Desk-edge detection
* QR-code scanning
* Barcode scanning

The objective is to run as much perception as possible directly on-device.

---

# System Architecture

The long-term FRIES architecture can be thought of as:

```text
                 ┌─────────────────────┐
                 │       FRIES         │
                 │   AI Orchestrator   │
                 └──────────┬──────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
   Tuff Potato       Tuff Potato Maths   Other Models
   Conversation       Symbolic Math      Specialized AI
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
                            ▼
                     Tools / Memory
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
           Voice          Vision         Robotics
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                       FRIES Body
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
       Display          Motors/Arm       Home Devices
```

---

# Development Phases

### Phase 1

**Tiny Multimodal AI**

* Small Transformer
* Specialized models
* Math engine
* Tool calling
* Basic conversation
* Embedded-device optimization

### Phase 2

**Physical AI Companion**

* Body
* Display eyes
* Voice
* Vision
* Memory
* Emotions
* Robotics

### Phase 3

**Autonomous Desktop Companion**

* Home automation
* Presence awareness
* Object interaction
* Learned behaviors
* Multimodal interaction
* Persistent personality and memory

---

# Final Goal

The final vision is:

> **FRIES = tiny AI models + tools + memory + voice + vision + emotions + robotics + home control**

A small AI system that does not need a massive cloud model to perform every task, but instead combines **specialized intelligence, deterministic tools, and an interactive physical body**.

---

## Design Philosophy

FRIES is deliberately minimal.

The project does not attempt to reproduce a frontier-scale model.

Instead, it explores how much useful intelligence can be created by combining:

```text
Small Models
     +
Specialized Tools
     +
Deterministic Systems
     +
On-Device AI
     +
Robotics
     =
FRIES
```

No RLHF, DPO, MoE, or unnecessary architecture complexity is required for the current core model.

The objective is to build a system that is **small enough to understand, train, modify, and eventually deploy on real hardware.**
