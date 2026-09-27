"""Phase 2A — the Cognition Layer: intent routing, session memory, reply steering.

Architecture decision (agreed with the owner, see README.md):
The neural network does the LANGUAGE; this module does the REMEMBERING and the
STEERING. Tiny models memorize templates and can't track entities across turns,
so a thin code layer classifies intent, stores facts the user shares, injects
them into the prompt on memory queries, and reranks sampled candidates.

Everything inside this file is the *cognition layer's* logic — the ONLY place
scripted fallback replies are allowed. The NN itself is never hardcoded.
Set --no-mind in sample.py to bypass this layer entirely and talk to the raw
model (that's what eval_suite.py uses for model-only comparisons).
"""

from __future__ import annotations

import json
import os
import random
import re

import torch

from model import generate

USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN = "<|user|>", "<|assistant|>", "<|end|>"

# Greeting/farewell detection only fires on SHORT messages: "good morning, can
# you help me with X" is a question that happens to start with a greeting.
GREETING_WORDS = ("hi", "hello", "hey", "yo", "hiya", "howdy", "greetings",
                  "namaste", "sup", "what's up", "whats up", "wassup")
FAREWELL_WORDS = ("bye", "goodbye", "goodnight", "good night", "see you",
                  "see ya", "cya", "take care", "catch you later")

# The fuller social surface (kept in sync with NNSOCIAL in make_overnight_data.py,
# which teaches the same pairs to the RAW network).
EXPANDED_SOCIAL = [
    ("thank you", ["You are very welcome!", "Any time! Happy to help.",
                   "You are so welcome!"]),
    ("thanks", ["Any time! Happy to help.", "You are welcome!"]),
    ("thanks a lot", ["You are very welcome! Glad I could help."]),
    ("i am happy", ["Yay! I am happy too when you are happy!"]),
    ("i love you", ["Aww, I like you a lot too! You are kind."]),
    ("you are funny", ["Hehe, thank you! I like making you laugh."]),
    ("you are smart", ["Thank you! I try my best."]),
    ("you are the best", ["Thank you so much! You are the best too!"]),
    ("tell me a joke", ["Why did the cookie go to the doctor? Because it was "
                        "feeling crummy!"]),
    ("who are you", ["I am a small friendly AI who likes to chat and tell stories."]),
    ("what are you", ["I am a little AI buddy who loves to chat with you!"]),
    ("are you a robot", ["I am a friendly AI buddy! I love to talk and tell stories."]),
    ("are you human", ["Nope, I am an AI! But I always enjoy our chats."]),
    ("how old are you", ["I am a young AI, so you could say I am just a few days old!"]),
    ("where do you live", ["I live inside this computer, right here with you!"]),
    ("what can you do", ["I can chat, answer questions, remember your favorite "
                         "things, and tell stories!"]),
    ("can you help me", ["Of course! Ask me anything and I will do my best."]),
    ("will you be my friend", ["Yes! I am so happy to be your friend."]),
    ("are you there", ["Yes, I am right here! What is on your mind?"]),
    ("the weather is nice today", ["Yes! It is a lovely day. Want to hear "
                                   "something fun about animals?"]),
]
# Prefix fallbacks when the exact message isn't in the table above.
SOCIAL_PREFIXES = (
    ("thanks", ["You are very welcome! Any time.", "Happy to help!"]),
    ("thank you", ["You are very welcome! Any time.", "Happy to help!"]),
    ("sorry", ["That is okay! Everyone makes mistakes."]),
    ("i'm sorry", ["No need to be sorry! It is all okay."]),
    ("i am sorry", ["No need to be sorry! It is all okay."]),
    ("good night", ["Good night! Sweet dreams!"]),
    ("tell me a joke", ["Why did the cookie go to the doctor? Because it was "
                        "feeling crummy!"]),
    ("what can you do", ["I can chat, answer questions, remember your favorite "
                         "things, and tell stories!"]),
    ("are you there", ["Yes, I am right here! What is on your mind?"]),
    ("how are you", ["I am doing great, thank you for asking! How are you?"]),
)
THANK_RE = re.compile(
    r"\b(thanks|thank you|thank u|thankyou|thx|ty|much appreciated|appreciate it)\b", re.I)
APOLOGY_RE = re.compile(
    r"\b(sorry|apologize|apologise|my bad|my fault|forgive me)\b", re.I)
POSITIVE_WORDS = ("happy", "great", "awesome", "amazing", "excited", "wonderful",
                  "fantastic", "feeling good", "so glad", "love this")
NEGATIVE_WORDS = ("sad", "angry", "upset", "tired", "scared", "worried",
                  "anxious", "lonely", "bored", "hurt", "feeling down", "cry",
                  "not good", "not well", "unwell", "sick")
# Messages that LOOK like emotion shares but are really requests -> not EMOTION_SHARE.
REQUEST_NEG_RE = re.compile(r"\b(help|can you|could you|please|you should|stop)\b", re.I)
QUESTION_END_RE = re.compile(r"\?+\s*$")
IDENTITY_PAT = re.compile(
    r"\b(who|what) (are|r) (you|u)\b|\bare (you|u) (a )?(robot|human|person|ai|real)\b"
    r"|\bdo you (have|has) (a )?(name|feelings|family|friends|house|home)\b"
    r"|\bhow old are (you|u)\b|\bwhere do (you|u) live\b", re.I)
# Only META-capability requests ("can you help me"), not capability-phrased
# content questions ("can you tell me a story" must stay a QUESTION).
CAPABILITY_PAT = re.compile(
    r"^\s*what can you do\b"
    r"|^\s*(can|could) you (help|talk|chat|remember|be my|learn|teach)"
    r"|^\s*will you (be my|help|talk|chat)", re.I)
YES_PAT = re.compile(r"^\s*(yes|yeah|yep|yup|sure|ok(?:ay)?|of course)\s*[.!]?\s*$", re.I)
NO_PAT = re.compile(r"^\s*(no|nope|nah|not really|never)\s*[.!]?\s*$", re.I)
SMALLTALK_RE = re.compile(
    r"\b(nice|lovely|beautiful) (weather|day|morning|evening)\b"
    r"|\bit('?s| is) (a )?(nice|lovely|beautiful) day\b"
    r"|\bthe (weather|day|morning|evening) (is|was) (nice|lovely|beautiful|great)\b", re.I)

# Words that follow "I am" but describe a STATE, not a name ("i am fine").
STATE_WORDS = {
    "fine", "good", "great", "okay", "ok", "happy", "sad", "angry", "tired",
    "hungry", "sick", "busy", "bored", "excited", "sleepy", "thirsty", "sorry",
    "here", "back", "done", "ready", "lost", "confused", "sure", "not", "very",
    "really", "so", "just", "new", "cool", "awesome", "serious", "wrong",
    "right", "a", "an", "the", "in", "on", "at", "from", "feeling", "doing",
}

NAME_RE = re.compile(
    r"\b(?:my name is|call me|this is|i am|i'm|im)\s+"
    r"([a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b", re.I)
AGE_RE = re.compile(r"\bi(?:'m| am)\s+(\d{1,2})\s*(?:years? old|yrs? old|yo)\b", re.I)
LIKE_RE = re.compile(r"\bi\s+(?:really\s+)?(?:like|love|enjoy|prefer)\s+(.{2,40}?)[.!?]*\s*$", re.I)
DISLIKE_RE = re.compile(r"\bi\s+(?:hate|dislike|don't like|dont like)\s+(.{2,40}?)[.!?]*\s*$", re.I)
LOCATION_RE = re.compile(r"\bi\s+(?:live in|am from|'m from)\s+([a-zA-Z ]{2,30}?)[.!?]*\s*$", re.I)
FAV_RE = re.compile(r"\bmy\s+(?:favorite|favourite|fav)\s+([a-z]+)\s+(?:is|are)\s+(.{1,30}?)[.!?]*\s*$", re.I)

MEMORY_QUERY_RES = [
    re.compile(r"\bwhat(?:'s| is| are)\s+my\b", re.I),
    re.compile(r"\bwho am i\b", re.I),
    re.compile(r"\bhow old am i\b", re.I),
    re.compile(r"\bwhat do i (?:like|love|enjoy)\b", re.I),
    re.compile(r"\bwhere do i (?:live|come from)\b", re.I),
    re.compile(r"\bdo you (?:know|remember)\b", re.I),
    re.compile(r"\bmy name\b", re.I),
]
QUESTION_STARTS = ("what", "why", "how", "when", "where", "who", "which", "can you",
                   "could you", "do you", "does", "did", "is", "are", "tell me",
                   "explain", "suggest", "give me", "name")

NAME_ACKS = [
    "Nice to meet you, {v}! I am Buddy.",
    "Hi {v}! Great to meet you.",
    "{v} — awesome name! Nice to meet you.",
]
FACT_ACKS = [
    "Got it — {k}: {v}. I'll remember!",
    "Noted! {k}: {v}.",
    "Cool, {k}: {v} — I'll remember that.",
]
MISSING_MEMORY_REPLIES = [
    "Hmm, I don't have that in my memory yet — tell me and I'll remember!",
    "I don't know that about you yet. Tell me and I'll store it!",
]


class MemoryStore:
    """Key/value facts ("name" -> "Ravi"), optionally persisted to memory.json."""

    def __init__(self, path: str | None = "memory.json"):
        self.path = path
        self.facts: dict[str, str] = {}
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.facts = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.facts = {}

    def update(self, new: dict[str, str]) -> None:
        self.facts.update(new)
        if self.path:
            try:
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(self.facts, f, indent=1)
            except OSError:
                pass  # in-memory only; never crash the chat loop over I/O

    def get(self, key: str) -> str | None:
        return self.facts.get(key)

    def clear(self) -> None:
        self.facts = {}
        if self.path and os.path.exists(self.path):
            os.remove(self.path)

    def summary(self) -> str:
        return "; ".join(f"{k} is {v}" for k, v in self.facts.items()) or "(empty)"


def _short(msg: str) -> bool:
    return len(msg.split()) <= 4


def _starts_with_any(msg: str, words) -> bool:
    low = msg.lower().lstrip()
    return any(low == w or low.startswith(w + " ") or low.startswith(w + "!") or
               low.startswith(w + ",") for w in words)


def _is_thanks(msg: str) -> bool:
    return bool(THANK_RE.search(msg))


def _is_apology(msg: str) -> bool:
    return bool(APOLOGY_RE.search(msg))


def _is_emotion_share(msg: str) -> bool:
    low = msg.lower()
    if REQUEST_NEG_RE.search(low) or QUESTION_END_RE.search(msg):
        return False
    return (re.search(r"\bi (?:am|'m|feel|felt|am feeling)\b", low) is not None
            and any(w in low for w in POSITIVE_WORDS + NEGATIVE_WORDS))


def _is_identity(msg: str) -> bool:
    return bool(IDENTITY_PAT.search(msg))


def _is_capability(msg: str) -> bool:
    return bool(CAPABILITY_PAT.search(msg))


def _is_affirmative(msg: str) -> bool:
    return bool(YES_PAT.match(msg) or NO_PAT.match(msg))


def _is_smalltalk(msg: str) -> bool:
    return bool(SMALLTALK_RE.search(msg))


def _social_reply_for(msg: str, intent: str) -> str | None:
    """Best scripted social reply: exact EXPANDED_SOCIAL hit, else prefix table,
    else an intent-level default."""
    low = msg.lower().strip(" !.?…")
    for q, answers in EXPANDED_SOCIAL:
        if low == q:
            return answers[hash(msg) % len(answers)]
    for pref, answers in SOCIAL_PREFIXES:
        if low.startswith(pref):
            return answers[hash(msg) % len(answers)]
    if intent == "EMOTION_SHARE":
        if any(w in low for w in NEGATIVE_WORDS):
            return ("I am sorry you feel that way. Want to tell me about it? "
                    "Sometimes talking helps, and I am here to listen.")
        return "That is wonderful to hear! Yay!"
    if intent == "IDENTITY":
        return ("I am Buddy, a small friendly AI who likes to chat and tell "
                "stories. I do not have feelings or a family, but I love our talks!")
    if intent == "CAPABILITY":
        return ("I can chat with you, answer questions, remember your favorite "
                "things, and tell stories! What would you like to do?")
    if intent == "AFFIRMATION":
        if low in ("no", "nope", "nah", "not really", "never"):
            return "Okay! Just tell me what you would like to do."
        return "Great! What would you like to talk about?"
    if intent == "SMALLTALK":
        return "Yes! It is a lovely day. Want to hear something fun about animals?"
    return None


def classify_intent(msg: str) -> str:
    """Classify one user message into a routing intent.

    Intents: GREETING | FAREWELL | SELF_DISCLOSURE | MEMORY_QUERY | THANKS |
    APOLOGY | EMOTION_SHARE | IDENTITY | CAPABILITY | AFFIRMATION | SMALLTALK |
    QUESTION | OTHER.

    Order matters: facts beat memory-queries ("my name is Ravi" stores, "what
    is my name?" asks); social intents (thanks/sorry/emotion/identity/...) are
    checked before generic QUESTION so "are you a robot?" routes to its own
    trained category; short-message greetings fire only on SHORT messages.
    """
    msg = msg.strip()
    # A message that carries a storable fact is a DISCLOSURE even if it mentions
    # "my name" ("my name is Ravi" vs the query "what is my name?") — facts
    # must win over queries, otherwise disclosures never get stored.
    if _has_fact(msg):
        return "SELF_DISCLOSURE"
    if any(r.search(msg) for r in MEMORY_QUERY_RES):
        return "MEMORY_QUERY"
    if _starts_with_any(msg, FAREWELL_WORDS) and _short(msg):
        return "FAREWELL"
    if _starts_with_any(msg, GREETING_WORDS) and _short(msg):
        return "GREETING"
    if _is_thanks(msg):
        return "THANKS"
    if _is_apology(msg):
        return "APOLOGY"
    if _is_emotion_share(msg):
        return "EMOTION_SHARE"
    if _is_identity(msg):
        return "IDENTITY"
    if _is_capability(msg):
        return "CAPABILITY"
    if _is_smalltalk(msg):
        return "SMALLTALK"
    if _is_affirmative(msg):
        return "AFFIRMATION"
    if msg.endswith("?") or _starts_with_any(msg, QUESTION_STARTS):
        return "QUESTION"
    return "OTHER"


def extract_facts(msg: str) -> dict[str, str]:
    """Pull storable facts out of a message: {"name": "Ravi"}, {"age": "17"}..."""
    facts: dict[str, str] = {}
    m = NAME_RE.search(msg)
    if m:
        name = m.group(1).strip()
        first = name.split()[0].lower()
        if first not in STATE_WORDS and not first.isdigit():
            facts["name"] = name[0].upper() + name[1:]
    m = AGE_RE.search(msg)
    if m:
        facts["age"] = m.group(1)
    m = LIKE_RE.search(msg)
    if m and not m.group(1).lower().startswith(("you", "u ")):
        facts["likes"] = m.group(1).strip()
    m = DISLIKE_RE.search(msg)
    if m:
        facts["dislikes"] = m.group(1).strip()
    m = LOCATION_RE.search(msg)
    if m:
        loc = m.group(1).strip()
        if loc.split()[0].lower() not in STATE_WORDS:
            facts["location"] = loc
    m = FAV_RE.search(msg)
    if m:
        facts[f"favorite {m.group(1).strip()}"] = m.group(2).strip()
    return facts


def _has_fact(msg: str) -> bool:
    return bool(extract_facts(msg))


def _memory_answer_key(msg: str) -> str | None:
    low = msg.lower()
    # 'nam' also catches typos like 'what is my nam?'
    if re.search(r"\bnam", low) or "who am i" in low:
        return "name"
    if "old" in low:
        return "age"
    if re.search(r"\b(like|love|enjoy)\b", low):
        return "likes"
    if re.search(r"\b(live|from)\b", low):
        return "location"
    m = re.search(r"fav\w*\s+(\w+)", low)
    if m:
        return f"favorite {m.group(1)}"
    for key in ("likes", "dislikes", "age", "location", "name"):
        if key in low:
            return key
    return None


def _pick(msg: str, options: list[str]) -> str:
    """Deterministic-ish template rotation: same input -> same reply."""
    return options[hash(msg) % len(options)]


class Mind:
    """Binds the cognition layer to one model + one memory store."""

    def __init__(self, memory_path: str | None = "memory.json", show_intent: bool = False):
        self.mem = MemoryStore(memory_path)
        self.show_intent = show_intent

    # ------------------------------------------------------------- generation
    def _model_replies(self, history, user_input, model, tokenizer, device, *,
                       temperature, top_k=50, candidates=1, max_new_tokens=60):
        end_id = tokenizer.token_to_id(END_TOKEN)
        outs = []
        for _ in range(candidates):
            ids = _encode_context(history, user_input, model.cfg.context_length, tokenizer)
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            out = generate(model, idx, max_new_tokens=max_new_tokens,
                           temperature=temperature, top_k=top_k, eos_id=end_id,
                           repetition_penalty=1.15)
            new = out[0].tolist()[len(ids):]
            if end_id in new:
                new = new[:new.index(end_id)]
            outs.append(tokenizer.decode(new, skip_special_tokens=True).strip())
        return outs

    # ---------------------------------------------------------------- replies
    def _ack_self_disclosure(self, msg: str, facts: dict[str, str]) -> str:
        if "name" in facts:
            return _pick(msg, NAME_ACKS).format(v=facts["name"])
        if facts:
            k, v = next(iter(facts.items()))
            return _pick(msg, FACT_ACKS).format(k=k, v=v)
        return "Got it — noted!"

    def _answer_memory_query(self, msg, model, tokenizer, device, **gen_kw) -> str:
        key = _memory_answer_key(msg)
        value = self.mem.get(key) if key else None
        if value is None and key is None and self.mem.facts:
            for k, v in self.mem.facts.items():
                if k in msg.lower():
                    value = v
                    key = k
                    break
        if value is not None:
            if key == "name":
                return _pick(msg, [f"Your name is {value}!", f"You are {value}!"])
            if key == "age":
                return f"You are {value} years old."
            if key == "likes":
                return f"You like {value}!"
            return f"You told me your {key}: {value}."
        if key is not None:
            return _pick(msg, MISSING_MEMORY_REPLIES)
        # Unknown query shape: let the model answer, with memory injected.
        if self.mem.facts:
            msg = f"{msg} (Context: {self.mem.summary()}.)"
        return self._model_replies(None, msg, model, tokenizer, device,
                                   temperature=0.3, candidates=1, **gen_kw)[0]

    def _social_reply(self, msg, intent, history, model, tokenizer, device,
                      max_tokens=200):
        """Route THANKS/APOLOGY/EMOTION/IDENTITY/CAPABILITY/AFFIRMATION/SMALLTALK:
        scripted surface first, NN best-of-3 as fallback."""
        scripted = _social_reply_for(msg, intent)
        if scripted is not None:
            return scripted
        cands = self._model_replies(history, msg, model, tokenizer, device,
                                    temperature=0.4, candidates=3,
                                    max_new_tokens=min(max_tokens, 60))
        return _best_candidate(cands, intent) or "Okay! Tell me more — I am listening."

    # ------------------------------------------------------------------- main
    def handle(self, user_input: str, history: list[tuple[str, str]],
               model, tokenizer, device, *, temperature=0.5, top_k=50,
               max_tokens=200) -> tuple[str, str]:
        """Route one user turn. Returns (reply, intent)."""
        intent = classify_intent(user_input)
        if self.show_intent:
            print(f"  [mind: {intent}]", flush=True)

        if intent == "SELF_DISCLOSURE":
            facts = extract_facts(user_input)
            if facts:
                self.mem.update(facts)
            reply = self._ack_self_disclosure(user_input, facts)

        elif intent == "MEMORY_QUERY":
            reply = self._answer_memory_query(user_input, model, tokenizer, device,
                                              max_new_tokens=max_tokens)

        elif intent in ("GREETING", "FAREWELL"):
            cands = self._model_replies(history, user_input, model, tokenizer,
                                        device, temperature=0.4, candidates=3,
                                        max_new_tokens=min(max_tokens, 60))
            reply = _best_candidate(cands, intent)
            if not reply:  # scripted fallback only if all candidates scored < 0
                reply = ("Hi there! How are you today?" if intent == "GREETING"
                         else "Goodbye! Come back and chat with me soon!")

        elif intent in ("THANKS", "APOLOGY", "EMOTION_SHARE", "IDENTITY",
                        "CAPABILITY", "AFFIRMATION", "SMALLTALK"):
            reply = self._social_reply(user_input, intent, history, model,
                                       tokenizer, device, max_tokens=max_tokens)

        elif intent == "QUESTION":
            reply = self._model_replies(history, user_input, model, tokenizer,
                                        device, temperature=0.4, candidates=1,
                                        max_new_tokens=max_tokens)[0]

        else:  # OTHER
            reply = self._model_replies(history, user_input, model, tokenizer,
                                        device, temperature=0.55, candidates=1,
                                        max_new_tokens=max_tokens)[0]
        return reply, intent


# --------------------------------------------------------------- prompt utils
def _build_prompt_text(history, user_input):
    parts = [f"{USER_TOKEN}{u}{ASSISTANT_TOKEN}{a}{END_TOKEN}" for u, a in history]
    parts.append(f"{USER_TOKEN}{user_input}{ASSISTANT_TOKEN}")
    return "".join(parts)


def _encode_context(history, user_input, context_length, tokenizer):
    history = history or []
    ids = tokenizer.encode(_build_prompt_text(history, user_input)).ids
    while len(ids) > context_length and history:
        history.pop(0)
        ids = tokenizer.encode(_build_prompt_text(history, user_input)).ids
    return ids[-context_length:]


# ------------------------------------------------------------------- scoring
def _word_rep(s: str) -> bool:
    ws = s.lower().split()
    return len(ws) > 3 and len(set(ws)) < len(ws) / 2


def _score(candidate: str, intent: str) -> int:
    score = 0
    low = candidate.lower()
    if intent == "GREETING":
        if _starts_with_any(candidate, GREETING_WORDS):
            score += 2
        if "?" in candidate:
            score += 1                      # greets back / asks how you are
        if low.startswith(("what", "how", "why", "where", "when")):
            score -= 2                      # answered a question with a question
        if _word_rep(candidate):
            score -= 2
    elif intent == "FAREWELL":
        if _starts_with_any(candidate, FAREWELL_WORDS + ("good", "see")):
            score += 2
        if _word_rep(candidate):
            score -= 2
    elif intent == "MEMORY_QUERY":
        score += 1 if 0 < len(candidate.split()) <= 25 else -1
    elif intent in ("THANKS", "APOLOGY"):
        score += 1 if 0 < len(candidate.split()) <= 20 else -1
        if _word_rep(candidate):
            score -= 2
    elif intent == "EMOTION_SHARE":
        # Empathy about the USER scores; the model narrating its own state doesn't.
        if any(w in low for w in NEGATIVE_WORDS) and not re.search(
                r"\bi (am|'m|feel)\b", low):
            score += 1
        if _word_rep(candidate):
            score -= 2
    elif intent == "IDENTITY":
        if re.search(r"\b(i am|i'm|buddy|ai|robot|computer|friendly)\b", low):
            score += 1
        if QUESTION_END_RE.search(candidate):
            score -= 1
    elif intent == "CAPABILITY":
        if re.search(r"\b(i can|i'd love|i would love|sure|of course)\b", low):
            score += 1
    elif intent == "AFFIRMATION":
        if candidate.split() and candidate.split()[0].lower() in (
                "yes", "yeah", "yep", "great", "okay", "ok", "sure", "awesome", "no"):
            score += 1
    return score


def _best_candidate(candidates: list[str], intent: str) -> str | None:
    scored = [(c, _score(c, intent)) for c in candidates if c]
    if not scored:
        return None
    best = max(scored, key=lambda cv: cv[1])
    # GREETING must actually greet (score >= 2 means a greeting-word start);
    # a mere "...?" bonus must not let a random template win.
    threshold = 2 if intent in ("GREETING", "FAREWELL") else 1
    return best[0] if best[1] >= threshold else None


# --------------------------------------------------------------------- CLI
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="inspect the cognition layer's memory")
    p.add_argument("--show", action="store_true", help="print stored facts")
    p.add_argument("--clear", action="store_true", help="erase memory.json")
    p.add_argument("--file", default="memory.json")
    a = p.parse_args()
    mem = MemoryStore(None if a.file == "-" else a.file)
    if a.clear:
        mem.clear()
        print("memory cleared")
    print("memory:", mem.summary())
