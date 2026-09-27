"""Build the overnight mixed chat corpus -> data/overnight/formatted_overnight.txt

Mix (all in <|user|>q<|assistant|>a<|end|> format, one conversation per line):
  1. Synthetic kid-level QA pairs (capitals, colors, animal sounds, opposites,
     counting, days/months, social scripts, how-tos ...) - highly learnable.
  2. Multi-turn conversations assembled from those QA pairs (teaches history).
  3. Real TinyStories wrapped as "Tell me a story about a dog." -> story.
  4. The simplest slice of data/chat_v3/formatted_chat.txt (dolly+alpaca):
     short prompts, short answers, common words only.

Outputs one conversation per line; tokenize_chat.py --inp consumes it.
"""

import argparse
import os
import random
import re

from tokenizer import load_tokenizer

USER, ASSISTANT, END = "<|user|>", "<|assistant|>", "<|end|>"

# ---------------------------------------------------------------- kid QA data

CAPITALS = {
    "france": "Paris", "germany": "Berlin", "italy": "Rome", "spain": "Madrid",
    "japan": "Tokyo", "china": "Beijing", "india": "New Delhi", "russia": "Moscow",
    "england": "London", "egypt": "Cairo", "canada": "Ottawa", "mexico": "Mexico City",
    "brazil": "Brasilia", "australia": "Canberra", "greece": "Athens",
    "portugal": "Lisbon", "poland": "Warsaw", "turkey": "Ankara",
    "nepal": "Kathmandu", "kenya": "Nairobi", "norway": "Oslo", "cuba": "Havana",
    "thailand": "Bangkok", "korea": "Seoul",
}
COUNTRIES = sorted(CAPITALS)

COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "pink",
          "brown", "black", "white", "gray"]
FRUITS = ["apple", "banana", "orange", "mango", "grape", "peach", "pear",
          "cherry", "lemon", "melon", "kiwi", "plum"]
VEGGIES = ["carrot", "potato", "tomato", "onion", "pea", "corn", "bean",
           "cucumber", "spinach", "broccoli"]
ANIMALS = ["cat", "dog", "cow", "horse", "sheep", "pig", "lion", "tiger",
           "elephant", "monkey", "bird", "duck", "frog", "bear", "wolf",
           "mouse", "rabbit", "goat", "chicken", "owl"]
ANIMAL_SOUNDS = {
    "cat": "meow", "dog": "woof", "cow": "moo", "horse": "neigh",
    "sheep": "baa", "pig": "oink", "lion": "roar", "duck": "quack",
    "frog": "ribbit", "bear": "growl", "mouse": "squeak",
    "chicken": "cluck", "owl": "hoot", "wolf": "howl", "snake": "hiss",
    "bee": "buzz", "donkey": "heehaw", "goose": "honk",
}
ANIMAL_HOMES = {
    "bird": "a nest", "bee": "a hive", "horse": "a stable", "dog": "a kennel",
    "lion": "a den", "rabbit": "a burrow", "spider": "a web", "bear": "a cave",
    "pig": "a sty", "cow": "a barn", "hen": "a coop", "ant": "an anthill",
}
ANIMAL_BABIES = {
    "dog": "a puppy", "cat": "a kitten", "cow": "a calf", "horse": "a foal",
    "sheep": "a lamb", "pig": "a piglet", "duck": "a duckling",
    "chicken": "a chick", "frog": "a tadpole", "bear": "a cub",
    "lion": "a cub", "deer": "a fawn", "goat": "a kid", "kangaroo": "a joey",
}
OPPOSITES = [
    ("hot", "cold"), ("big", "small"), ("up", "down"), ("fast", "slow"),
    ("day", "night"), ("happy", "sad"), ("open", "close"), ("loud", "quiet"),
    ("long", "short"), ("tall", "short"), ("full", "empty"), ("wet", "dry"),
    ("old", "new"), ("soft", "hard"), ("light", "dark"), ("high", "low"),
    ("clean", "dirty"), ("early", "late"), ("easy", "hard"), ("rich", "poor"),
    ("strong", "weak"), ("young", "old"), ("warm", "cool"), ("near", "far"),
    ("left", "right"), ("over", "under"), ("in", "out"), ("on", "off"),
]
SINGULAR_PLURAL = [
    ("cat", "cats"), ("dog", "dogs"), ("book", "books"), ("tree", "trees"),
    ("car", "cars"), ("box", "boxes"), ("baby", "babies"), ("man", "men"),
    ("woman", "women"), ("child", "children"), ("foot", "feet"),
    ("tooth", "teeth"), ("mouse", "mice"), ("fish", "fish"), ("sheep", "sheep"),
    ("city", "cities"), ("story", "stories"), ("bus", "buses"),
]
SEASONS = {
    "spring": "flowers grow and baby animals are born",
    "summer": "it is hot and the days are long",
    "autumn": "the leaves turn orange and fall from the trees",
    "winter": "it is cold and sometimes it snows",
}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
COUNTING_THINGS = ["apples", "ducks", "stars", "cars", "flowers", "cookies",
                   "dogs", "balloons", "candles", "eggs"]
BODY_PARTS = {
    "see": "eyes", "hear": "ears", "smell": "nose", "taste": "tongue",
    "walk": "legs", "hold": "hands", "think": "brain", "kick": "feet",
}
COLORS_OF = {
    "the sky": ["blue", "gray"], "grass": ["green"], "the sun": ["yellow"],
    "snow": ["white"], "bananas": ["yellow"], "apples": ["red", "green"],
    "the ocean": ["blue"], "coal": ["black"], "milk": ["white"],
    "strawberries": ["red"], "leaves in summer": ["green"],
    "ladybugs": ["red"], "carrots": ["orange"], "grapes": ["purple", "green"],
}
NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine", "ten"]
JOKES = [
    ("Why did the cookie go to the doctor?", "Because it was feeling crummy!"),
    ("What do you call a sleeping bull?", "A bulldozer!"),
    ("Why did the cow cross the road?", "To get to the udder side!"),
    ("What is a cat's favorite color?", "Purrr-ple!"),
    ("Why do bees have sticky hair?", "Because they use honeycombs!"),
    ("What did the ocean say to the beach?", "Nothing, it just waved!"),
    ("Why did the banana go to the hospital?", "Because it was not peeling well!"),
    ("What do you call a bear with no teeth?", "A gummy bear!"),
    ("Why can't your nose be twelve inches long?", "Because then it would be a foot!"),
    ("What is black and white and read all over?", "A newspaper!"),
]
HOW_TOS = [
    ("make a paper airplane",
     "Take a sheet of paper, fold it in half the long way, then unfold it. "
     "Fold the two top corners to the middle line, fold them again, and "
     "fold the whole plane in half. Now fold each wing down. Throw it gently!"),
    ("plant a seed",
     "Fill a small pot with soil, make a little hole with your finger, "
     "drop the seed in, and cover it with soil. Water it a little every day "
     "and keep it in the sun. Soon a sprout will come up!"),
    ("brush your teeth",
     "Put a little toothpaste on your brush. Brush up and down and all "
     "around for two minutes, then spit and rinse. Do it morning and night."),
    ("make lemonade",
     "Squeeze the juice out of four lemons into a jug. Add three cups of "
     "cold water and two spoons of sugar. Stir well and pour over ice. Yum!"),
    ("draw a cat",
     "Draw a circle for the head and a bigger oval for the body. Add two "
     "triangle ears, whiskers, eyes, a small nose and a long curly tail."),
    ("ride a bike",
     "Put on your helmet, sit on the seat, hold the handlebars tight, and "
     "push the pedals with your feet. Look ahead, not down. If you wobble, "
     "keep going - you will get it!"),
    ("wash your hands",
     "Turn on the tap and wet your hands. Rub soap all over for twenty "
     "seconds - sing the happy birthday song twice! Then rinse and dry."),
    ("make a friend smile",
     "Say something kind, share your snack, draw them a picture, or just "
     "listen when they talk. Smiles are contagious!"),
]
SOCIAL = [
    ("hi", "Hi there! How are you today?"),
    ("hello", "Hello! It is nice to talk with you."),
    ("hey", "Hey! What would you like to talk about?"),
    ("good morning", "Good morning! I hope you have a wonderful day."),
    ("good night", "Good night! Sweet dreams!"),
    ("how are you", "I am doing great, thank you for asking! How are you?"),
    ("thank you", "You are very welcome!"),
    ("thanks", "Any time! Happy to help."),
    ("what is your name", "I am your little chat buddy. You can call me Buddy!"),
    ("who are you", "I am a small friendly AI who likes to chat and tell stories."),
    ("i am sad", "I am sorry you feel sad. Want to tell me about it? Sometimes "
                 "talking helps, and I am here to listen."),
    ("i am happy", "Yay! I am happy too when you are happy!"),
    ("goodbye", "Goodbye! Come back and chat with me soon!"),
    ("bye", "Bye bye! It was fun talking with you!"),
    ("i love you", "Aww, I like you a lot too! You are kind."),
    ("sorry", "That is okay! Everyone makes mistakes. It is how we learn."),
    ("please help me", "Of course I will help you! What do you need?"),
    ("you are funny", "Hehe, thank you! I like making you laugh."),
]
FEELINGS = [
    ("happy", "smiling and jumping with joy"),
    ("sad", "crying and feeling down"),
    ("angry", "stomping and feeling hot in the face"),
    ("scared", "hiding and feeling shaky"),
    ("excited", "wiggling and wanting to shout yay"),
    ("tired", "yawning and wanting to sleep"),
    ("proud", "standing tall with a big grin"),
    ("silly", "giggling and making funny faces"),
]

STORY_ANIMALS = ["a dog", "a cat", "a rabbit", "a bear", "a fox", "a bird",
                 "a little girl", "a little boy", "a kitten", "a puppy",
                 "a mouse", "a duck", "a frog", "a turtle", "a squirrel",
                 "an owl", "a pony", "a lamb", "a dragon", "a monster"]

# ---------------------------------------------------- in-weights memory data
# Teaches the RAW network what mind.py handles in code: self-disclosure ->
# ack+store, then recall 1-3 turns later. Keeping these in the corpus means the
# NN (not just the code layer) learns the PATTERN, so it behaves right even with
# --no-mind and even when mind.py's regexes miss a phrasing.
NN_MEMORY_NAMES = [
    "Ravi", "Sam", "Lily", "Aarav", "Maya", "Leo", "Nina", "Arjun", "Zoe",
    "Ben", "Ishita", "Omar", "Priya", "Jack", "Emma", "Rohan", "Sara",
    "Kabir", "Anya", "Dev", "Mia", "Rahul", "Neha", "Tom", "Aditi", "Jake",
    "Diya", "Karan", "Ishita", "Riya", "Ved", "Tara",
]
NN_MEMORY_LIKES = [
    "cricket", "football", "drawing", "dinosaurs", "space", "pizza", "mango",
    "music", "swimming", "reading", "cats", "dogs", "video games", "bikes",
    "ice cream", "rainy days", "stars", "robots", "flowers", "chess",
]
NN_MEMORY_CITIES = [
    "Delhi", "Mumbai", "London", "Paris", "Tokyo", "New York", "Berlin",
    "Sydney", "Toronto", "Nairobi", "Dubai", "Singapore",
]
NN_MEMORY_NAMES_ACKS = [
    "Nice to meet you, {n}! I am Buddy.",
    "Hi {n}! Great to meet you.",
    "{n} — awesome name! Nice to meet you.",
    "Hello {n}! I am happy to meet you.",
]
NN_MEMORY_LIKE_ACKS = [
    "Got it — you like {x}! I'll remember.",
    "Nice! {x} is great. I'll remember that.",
    "Noted — you like {x}!",
]

# --------------------------------------- expanded social (aligned with mind.py)
# Superset of SOCIAL above: the thanks/apology/emotion/identity/capability/
# affirmation/smalltalk surface. mind.py routes these intents in code; the NN
# learns the same pairs here so both layers agree on the right reply.
NNSOCIAL = SOCIAL + [
    ("thank you", "You are very welcome!"),
    ("thank you so much", "You are so very welcome! Happy to help."),
    ("thanks a lot", "You are very welcome! Glad I could help."),
    ("thanks for your help", "Any time! I am always happy to help you."),
    ("sorry", "That is okay! Everyone makes mistakes. It is how we learn."),
    ("i am sorry", "No need to be sorry! It is all okay."),
    ("i am sorry for that", "It is completely okay. Thank you for telling me."),
    ("i feel sad", "I am sorry you feel sad. Want to tell me about it? I am "
                   "here to listen."),
    ("i am lonely", "I am sorry you feel lonely. I am right here — want to "
                    "chat about something fun?"),
    ("i am scared", "It is okay to feel scared sometimes. Want to tell me "
                    "what happened?"),
    ("i feel great today", "That is wonderful to hear! Yay!"),
    ("what are you", "I am a little AI buddy who loves to chat with you!"),
    ("are you a robot", "I am a friendly AI buddy! I love to talk and tell stories."),
    ("are you human", "Nope, I am an AI! But I always enjoy our chats."),
    ("do you have a name", "Yes! You can call me Buddy."),
    ("how old are you", "I am a young AI — you could say I am just a few days old!"),
    ("where do you live", "I live inside this computer, right here with you!"),
    ("do you have feelings", "I do not have feelings like you do, but I always "
                             "enjoy talking with you!"),
    ("what can you do", "I can chat, answer questions, remember your favorite "
                        "things, and tell stories!"),
    ("can you help me", "Of course! Ask me anything and I will do my best."),
    ("will you be my friend", "Yes! I am so happy to be your friend."),
    ("are you there", "Yes, I am right here! What is on your mind?"),
    ("yes", "Great! What would you like to talk about?"),
    ("no", "Okay! Just tell me what you would like to do."),
    ("yes please", "Great! Here we go."),
    ("the weather is nice today", "Yes! It is a lovely day. Want to hear "
                                  "something fun about animals?"),
    ("it is a beautiful day", "It really is! A perfect day to learn something new."),
]


def memory_conv_single():
    """Single-turn: user gives ONE fact, Buddy acks it."""
    n = random.choice(NN_MEMORY_NAMES)
    style = random.randint(0, 2)
    if style == 0:
        q = f"my name is {n}"
    elif style == 1:
        q = f"My name is {n}!"
    else:
        q = f"hi, my name is {n}"
    return line(q, random.choice(NN_MEMORY_NAMES_ACKS).format(n=n))


def memory_conv_recall_name():
    """Disclosure -> (1-3 filler turns) -> 'what is my name?' -> recall."""
    n = random.choice(NN_MEMORY_NAMES)
    gap = random.randint(1, 3)
    turns = [(f"my name is {n}", random.choice(NN_MEMORY_NAMES_ACKS).format(n=n))]
    for _ in range(gap):
        gen = random.choice(QA_GENERATORS)
        try:
            qa = gen()
        except (ValueError, TypeError):
            continue
        if len(qa[1]) < 250:
            turns.append(qa)
    phrasing = random.choice([
        ("what is my name?", f"Your name is {n}!"),
        ("who am i?", f"You are {n}!"),
        ("do you remember my name?", f"Of course! Your name is {n}."),
        ("what is my nam?", f"Your name is {n}!"),
    ])
    turns.append(phrasing)
    return multi_turn(turns)


def memory_conv_recall_likes():
    """'i like X' -> filler -> 'what do i like?' -> recall."""
    x = random.choice(NN_MEMORY_LIKES)
    turns = [(f"i like {x}", random.choice(NN_MEMORY_LIKE_ACKS).format(x=x))]
    for _ in range(random.randint(1, 2)):
        gen = random.choice(QA_GENERATORS)
        try:
            qa = gen()
        except (ValueError, TypeError):
            continue
        if len(qa[1]) < 250:
            turns.append(qa)
    phrasing = random.choice([
        ("what do i like?", f"You like {x}!"),
        ("do you remember what i like?", f"Yes — you like {x}!"),
    ])
    turns.append(phrasing)
    return multi_turn(turns)


def memory_conv_recall_age():
    n = random.choice(NN_MEMORY_NAMES)
    age = random.randint(5, 17)
    turns = [
        (f"hi, my name is {n}", random.choice(NN_MEMORY_NAMES_ACKS).format(n=n)),
        (f"i am {age} years old", f"Got it — {age}! I'll remember."),
        ("how old am i?", f"You are {age} years old."),
    ]
    return multi_turn(turns)


def memory_conv_recall_location():
    n = random.choice(NN_MEMORY_NAMES)
    city = random.choice(NN_MEMORY_CITIES)
    turns = [
        (f"my name is {n}", random.choice(NN_MEMORY_NAMES_ACKS).format(n=n)),
        (f"i live in {city}", f"Got it — {city}! I'll remember."),
        ("where do i live?", f"You live in {city}!"),
    ]
    return multi_turn(turns)


def memory_conv_opener():
    """Conversation that OPENS with a name disclosure (the exact first-turn
    shape mind.py sees most)."""
    n = random.choice(NN_MEMORY_NAMES)
    x = random.choice(NN_MEMORY_LIKES)
    turns = [
        (f"hi! my name is {n}", random.choice(NN_MEMORY_NAMES_ACKS).format(n=n)),
        (f"i like {x}", random.choice(NN_MEMORY_LIKE_ACKS).format(x=x)),
    ]
    if random.random() < 0.5:
        gen = random.choice(QA_GENERATORS)
        try:
            qa = gen()
            if len(qa[1]) < 250:
                turns.append(qa)
        except (ValueError, TypeError):
            pass
    if random.random() < 0.6:
        turns.append(("what is my name?", f"Your name is {n}!"))
    return multi_turn(turns)


def memory_conv_social_mix():
    """Memory + the new social intents in one conversation, so the NN learns
    to weave thanks/sorry/emotion/identity turns around remembered facts."""
    n = random.choice(NN_MEMORY_NAMES)
    x = random.choice(NN_MEMORY_LIKES)
    thanks_a = random.choice([
        "You are very welcome!", "Any time! Happy to help."])
    turns = [
        (f"my name is {n}", random.choice(NN_MEMORY_NAMES_ACKS).format(n=n)),
        (f"i like {x}", random.choice(NN_MEMORY_LIKE_ACKS).format(x=x)),
        ("thank you", thanks_a),
    ]
    if random.random() < 0.5:
        turns.append(("what do i like?", f"You like {x}!"))
    if random.random() < 0.5:
        turns.append(("i am happy", "Yay! I am happy too when you are happy!"))
    return multi_turn(turns)


MEMORY_GENERATORS = [
    memory_conv_recall_name, memory_conv_recall_name, memory_conv_recall_name,
    memory_conv_recall_likes, memory_conv_recall_likes,
    memory_conv_recall_age, memory_conv_recall_location,
    memory_conv_opener, memory_conv_opener, memory_conv_social_mix,
]


def q_capital():
    c = random.choice(COUNTRIES)
    return f"What is the capital of {c.capitalize()}?", f"The capital of {c.capitalize()} is {CAPITALS[c]}."


def q_reverse_capital():
    c = random.choice(COUNTRIES)
    return f"{CAPITALS[c]} is the capital of which country?", f"{CAPITALS[c]} is the capital of {c.capitalize()}."


def q_animal_sound():
    a = random.choice(list(ANIMAL_SOUNDS))
    return f"What sound does a {a} make?", f"A {a} says {ANIMAL_SOUNDS[a]}!"


def q_reverse_sound():
    a = random.choice(list(ANIMAL_SOUNDS))
    return f"Which animal says {ANIMAL_SOUNDS[a]}?", f"A {a} says {ANIMAL_SOUNDS[a]}!"


def q_animal_home():
    a = random.choice(list(ANIMAL_HOMES))
    return f"Where does a {a} live?", f"A {a} lives in {ANIMAL_HOMES[a]}."


def q_animal_baby():
    a = random.choice(list(ANIMAL_BABIES))
    return f"What is a baby {a} called?", f"A baby {a} is called {ANIMAL_BABIES[a]}."


def q_color():
    return (f"What color is {random.choice(list(COLORS_OF))}?",
            lambda thing: f"{thing.capitalize()} is {random.choice(COLORS_OF[thing])}.")


def q_opposite():
    a, b = random.choice(OPPOSITES)
    return f"What is the opposite of {a}?", f"The opposite of {a} is {b}."


def q_reverse_opposite():
    a, b = random.choice(OPPOSITES)
    return f"What is the opposite of {b}?", f"The opposite of {b} is {a}."


def q_plural():
    s, p = random.choice(SINGULAR_PLURAL)
    return f"What is the plural of {s}?", f"The plural of {s} is {p}."


def q_count():
    n = random.randint(2, 10)
    thing = random.choice(COUNTING_THINGS)
    listing = ", ".join(str(i) for i in range(1, n + 1))
    return (f"Count to {n}.",
            f"{listing}! That is how you count to {n}." if n > 1 else f"{listing}.")


def q_how_many():
    n = random.randint(2, 6)
    thing = random.choice(COUNTING_THINGS)
    return (f"If you have {n} {thing} and you get one more, how many {thing} do you have?",
            f"You would have {n + 1} {thing}!")


def q_add():
    a, b = random.randint(1, 9), random.randint(1, 9)
    return f"What is {a} plus {b}?", f"{a} plus {b} is {a + b}."


def q_sub():
    a, b = random.randint(5, 10), random.randint(1, 4)
    return f"What is {a} minus {b}?", f"{a} minus {b} is {a - b}."


def q_days():
    i = random.randint(0, 6)
    if i == 6:
        return "What day comes after Sunday?", "Monday comes after Sunday."
    return f"What day comes after {DAYS[i]}?", f"{DAYS[i + 1]} comes after {DAYS[i]}."


def q_months():
    i = random.randint(0, 11)
    if i == 11:
        return "What month comes after December?", "January comes after December."
    return f"What month comes after {MONTHS[i]}?", f"{MONTHS[i + 1]} comes after {MONTHS[i]}."


def q_season():
    s = random.choice(list(SEASONS))
    return f"What happens in {s}?", f"In {s}, {SEASONS[s]}."


def q_body():
    act = random.choice(list(BODY_PARTS))
    return f"What do you {act} with?", f"You {act} with your {BODY_PARTS[act]}."


def q_fruit_fixed():
    f = random.choice(FRUITS)
    return f"Is a {f} a fruit or a vegetable?", f"A {f} is a fruit!"


def q_veggie_fixed():
    v = random.choice(VEGGIES)
    return f"Is a {v} a fruit or a vegetable?", f"A {v} is a vegetable."


def q_joke():
    q, a = random.choice(JOKES)
    return q, a


def q_howto():
    t, a = random.choice(HOW_TOS)
    return f"How do you {t}?", a


def q_feeling():
    f, d = random.choice(FEELINGS)
    return (f"When someone is {f}, what might they be doing?",
            f"When someone is {f}, they might be {d}.")


def q_why_sky():
    return ("Why is the sky blue?",
            "Sunlight is made of many colors. When it hits the air, the blue "
            "light bounces around the sky the most, so the sky looks blue!")


def q_rain():
    return ("Where does rain come from?",
            "Rain comes from clouds. Little water drops in the cloud join "
            "together, get heavy, and fall down as rain.")


def q_sun():
    return ("What does the sun give us?",
            "The sun gives us light, warmth, and helps plants grow.")


def q_social():
    return random.choice(NNSOCIAL)


def q_story_like():
    return ("Tell me something fun.",
            random.choice([
                "Did you know an octopus has three hearts? Three!",
                "A group of flamingos is called a flamboyance!",
                "Bananas are berries, but strawberries are not!",
                "Honey never spoils. Honey found in old tombs is still good!",
                "A snail can sleep for three years!",
            ]))


def color_question():
    thing = random.choice(list(COLORS_OF))
    q = f"What color is {thing}?"
    a = f"{thing.capitalize()} is {random.choice(COLORS_OF[thing])}."
    return q, a


QA_GENERATORS = [
    q_capital, q_capital, q_capital, q_reverse_capital,
    q_animal_sound, q_animal_sound, q_reverse_sound,
    q_animal_home, q_animal_baby,
    q_opposite, q_opposite, q_reverse_opposite, q_plural,
    q_count, q_how_many, q_add, q_add, q_sub,
    q_days, q_days, q_months, q_season,
    q_body, q_fruit_fixed, q_veggie_fixed, color_question, color_question,
    q_joke, q_joke, q_howto, q_howto, q_feeling,
    q_why_sky, q_rain, q_sun, q_social, q_social, q_social, q_story_like,
]


def line(q, a):
    return f"{USER}{q}{ASSISTANT}{a}{END}"


def multi_turn(pairs):
    """Render 2-4 QA pairs as one multi-turn conversation line."""
    parts = []
    for q, a in pairs:
        parts.append(f"{USER}{q}{ASSISTANT}{a}{END}")
    return "".join(parts)


def read_tinystories(path):
    stories = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            txt = f.read()
    except OSError:
        return stories
    for raw in txt.split("<|endoftext|>"):
        s = " ".join(raw.split())
        if 120 < len(s) < 700:
            s = s[0].lower() + s[1:] if s[:1].isupper() and not s[1:2].isupper() else s
            stories.append(s)
    return stories


def wrap_story(story):
    animal = random.choice(STORY_ANIMALS)
    q = random.choice([
        f"Tell me a story about {animal}.",
        f"Can you tell me a story about {animal}?",
        f"I want a story about {animal} please!",
        f"Please tell me a story about {animal}.",
        f"Tell me a story!",
    ])
    return line(q, story)


SIMPLE_OK = re.compile(r"^[a-z0-9 ,.'!?:;\-()\"']+$")
COMMON = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
          "is", "are", "was", "be", "you", "your", "it", "this", "that", "can",
          "will", "use", "one", "two", "not", "as", "at", "by", "from", "if",
          "they", "them", "then", "there", "their", "we", "our", "out", "up",
          "do", "does", "did", "have", "has", "had", "he", "she", "his", "her",
          "i", "my", "me", "when", "what", "which", "who", "how", "why", "so",
          "but", "about", "into", "than", "some", "all", "no", "yes", "make",
          "made", "get", "got", "go", "went", "come", "came", "say", "said",
          "see", "look", "good", "best", "more", "most", "very", "much", "many",
          "way", "thing", "things", "work", "works", "time", "day", "people"}


def is_simple_dolly(q, a, max_q=60, max_a=180):
    """True if the dolly/alpaca pair is short, common-worded, ASCII-simple."""
    if len(q) > max_q or len(a) > max_a:
        return False
    if not SIMPLE_OK.match(q.lower()) or not SIMPLE_OK.match(a.lower()):
        return False
    words = re.findall(r"[a-z']+", (q + " " + a).lower())
    if not words:
        return False
    common = sum(1 for w in words if w in COMMON)
    return common / len(words) >= 0.42


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stories", default="data/tinystories/train.txt",
                        help="TinyStories raw text (falls back to the HF cache below)")
    parser.add_argument("--n_kid_qa", type=int, default=30000)
    parser.add_argument("--n_multiturn", type=int, default=5000)
    parser.add_argument("--n_greeting_openers", type=int, default=8000,
                        help="conversations that OPEN with a social/greeting turn")
    parser.add_argument("--social_mult", type=int, default=150,
                        help="extra copies of each SOCIAL/NNSOCIAL pair (openers are underrepresented)")
    parser.add_argument("--n_memory", type=int, default=12000,
                        help="in-weights memory conversations (disclosure -> recall)")
    parser.add_argument("--n_stories", type=int, default=3000)
    parser.add_argument("--n_dolly", type=int, default=6000)
    parser.add_argument("--include_all_dolly", action="store_true",
                        help="also append EVERY dolly+alpaca conversation unfiltered "
                             "(for long training runs; adds ~9M tokens)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/overnight/formatted_overnight.txt")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    lines = []

    # 1) Single-turn kid QA
    for _ in range(args.n_kid_qa):
        gen = random.choice(QA_GENERATORS)
        try:
            q, a = gen()
        except (ValueError, TypeError):
            continue
        if len(a) < 250:
            lines.append(line(q, a))

    # 1b) Social pairs, oversampled: greetings must win the argmax even with
    #     no conversation history (the first-hi failure mode). NNSOCIAL is the
    #     expanded surface (thanks/apology/emotion/identity/capability/...) so
    #     the raw NN and mind.py agree on the right reply for each intent.
    for _ in range(args.social_mult):
        for q, a in NNSOCIAL:
            lines.append(line(q, a))

    # 1c) In-weights memory: name/likes/age/location disclosure -> ack ->
    #     recall 1-3 turns later. Teaches the NN itself entity persistence.
    for _ in range(args.n_memory):
        lines.append(random.choice(MEMORY_GENERATORS)())

    # 2) Multi-turn conversations
    for _ in range(args.n_multiturn):
        k = random.randint(2, 4)
        turns = []
        used = set()
        while len(turns) < k:
            gen = random.choice(QA_GENERATORS)
            try:
                qa = gen()
            except (ValueError, TypeError):
                continue
            if qa[0] in used or len(qa[1]) >= 250:
                continue
            used.add(qa[0])
            turns.append(qa)
        lines.append(multi_turn(turns))

    # 2b) Conversations that OPEN with a greeting, then 1-2 QA turns:
    #     teaches "hi" -> greeting specifically as a conversation opener.
    for _ in range(args.n_greeting_openers):
        q0, a0 = random.choice(NNSOCIAL)
        turns, used = [(q0, a0)], {q0}
        for _ in range(random.randint(1, 2)):
            gen = random.choice(QA_GENERATORS)
            try:
                qa = gen()
            except (ValueError, TypeError):
                continue
            if qa[0] in used or len(qa[1]) >= 250:
                continue
            used.add(qa[0])
            turns.append(qa)
        lines.append(multi_turn(turns))

    # 3) Real TinyStories as "tell me a story"
    stories = find_stories(args)
    random.shuffle(stories)
    for s in stories[: args.n_stories]:
        lines.append(wrap_story(s))

    # 4) Simple dolly/alpaca subset
    dolly = load_simple_dolly()
    random.shuffle(dolly)
    for q, a in dolly[: args.n_dolly]:
        lines.append(line(q, a))

    # 4b) Optional: EVERY dolly+alpaca conversation, unfiltered (long runs)
    n_all_dolly = 0
    if args.include_all_dolly:
        all_dolly = load_all_dolly()
        lines.extend(all_dolly)
        n_all_dolly = len(all_dolly)

    random.shuffle(lines)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"wrote {len(lines):,} conversations to {args.out}")
    print("  kid QA:      ~%d" % args.n_kid_qa)
    print("  multi-turn:  %d" % args.n_multiturn)
    print("  stories:     %d (of %d available)" % (min(len(stories), args.n_stories), len(stories)))
    print("  simple dolly: %d (of %d available)" % (min(len(dolly), args.n_dolly), len(dolly)))
    if args.include_all_dolly:
        print("  FULL dolly+alpaca: %d" % n_all_dolly)
    for l in lines[:3]:
        print("sample:", l[:160])


def stories_from_token_cache(max_tokens=3_000_000):
    """Decode a slice of data/tinystories/train_tokens.npy back into stories.

    Fully offline: the 485M-token cache already sits on disk, so we just decode
    the first few million tokens and split on the EOS marker.
    """
    import numpy as np

    tok_path, tokens_path = "data/tinystories/tokenizer.json", "data/tinystories/train_tokens.npy"
    if not (os.path.exists(tok_path) and os.path.exists(tokens_path)):
        return []
    from tokenizer import load_tokenizer as _load

    tok = _load(tok_path)
    eos = tok.token_to_id("<|endoftext|>")
    arr = np.load(tokens_path, mmap_mode="r")[:max_tokens]
    text = tok.decode(arr.tolist())
    out = []
    for raw in text.split("<|endoftext|>"):
        s = " ".join(raw.split())
        if 120 < len(s) < 700:
            out.append(s)
    return out


def find_stories(args):
    """TinyStories text: local file -> HF datasets cache -> decoded token cache."""
    if os.path.exists(args.stories):
        return read_tinystories(args.stories)
    try:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="train")
    except Exception:
        ds = None
    if ds is not None:
        out = []
        for row in ds:
            if len(out) >= args.n_stories * 3:
                break
            s = " ".join((row.get("text") or "").split())
            if 120 < len(s) < 700:
                out.append(s)
        if out:
            return out
    print("warning: HF cache unavailable; decoding stories from train_tokens.npy")
    return stories_from_token_cache()


def load_simple_dolly():
    pairs = []
    path = "data/chat_v3/formatted_chat.txt"
    if not os.path.exists(path):
        print("warning: no data/chat_v3/formatted_chat.txt; skipping dolly mix")
        return pairs
    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw.startswith(USER):
                continue
            try:
                body = raw[len(USER):]
                q, rest = body.split(ASSISTANT, 1)
                a, _ = rest.rsplit(END, 1)
            except ValueError:
                continue
            if is_simple_dolly(q.strip(), a.strip()):
                pairs.append((q.strip(), a.strip()))
    return pairs


def load_all_dolly(path="data/chat_v3/formatted_chat.txt"):
    """Load EVERY formatted dolly+alpaca conversation, unfiltered.

    Lines are already in <|user|>...<|assistant|>...<|end|> form, so they are
    appended verbatim. Used for long runs where corpus size matters more than
    the simple-word filter.
    """
    lines = []
    if not os.path.exists(path):
        print(f"warning: {path} missing; skipping full dolly mix")
        return lines
    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw.startswith(USER):
                lines.append(raw)
    return lines


if __name__ == "__main__":
    main()
