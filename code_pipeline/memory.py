# ============================================================
# memory.py
# ============================================================

import json
from pathlib import Path

MEMORY_DIR = Path(__file__).resolve().parent.parent / "memory"
MEMORY_DIR.mkdir(exist_ok=True)

MEMORY_FILE = MEMORY_DIR / "user_memory.json"

if not MEMORY_FILE.exists():
    with open(MEMORY_FILE, "w") as f:
        json.dump({}, f, indent=4)


def load_memory():

    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


def save_memory(memory):

    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)


def save_user_memory(value):

    memory = load_memory()

    # Create the list if it doesn't exist
    if "memories" not in memory:
        memory["memories"] = []

    memory["memories"].append(value)

    save_memory(memory)


def get_user_memory(key):

    memory = load_memory()

    return memory.get(key)


def get_all_memory():

    return load_memory()