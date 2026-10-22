"""
Derive the splitter's action vocabulary from the dataset instead of guessing.

Reads the B-ACTION / I-ACTION spans in data/docEdit1000.json and prints every
distinct action span with its count. This is where multi_action/splitter.py's
ACTION_WORDS comes from.

    .\.venv\Scripts\python.exe scripts\mine_action_words.py
"""
import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "data", "docEdit1000.json")


def action_spans(example):
    """Yield each contiguous B-ACTION/I-ACTION span as lowercase text."""
    current = []
    for token, label in zip(example["tokens"], example["labels"]):
        if label == "B-ACTION":
            if current:
                yield " ".join(current)
            current = [token.lower()]
        elif label == "I-ACTION" and current:
            current.append(token.lower())
        else:
            if current:
                yield " ".join(current)
            current = []
    if current:
        yield " ".join(current)


def main():
    with open(DATASET, encoding="utf-8") as handle:
        examples = json.load(handle)

    counts = collections.Counter()
    for example in examples:
        counts.update(action_spans(example))

    print(f"{len(examples)} commands, {len(counts)} distinct action spans\n")
    for span, count in counts.most_common():
        print(f"{count:>5}  {span}")

    print("\nExcluded from ACTION_WORDS (nouns elsewhere in the data):")
    print("  sub, title — see the comment in multi_action/splitter.py")


if __name__ == "__main__":
    main()
