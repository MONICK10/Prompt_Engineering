"""Generate data/docEdit_tanglish_500.json and data/docEdit1000.json."""

import json
import os

ANCHOR_PAIRS = {
    'bold': [
        ('introduction', 'conclusion'), ('car', 'accident'), ('title', 'summary'),
        ('name', 'address'), ('chapter', 'end'), ('first', 'last'),
        ('header', 'footer'), ('question', 'answer'), ('start', 'finish'), ('topic', 'result'),
    ],
    'italic': [
        ('note', 'end'), ('warning', 'stop'), ('quote', 'citation'), ('definition', 'example'),
        ('term', 'description'), ('phrase', 'period'), ('word', 'sentence'), ('line', 'paragraph'),
        ('clause', 'section'), ('label', 'value'),
    ],
    'delete': [
        ('paragraph', 'end'), ('line', 'stop'), ('sentence', 'period'), ('word', 'comma'),
        ('section', 'footer'), ('phrase', 'end'), ('clause', 'break'), ('block', 'boundary'),
        ('item', 'close'), ('note', 'margin'),
    ],
    'underline': [
        ('keyword', 'term'), ('link', 'reference'), ('date', 'time'), ('author', 'publisher'),
        ('title', 'subtitle'), ('label', 'field'), ('point', 'conclusion'), ('step', 'result'),
        ('entry', 'value'), ('source', 'citation'),
    ],
    'heading2': [
        ('overview', 'details'), ('abstract', 'introduction'), ('background', 'methods'),
        ('results', 'discussion'), ('summary', 'conclusion'), ('section', 'end'),
        ('chapter', 'close'), ('topic', 'subtopic'), ('main', 'sub'), ('part', 'segment'),
    ],
    'heading3': [
        ('subsection', 'end'), ('point', 'note'), ('item', 'detail'), ('step', 'description'),
        ('element', 'component'), ('unit', 'measure'), ('entry', 'exit'), ('start', 'close'),
        ('begin', 'finish'), ('open', 'shut'),
    ],
    'replace': [
        ('car', 'vehicle'), ('accident', 'incident'), ('introduction', 'overview'),
        ('title', 'heading'), ('conclusion', 'summary'), ('chapter', 'section'),
        ('method', 'approach'), ('result', 'outcome'), ('note', 'remark'), ('bold', 'strong'),
    ],
    'insert-after': [
        ('conclusion', 'summary'), ('hello', 'introduction'), ('note', 'warning'),
        ('important', 'title'), ('disclaimer', 'header'), ('appendix', 'section'),
        ('update', 'chapter'), ('comment', 'paragraph'), ('signature', 'footer'),
        ('end', 'conclusion'),
    ],
    'font-increase': [
        ('title', 'header'), ('introduction', 'conclusion'), ('abstract', 'keywords'),
        ('chapter', 'section'), ('heading', 'subheading'), ('start', 'end'),
        ('first', 'last'), ('topic', 'result'), ('note', 'summary'), ('paragraph', 'footer'),
    ],
    'highlight': [
        ('important', 'note'), ('keyword', 'term'), ('definition', 'example'),
        ('title', 'abstract'), ('warning', 'stop'), ('introduction', 'conclusion'),
        ('car', 'accident'), ('chapter', 'end'), ('header', 'footer'), ('start', 'finish'),
    ],
}

# Each action phrase: (tokens_list, labels_list)  — first token = B-ACTION, rest = O
ACTION_PHRASES = {
    'bold': [
        (['bold', 'panu'],        ['B-ACTION', 'O']),
        (['bold', 'sei'],         ['B-ACTION', 'O']),
        (['bold', 'panuka'],      ['B-ACTION', 'O']),
        (['bold', 'aa', 'maathu'],['B-ACTION', 'O', 'O']),
        (['bold', 'pannuga'],     ['B-ACTION', 'O']),
    ],
    'italic': [
        (['italic', 'panu'],         ['B-ACTION', 'O']),
        (['italic', 'sei'],          ['B-ACTION', 'O']),
        (['italic', 'aa', 'maathu'], ['B-ACTION', 'O', 'O']),
        (['italic', 'panuka'],       ['B-ACTION', 'O']),
        (['italic', 'pannuga'],      ['B-ACTION', 'O']),
    ],
    'delete': [
        (['delete', 'panu'],    ['B-ACTION', 'O']),
        (['delete', 'sei'],     ['B-ACTION', 'O']),
        (['delete', 'panra'],   ['B-ACTION', 'O']),
        (['edutha', 'podu'],    ['B-ACTION', 'O']),
        (['delete', 'pannuga'], ['B-ACTION', 'O']),
    ],
    'underline': [
        (['underline', 'panu'],          ['B-ACTION', 'O']),
        (['keezha', 'kodu', 'podu'],     ['B-ACTION', 'O', 'O']),
        (['keezhey', 'kodu'],            ['B-ACTION', 'O']),
        (['underline', 'sei'],           ['B-ACTION', 'O']),
        (['underline', 'panuka'],        ['B-ACTION', 'O']),
    ],
    'heading2': [
        (['heading', 'aa', 'maathu'],  ['B-ACTION', 'O', 'O']),
        (['title', 'aa', 'panu'],      ['B-ACTION', 'O', 'O']),
        (['heading', 'ku', 'maathu'],  ['B-ACTION', 'O', 'O']),
        (['heading', 'aa', 'panu'],    ['B-ACTION', 'O', 'O']),
        (['title', 'ku', 'maathu'],    ['B-ACTION', 'O', 'O']),
    ],
    'heading3': [
        (['heading3', 'aa', 'maathu'],       ['B-ACTION', 'O', 'O']),
        (['sub', 'heading', 'aa', 'panu'],   ['B-ACTION', 'O', 'O', 'O']),
        (['heading3', 'panu'],               ['B-ACTION', 'O']),
        (['heading3', 'ku', 'maathu'],       ['B-ACTION', 'O', 'O']),
        (['sub', 'heading', 'panu'],         ['B-ACTION', 'O', 'O']),
    ],
    'font-increase': [
        (['perusu', 'panu'],                   ['B-ACTION', 'O']),
        (['perusu', 'sei'],                    ['B-ACTION', 'O']),
        (['font', 'perusu', 'aa', 'maathu'],   ['B-ACTION', 'O', 'O', 'O']),
        (['perusu', 'panuka'],                 ['B-ACTION', 'O']),
        (['perusu', 'pannuga'],                ['B-ACTION', 'O']),
    ],
    'highlight': [
        (['highlight', 'panu'],   ['B-ACTION', 'O']),
        (['mark', 'panu'],        ['B-ACTION', 'O']),
        (['highlight', 'sei'],    ['B-ACTION', 'O']),
        (['highlight', 'panuka'], ['B-ACTION', 'O']),
        (['mark', 'sei'],         ['B-ACTION', 'O']),
    ],
}

# 5 range templates: (from_word_tokens, to_word_tokens_or_None)
# None means use English "to" directly (no separate to_word after to_anchor)
RANGE_TEMPLATES = [
    (['la', 'irunthu'], ['varikum']),
    (['irunthu'],        ['vara']),
    (['irundhu'],        ['vare']),
    (['irundu'],         ['varikum']),
    (['to'],             None),          # English "to" in between anchors, no trailing to_word
]

# replace templates: suffix after {to_anchor}; action label position varies
REPLACE_SUFFIX = [
    (['aa', 'maathu'],       ['O', 'B-ACTION']),
    (['aa', 'change', 'panu'], ['O', 'B-ACTION', 'O']),
    (['maathi', 'podu'],     ['B-ACTION', 'O']),
    (['replace', 'panu'],    ['B-ACTION', 'O']),
    (['aa', 'maathi', 'podu'], ['O', 'B-ACTION', 'O']),
]

# insert-after templates: (middle_tokens_between_to_and_from, action_tokens, action_labels)
INSERT_TEMPLATES = [
    (['ku', 'aprom'],    ['insert', 'panu'], ['B-ACTION', 'O']),
    (['ku', 'next', 'uh'], ['insert', 'panu'], ['B-ACTION', 'O']),
    (['ku', 'piragu'],   ['add', 'panu'],    ['B-ACTION', 'O']),
    (['aprom'],          ['insert', 'panu'], ['B-ACTION', 'O']),
    (['piragu'],         ['add', 'panu'],    ['B-ACTION', 'O']),
]

RANGE_ACTIONS = ['bold', 'italic', 'delete', 'underline', 'heading2', 'heading3',
                 'font-increase', 'highlight']

ACTION_ORDER = ['bold', 'italic', 'delete', 'underline', 'heading2', 'heading3',
                'replace', 'insert-after', 'font-increase', 'highlight']


def make_range_entry(entry_id, intent, from_anchor, to_anchor, t_idx, phrase_idx):
    """Build one range-based Tanglish entry."""
    fw_tokens, tw_tokens = RANGE_TEMPLATES[t_idx]
    ap_tokens, ap_labels = ACTION_PHRASES[intent][phrase_idx % len(ACTION_PHRASES[intent])]

    if tw_tokens is None:
        # {from} to {to} {action}
        tokens = [from_anchor] + fw_tokens + [to_anchor] + ap_tokens
        labels = ['B-FROM'] + ['O'] * len(fw_tokens) + ['B-TO'] + ap_labels
    else:
        tokens = [from_anchor] + fw_tokens + [to_anchor] + tw_tokens + ap_tokens
        labels = ['B-FROM'] + ['O'] * len(fw_tokens) + ['B-TO'] + ['O'] * len(tw_tokens) + ap_labels

    return {
        'id': entry_id,
        'text': ' '.join(tokens),
        'tokens': tokens,
        'labels': labels,
        'intent': intent,
        'from_anchor': from_anchor,
        'to_anchor': to_anchor,
    }


def make_replace_entry(entry_id, from_anchor, to_anchor, t_idx):
    suffix_tokens, suffix_labels = REPLACE_SUFFIX[t_idx]
    tokens = [from_anchor, 'ah', to_anchor] + suffix_tokens
    labels = ['B-FROM', 'O', 'B-TO'] + suffix_labels
    return {
        'id': entry_id,
        'text': ' '.join(tokens),
        'tokens': tokens,
        'labels': labels,
        'intent': 'replace',
        'from_anchor': from_anchor,
        'to_anchor': to_anchor,
    }


def make_insert_entry(entry_id, from_anchor, to_anchor, t_idx):
    """
    insert-after: from_anchor = what to insert (B-FROM), to_anchor = insert after here (B-TO)
    Tanglish pattern: {to_anchor} {middle} {from_anchor} {action}
    """
    middle, action_tokens, action_labels = INSERT_TEMPLATES[t_idx]
    tokens = [to_anchor] + middle + [from_anchor] + action_tokens
    labels = ['B-TO'] + ['O'] * len(middle) + ['B-FROM'] + action_labels
    return {
        'id': entry_id,
        'text': ' '.join(tokens),
        'tokens': tokens,
        'labels': labels,
        'intent': 'insert-after',
        'from_anchor': from_anchor,
        'to_anchor': to_anchor,
    }


def generate_tanglish():
    dataset = []
    entry_id = 1

    for intent in ACTION_ORDER:
        pairs = ANCHOR_PAIRS[intent]  # 10 pairs

        if intent == 'replace':
            for pair_idx, (from_a, to_a) in enumerate(pairs):
                for t_idx in range(5):
                    entry = make_replace_entry(entry_id, from_a, to_a, t_idx)
                    dataset.append(entry)
                    entry_id += 1

        elif intent == 'insert-after':
            for pair_idx, (from_a, to_a) in enumerate(pairs):
                for t_idx in range(5):
                    entry = make_insert_entry(entry_id, from_a, to_a, t_idx)
                    dataset.append(entry)
                    entry_id += 1

        else:  # range-based
            for pair_idx, (from_a, to_a) in enumerate(pairs):
                for t_idx in range(5):
                    phrase_idx = pair_idx * 5 + t_idx
                    entry = make_range_entry(entry_id, intent, from_a, to_a, t_idx, phrase_idx)
                    dataset.append(entry)
                    entry_id += 1

    return dataset


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    tanglish = generate_tanglish()
    assert len(tanglish) == 500, f"Expected 500, got {len(tanglish)}"

    # Verify label-token alignment
    for ex in tanglish:
        assert len(ex['tokens']) == len(ex['labels']), (
            f"id={ex['id']}: token/label count mismatch\n"
            f"  tokens={ex['tokens']}\n  labels={ex['labels']}"
        )

    tanglish_path = os.path.join(data_dir, 'docEdit_tanglish_500.json')
    with open(tanglish_path, 'w', encoding='utf-8') as f:
        json.dump(tanglish, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(tanglish)} Tanglish entries to {tanglish_path}")

    english_path = os.path.join(data_dir, 'docEdit500.json')
    with open(english_path, encoding='utf-8') as f:
        english = json.load(f)

    # Re-ID and tag each split
    combined = []
    for i, ex in enumerate(english, start=1):
        combined.append({**ex, 'id': i, 'lang': 'en'})
    for j, ex in enumerate(tanglish, start=len(english) + 1):
        combined.append({**ex, 'id': j, 'lang': 'tanglish'})

    combined_path = os.path.join(data_dir, 'docEdit1000.json')
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(combined)} combined entries to {combined_path}")

    # Print sample entries
    print("\n--- Sample Tanglish entries ---")
    for intent in ['bold', 'replace', 'insert-after', 'font-increase', 'highlight']:
        sample = next(ex for ex in tanglish if ex['intent'] == intent)
        print(f"[{intent}] {sample['text']}")
        print(f"  tokens: {sample['tokens']}")
        print(f"  labels: {sample['labels']}")
        print()


if __name__ == '__main__':
    main()
