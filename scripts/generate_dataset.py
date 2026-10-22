import json
import random

random.seed(42)

ACTIONS = [
    "bold", "italic", "delete", "underline", "heading2", "heading3",
    "replace", "insert-after", "font-increase", "highlight",
]

# Anchor pairs per action: (from_anchor, to_anchor)
# For replace:       from=target word, to=replacement word
# For insert-after:  from=text to insert, to=anchor position
# For font-increase: from=range start, to=range end
# For highlight:     from=range start, to=range end
ANCHOR_PAIRS = {
    "bold": [
        ("introduction", "conclusion"),
        ("car", "accident"),
        ("title", "summary"),
        ("name", "address"),
        ("chapter", "end"),
        ("first", "last"),
        ("header", "footer"),
        ("question", "answer"),
        ("start", "finish"),
        ("topic", "result"),
    ],
    "italic": [
        ("note", "end"),
        ("warning", "stop"),
        ("quote", "citation"),
        ("definition", "example"),
        ("term", "description"),
        ("phrase", "period"),
        ("word", "sentence"),
        ("line", "paragraph"),
        ("clause", "section"),
        ("label", "value"),
    ],
    "delete": [
        ("paragraph", "end"),
        ("line", "stop"),
        ("sentence", "period"),
        ("word", "comma"),
        ("section", "footer"),
        ("phrase", "end"),
        ("clause", "break"),
        ("block", "boundary"),
        ("item", "close"),
        ("note", "margin"),
    ],
    "underline": [
        ("keyword", "term"),
        ("link", "reference"),
        ("date", "time"),
        ("author", "publisher"),
        ("title", "subtitle"),
        ("label", "field"),
        ("point", "conclusion"),
        ("step", "result"),
        ("entry", "value"),
        ("source", "citation"),
    ],
    "heading2": [
        ("overview", "details"),
        ("abstract", "introduction"),
        ("background", "methods"),
        ("results", "discussion"),
        ("summary", "conclusion"),
        ("section", "end"),
        ("chapter", "close"),
        ("topic", "subtopic"),
        ("main", "sub"),
        ("part", "segment"),
    ],
    "heading3": [
        ("subsection", "end"),
        ("point", "note"),
        ("item", "detail"),
        ("step", "description"),
        ("element", "component"),
        ("unit", "measure"),
        ("entry", "exit"),
        ("start", "close"),
        ("begin", "finish"),
        ("open", "shut"),
    ],
    "replace": [
        ("car", "vehicle"),
        ("accident", "incident"),
        ("introduction", "overview"),
        ("title", "heading"),
        ("conclusion", "summary"),
        ("chapter", "section"),
        ("method", "approach"),
        ("result", "outcome"),
        ("note", "remark"),
        ("bold", "strong"),
    ],
    "insert-after": [
        ("conclusion", "summary"),
        ("hello", "introduction"),
        ("note", "warning"),
        ("important", "title"),
        ("disclaimer", "header"),
        ("appendix", "section"),
        ("update", "chapter"),
        ("comment", "paragraph"),
        ("signature", "footer"),
        ("end", "conclusion"),
    ],
    "font-increase": [
        ("title", "header"),
        ("introduction", "conclusion"),
        ("abstract", "keywords"),
        ("chapter", "section"),
        ("heading", "subheading"),
        ("start", "end"),
        ("first", "last"),
        ("topic", "result"),
        ("note", "summary"),
        ("paragraph", "footer"),
    ],
    "highlight": [
        ("important", "note"),
        ("keyword", "term"),
        ("definition", "example"),
        ("title", "abstract"),
        ("warning", "stop"),
        ("introduction", "conclusion"),
        ("car", "accident"),
        ("chapter", "end"),
        ("header", "footer"),
        ("start", "finish"),
    ],
}

# ── Generic templates for the original 6 range-formatting actions ─────────────
# Each lambda takes (action, from_anchor, to_anchor) → sentence string.
TEMPLATES = [
    lambda a, f, t: f"make the text {f} to {t} {a}",
    lambda a, f, t: f"{a} from {f} to {t}",
    lambda a, f, t: f"i need {f} to {t} to be {a}",
    lambda a, f, t: f"apply {a} formatting between {f} and {t}",
    lambda a, f, t: f"can you {a} {f} to {t}",
    lambda a, f, t: f"{a} everything between {f} and {t}",
    lambda a, f, t: f"from {f} to {t} make it {a}",
    lambda a, f, t: f"please {a} the section from {f} to {t}",
    lambda a, f, t: f"set {f} to {t} as {a}",
    lambda a, f, t: f"i want the part from {f} to {t} to be {a}",
    lambda a, f, t: f"format {f} through {t} with {a}",
    lambda a, f, t: f"change {f} to {t} to {a}",
    lambda a, f, t: f"highlight from {f} to {t} in {a}",
    lambda a, f, t: f"make {f} through {t} {a}",
    lambda a, f, t: f"{a} the text starting from {f} ending at {t}",
    lambda a, f, t: f"apply {a} from {f} to {t}",
    lambda a, f, t: f"turn {f} to {t} into {a}",
    lambda a, f, t: f"i would like {f} to {t} formatted as {a}",
    lambda a, f, t: f"mark {f} to {t} as {a}",
    lambda a, f, t: f"style {f} through {t} as {a}",
    lambda a, f, t: f"make the section between {f} and {t} {a}",
    lambda a, f, t: f"can you please {a} from {f} to {t}",
    lambda a, f, t: f"go ahead and {a} {f} to {t}",
    lambda a, f, t: f"the text from {f} to {t} should be {a}",
    lambda a, f, t: f"ensure {f} to {t} is {a}",
    lambda a, f, t: f"i need you to {a} from {f} to {t}",
    lambda a, f, t: f"select {f} to {t} and make it {a}",
    lambda a, f, t: f"take the text from {f} to {t} and {a} it",
    lambda a, f, t: f"apply the {a} style to {f} through {t}",
    lambda a, f, t: f"set the formatting to {a} for {f} to {t}",
    lambda a, f, t: f"i want {a} applied between {f} and {t}",
    lambda a, f, t: f"please make everything from {f} to {t} {a}",
    lambda a, f, t: f"the region from {f} to {t} needs to be {a}",
    lambda a, f, t: f"could you {a} the part from {f} to {t}",
    lambda a, f, t: f"format the text between {f} and {t} as {a}",
    lambda a, f, t: f"{a} text from {f} until {t}",
    lambda a, f, t: f"i want to {a} from {f} to {t}",
    lambda a, f, t: f"make it {a} between {f} and {t}",
    lambda a, f, t: f"apply {a} to the range {f} to {t}",
    lambda a, f, t: f"starting at {f} ending at {t} apply {a}",
    lambda a, f, t: f"from {f} up to {t} make {a}",
    lambda a, f, t: f"i need {a} formatting on {f} to {t}",
    lambda a, f, t: f"put {a} on {f} through {t}",
    lambda a, f, t: f"give {a} style to {f} to {t}",
    lambda a, f, t: f"change the style of {f} to {t} to {a}",
    lambda a, f, t: f"make the words from {f} to {t} {a}",
    lambda a, f, t: f"italics are wrong, apply {a} from {f} to {t}",
    lambda a, f, t: f"bold is wrong, apply {a} from {f} to {t}",
    lambda a, f, t: f"update {f} to {t} with {a} formatting",
    lambda a, f, t: f"convert the text {f} to {t} into {a}",
]

# ── Dedicated templates for REPLACE (50) ─────────────────────────────────────
# f=target (to replace), t=replacement.  All use "replace" keyword for B-ACTION.
REPLACE_TEMPLATES = [
    lambda f, t: f"replace {f} with {t}",
    lambda f, t: f"please replace {f} with {t}",
    lambda f, t: f"can you replace {f} with {t}",
    lambda f, t: f"i need to replace {f} with {t}",
    lambda f, t: f"i want to replace {f} with {t}",
    lambda f, t: f"go ahead and replace {f} with {t}",
    lambda f, t: f"replace the word {f} with {t}",
    lambda f, t: f"replace the text {f} with {t}",
    lambda f, t: f"in this document replace {f} with {t}",
    lambda f, t: f"you should replace {f} with {t}",
    lambda f, t: f"i would like to replace {f} with {t}",
    lambda f, t: f"replace {f} and use {t} instead",
    lambda f, t: f"the word {f} should be replaced with {t}",
    lambda f, t: f"please replace all occurrences of {f} with {t}",
    lambda f, t: f"i need you to replace {f} with {t}",
    lambda f, t: f"replace {f} with {t} throughout the document",
    lambda f, t: f"can you please replace {f} with {t}",
    lambda f, t: f"replace {f} with the word {t}",
    lambda f, t: f"i want {f} to be replaced with {t}",
    lambda f, t: f"replace this word {f} with {t}",
    lambda f, t: f"could you replace {f} with {t}",
    lambda f, t: f"please replace the word {f} with {t}",
    lambda f, t: f"replace {f} with {t} in this text",
    lambda f, t: f"i need {f} replaced with {t}",
    lambda f, t: f"the word {f} needs to be replaced with {t}",
    lambda f, t: f"replace every instance of {f} with {t}",
    lambda f, t: f"i should replace {f} with {t}",
    lambda f, t: f"replace the term {f} with {t}",
    lambda f, t: f"find {f} and replace with {t}",
    lambda f, t: f"replace {f} with {t} now",
    lambda f, t: f"please replace {f} with the word {t}",
    lambda f, t: f"replace the occurrence of {f} with {t}",
    lambda f, t: f"i want you to replace {f} with {t}",
    lambda f, t: f"replace {f} and write {t} instead",
    lambda f, t: f"we need to replace {f} with {t}",
    lambda f, t: f"please go ahead and replace {f} with {t}",
    lambda f, t: f"i am asking you to replace {f} with {t}",
    lambda f, t: f"could you please replace {f} with {t}",
    lambda f, t: f"replace {f} with {t} in this section",
    lambda f, t: f"replace {f} with {t} here",
    lambda f, t: f"kindly replace {f} with {t}",
    lambda f, t: f"you need to replace {f} with {t}",
    lambda f, t: f"replace this {f} with {t}",
    lambda f, t: f"replace the existing {f} with {t}",
    lambda f, t: f"replace {f} with a new word {t}",
    lambda f, t: f"do replace {f} with {t}",
    lambda f, t: f"i am replacing {f} with {t}",
    lambda f, t: f"replace {f} using {t}",
    lambda f, t: f"replace {f} by {t}",
    lambda f, t: f"in the text replace {f} with {t}",
]

# ── Dedicated templates for INSERT-AFTER (50) ─────────────────────────────────
# f=text to insert, t=anchor position (insert after this word).
# Mix of "insert" (gets B-ACTION) and "add" variants.
INSERT_TEMPLATES = [
    lambda f, t: f"insert {f} after {t}",
    lambda f, t: f"add {f} after {t}",
    lambda f, t: f"insert the word {f} after {t}",
    lambda f, t: f"please insert {f} after {t}",
    lambda f, t: f"can you insert {f} after {t}",
    lambda f, t: f"i need to insert {f} after {t}",
    lambda f, t: f"i want to insert {f} after {t}",
    lambda f, t: f"go ahead and insert {f} after {t}",
    lambda f, t: f"insert {f} right after {t}",
    lambda f, t: f"please add {f} after {t}",
    lambda f, t: f"could you insert {f} after {t}",
    lambda f, t: f"insert {f} immediately after {t}",
    lambda f, t: f"i need you to insert {f} after {t}",
    lambda f, t: f"can you add {f} after {t}",
    lambda f, t: f"please insert the text {f} after {t}",
    lambda f, t: f"insert {f} following {t}",
    lambda f, t: f"add the word {f} after {t}",
    lambda f, t: f"i would like to insert {f} after {t}",
    lambda f, t: f"put {f} after {t}",
    lambda f, t: f"place {f} after {t}",
    lambda f, t: f"insert {f} just after {t}",
    lambda f, t: f"add {f} right after {t}",
    lambda f, t: f"insert {f} after the word {t}",
    lambda f, t: f"you should insert {f} after {t}",
    lambda f, t: f"i want {f} inserted after {t}",
    lambda f, t: f"please place {f} after {t}",
    lambda f, t: f"insert {f} after {t} in the document",
    lambda f, t: f"can you please insert {f} after {t}",
    lambda f, t: f"i need {f} inserted after {t}",
    lambda f, t: f"insert {f} after {t} here",
    lambda f, t: f"add {f} immediately after {t}",
    lambda f, t: f"kindly insert {f} after {t}",
    lambda f, t: f"insert the text {f} right after {t}",
    lambda f, t: f"i am asking you to insert {f} after {t}",
    lambda f, t: f"could you please insert {f} after {t}",
    lambda f, t: f"insert {f} after {t} now",
    lambda f, t: f"add {f} following {t}",
    lambda f, t: f"insert {f} after {t} in this section",
    lambda f, t: f"you need to insert {f} after {t}",
    lambda f, t: f"we need to insert {f} after {t}",
    lambda f, t: f"go insert {f} after {t}",
    lambda f, t: f"please go ahead and insert {f} after {t}",
    lambda f, t: f"insert {f} after the position of {t}",
    lambda f, t: f"i would like {f} inserted after {t}",
    lambda f, t: f"insert {f} after {t} please",
    lambda f, t: f"add {f} after the word {t}",
    lambda f, t: f"do insert {f} after {t}",
    lambda f, t: f"insert {f} following the word {t}",
    lambda f, t: f"i want you to insert {f} after {t}",
    lambda f, t: f"insert {f} after {t} in the text",
]

# ── Dedicated templates for FONT-INCREASE (50) ────────────────────────────────
# f=range start, t=range end.  All contain "increase" keyword.
FONTINC_TEMPLATES = [
    lambda f, t: f"increase font size from {f} to {t}",
    lambda f, t: f"increase the font size from {f} to {t}",
    lambda f, t: f"increase font from {f} to {t}",
    lambda f, t: f"please increase the font size from {f} to {t}",
    lambda f, t: f"can you increase the font from {f} to {t}",
    lambda f, t: f"i need to increase the font size from {f} to {t}",
    lambda f, t: f"increase the text size from {f} to {t}",
    lambda f, t: f"increase the size of the text from {f} to {t}",
    lambda f, t: f"increase font between {f} and {t}",
    lambda f, t: f"increase the font size between {f} and {t}",
    lambda f, t: f"please increase font size between {f} and {t}",
    lambda f, t: f"i want to increase the font from {f} to {t}",
    lambda f, t: f"increase the font size for {f} to {t}",
    lambda f, t: f"go ahead and increase the font from {f} to {t}",
    lambda f, t: f"increase the font on {f} through {t}",
    lambda f, t: f"can you increase the text size from {f} to {t}",
    lambda f, t: f"increase font size on the text from {f} to {t}",
    lambda f, t: f"i need you to increase the font from {f} to {t}",
    lambda f, t: f"please increase the text size from {f} to {t}",
    lambda f, t: f"increase the font size of {f} to {t}",
    lambda f, t: f"could you increase the font from {f} to {t}",
    lambda f, t: f"increase font size in the section from {f} to {t}",
    lambda f, t: f"increase the font of the text from {f} to {t}",
    lambda f, t: f"i would like to increase the font from {f} to {t}",
    lambda f, t: f"make the font bigger from {f} to {t}",
    lambda f, t: f"increase the font size from {f} through {t}",
    lambda f, t: f"increase font from {f} through {t}",
    lambda f, t: f"please increase font from {f} to {t}",
    lambda f, t: f"i need the font size increased from {f} to {t}",
    lambda f, t: f"increase text size from {f} to {t}",
    lambda f, t: f"increase the font size starting at {f} ending at {t}",
    lambda f, t: f"can you please increase the font from {f} to {t}",
    lambda f, t: f"increase font size here from {f} to {t}",
    lambda f, t: f"kindly increase the font size from {f} to {t}",
    lambda f, t: f"you should increase the font from {f} to {t}",
    lambda f, t: f"i want the font size increased from {f} to {t}",
    lambda f, t: f"increase the font in the range {f} to {t}",
    lambda f, t: f"go ahead and increase font from {f} to {t}",
    lambda f, t: f"please increase the font between {f} and {t}",
    lambda f, t: f"increase the size of text from {f} to {t}",
    lambda f, t: f"increase font size for the section {f} to {t}",
    lambda f, t: f"we need to increase the font from {f} to {t}",
    lambda f, t: f"could you please increase font size from {f} to {t}",
    lambda f, t: f"i am asking you to increase the font from {f} to {t}",
    lambda f, t: f"increase the text font from {f} to {t}",
    lambda f, t: f"increase font in the text from {f} to {t}",
    lambda f, t: f"i want you to increase the font from {f} to {t}",
    lambda f, t: f"please do increase the font from {f} to {t}",
    lambda f, t: f"you need to increase the font size from {f} to {t}",
    lambda f, t: f"increase the font size now from {f} to {t}",
]

# ── Dedicated templates for HIGHLIGHT (50) ────────────────────────────────────
# f=range start, t=range end.  All contain "highlight" keyword.
HIGHLIGHT_TEMPLATES = [
    lambda f, t: f"highlight from {f} to {t}",
    lambda f, t: f"highlight the text from {f} to {t}",
    lambda f, t: f"highlight between {f} and {t}",
    lambda f, t: f"please highlight from {f} to {t}",
    lambda f, t: f"can you highlight from {f} to {t}",
    lambda f, t: f"i need to highlight from {f} to {t}",
    lambda f, t: f"highlight the section from {f} to {t}",
    lambda f, t: f"highlight {f} through {t}",
    lambda f, t: f"highlight from {f} through {t}",
    lambda f, t: f"please highlight the text from {f} to {t}",
    lambda f, t: f"i want to highlight from {f} to {t}",
    lambda f, t: f"go ahead and highlight from {f} to {t}",
    lambda f, t: f"highlight the words from {f} to {t}",
    lambda f, t: f"can you highlight the text between {f} and {t}",
    lambda f, t: f"i need you to highlight from {f} to {t}",
    lambda f, t: f"highlight from {f} to {t} in the document",
    lambda f, t: f"could you highlight from {f} to {t}",
    lambda f, t: f"please highlight between {f} and {t}",
    lambda f, t: f"highlight the region from {f} to {t}",
    lambda f, t: f"i would like to highlight from {f} to {t}",
    lambda f, t: f"highlight everything from {f} to {t}",
    lambda f, t: f"highlight from {f} until {t}",
    lambda f, t: f"can you please highlight from {f} to {t}",
    lambda f, t: f"highlight the part from {f} to {t}",
    lambda f, t: f"i want the text from {f} to {t} highlighted",
    lambda f, t: f"highlight text starting from {f} ending at {t}",
    lambda f, t: f"please highlight the section between {f} and {t}",
    lambda f, t: f"highlight the range {f} to {t}",
    lambda f, t: f"i need the text from {f} to {t} highlighted",
    lambda f, t: f"highlight {f} to {t} now",
    lambda f, t: f"kindly highlight from {f} to {t}",
    lambda f, t: f"you should highlight from {f} to {t}",
    lambda f, t: f"i want you to highlight from {f} to {t}",
    lambda f, t: f"highlight the text between {f} and {t}",
    lambda f, t: f"go highlight from {f} to {t}",
    lambda f, t: f"please go ahead and highlight from {f} to {t}",
    lambda f, t: f"i am asking you to highlight from {f} to {t}",
    lambda f, t: f"could you please highlight from {f} to {t}",
    lambda f, t: f"highlight from {f} to {t} please",
    lambda f, t: f"we need to highlight from {f} to {t}",
    lambda f, t: f"highlight from {f} to {t} here",
    lambda f, t: f"highlight this section from {f} to {t}",
    lambda f, t: f"please highlight from {f} through {t}",
    lambda f, t: f"you need to highlight from {f} to {t}",
    lambda f, t: f"do highlight from {f} to {t}",
    lambda f, t: f"i would like {f} to {t} highlighted",
    lambda f, t: f"highlight from {f} to {t} in this text",
    lambda f, t: f"mark as highlighted from {f} to {t}",
    lambda f, t: f"highlight the content from {f} to {t}",
    lambda f, t: f"highlight all text from {f} to {t}",
]


def tokenize(text):
    return text.split()


def make_bio_labels(tokens, from_anchor, to_anchor, action_word):
    """
    BIO-tag a token list.
    action_word: the token that appears in the text to mark as B-ACTION
                 (may differ from the intent label, e.g. "insert" for "insert-after")
    """
    labels = ["O"] * len(tokens)

    def tag_span(token_list, b_tag, i_tag):
        n = len(token_list)
        for i in range(len(tokens) - n + 1):
            if tokens[i:i + n] == token_list:
                labels[i] = b_tag
                for j in range(1, n):
                    labels[i + j] = i_tag
                return True
        return False

    tag_span(action_word.split(), "B-ACTION", "I-ACTION")
    tag_span(from_anchor.split(), "B-FROM",   "I-FROM")
    tag_span(to_anchor.split(),   "B-TO",     "I-TO")
    return labels


def generate_examples():
    examples = []
    example_id = 1

    # ── Original 6 range-formatting actions (generic TEMPLATES) ──────────────
    original_actions = ["bold", "italic", "delete", "underline", "heading2", "heading3"]
    for action in original_actions:
        anchors   = ANCHOR_PAIRS[action]
        templates = TEMPLATES[:]
        random.shuffle(templates)

        for i in range(50):
            template     = templates[i % len(templates)]
            from_anchor, to_anchor = anchors[i % len(anchors)]
            text   = template(action, from_anchor, to_anchor)
            tokens = tokenize(text)
            labels = make_bio_labels(tokens, from_anchor, to_anchor, action_word=action)

            examples.append({
                "id":          example_id,
                "text":        text,
                "tokens":      tokens,
                "labels":      labels,
                "intent":      action,
                "from_anchor": from_anchor,
                "to_anchor":   to_anchor,
            })
            example_id += 1

    # ── REPLACE (dedicated templates, action_word="replace") ─────────────────
    replace_templates = REPLACE_TEMPLATES[:]
    random.shuffle(replace_templates)
    for i in range(50):
        tmpl       = replace_templates[i % len(replace_templates)]
        from_a, to_a = ANCHOR_PAIRS["replace"][i % len(ANCHOR_PAIRS["replace"])]
        text   = tmpl(from_a, to_a)
        tokens = tokenize(text)
        labels = make_bio_labels(tokens, from_a, to_a, action_word="replace")
        examples.append({
            "id": example_id, "text": text, "tokens": tokens,
            "labels": labels, "intent": "replace",
            "from_anchor": from_a, "to_anchor": to_a,
        })
        example_id += 1

    # ── INSERT-AFTER (action_word="insert" — matches most templates) ──────────
    insert_templates = INSERT_TEMPLATES[:]
    random.shuffle(insert_templates)
    for i in range(50):
        tmpl       = insert_templates[i % len(insert_templates)]
        from_a, to_a = ANCHOR_PAIRS["insert-after"][i % len(ANCHOR_PAIRS["insert-after"])]
        text   = tmpl(from_a, to_a)
        tokens = tokenize(text)
        labels = make_bio_labels(tokens, from_a, to_a, action_word="insert")
        examples.append({
            "id": example_id, "text": text, "tokens": tokens,
            "labels": labels, "intent": "insert-after",
            "from_anchor": from_a, "to_anchor": to_a,
        })
        example_id += 1

    # ── FONT-INCREASE (action_word="increase") ────────────────────────────────
    fontinc_templates = FONTINC_TEMPLATES[:]
    random.shuffle(fontinc_templates)
    for i in range(50):
        tmpl       = fontinc_templates[i % len(fontinc_templates)]
        from_a, to_a = ANCHOR_PAIRS["font-increase"][i % len(ANCHOR_PAIRS["font-increase"])]
        text   = tmpl(from_a, to_a)
        tokens = tokenize(text)
        labels = make_bio_labels(tokens, from_a, to_a, action_word="increase")
        examples.append({
            "id": example_id, "text": text, "tokens": tokens,
            "labels": labels, "intent": "font-increase",
            "from_anchor": from_a, "to_anchor": to_a,
        })
        example_id += 1

    # ── HIGHLIGHT (action_word="highlight") ───────────────────────────────────
    highlight_templates = HIGHLIGHT_TEMPLATES[:]
    random.shuffle(highlight_templates)
    for i in range(50):
        tmpl       = highlight_templates[i % len(highlight_templates)]
        from_a, to_a = ANCHOR_PAIRS["highlight"][i % len(ANCHOR_PAIRS["highlight"])]
        text   = tmpl(from_a, to_a)
        tokens = tokenize(text)
        labels = make_bio_labels(tokens, from_a, to_a, action_word="highlight")
        examples.append({
            "id": example_id, "text": text, "tokens": tokens,
            "labels": labels, "intent": "highlight",
            "from_anchor": from_a, "to_anchor": to_a,
        })
        example_id += 1

    return examples


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    examples = generate_examples()

    out_path = "data/docEdit500.json"
    with open(out_path, "w") as f:
        json.dump(examples, f, indent=2)

    print(f"Generated {len(examples)} examples. Saved to {out_path}")

    action_counts = {}
    for ex in examples:
        action_counts[ex["intent"]] = action_counts.get(ex["intent"], 0) + 1
    print("Per-action counts:", action_counts)

    sample = examples[300]  # first new-action example
    print("\nSample new-action entry:")
    print(f"  text:   {sample['text']}")
    print(f"  tokens: {sample['tokens']}")
    print(f"  labels: {sample['labels']}")
    print(f"  intent: {sample['intent']}")
