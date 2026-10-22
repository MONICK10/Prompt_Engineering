"""
DocEditBERT inference — loads the fine-tuned model once and exposes predict().
"""
import os
import json
import torch
from torch import nn
from transformers import DistilBertTokenizerFast, DistilBertModel

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_MODEL_DIR = os.path.join(_ROOT, "models", "doceditbert_1000")

# ── Label constants (must match training) ─────────────────────────────────────
with open(os.path.join(_MODEL_DIR, "label_maps.json")) as _f:
    _maps = json.load(_f)

INTENT_LABELS = _maps["intent_labels"]   # ["bold", "italic", ...]
SLOT_LABELS   = _maps["slot_labels"]     # ["O", "B-ACTION", ..., "B-TO", ...]
NUM_INTENTS   = len(INTENT_LABELS)
NUM_SLOTS     = len(SLOT_LABELS)


# ── Model architecture (identical to train_model.py) ─────────────────────────
class DocEditBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.distilbert  = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden           = self.distilbert.config.dim
        self.dropout     = nn.Dropout(0.1)
        self.intent_head = nn.Linear(hidden, NUM_INTENTS)
        self.slot_head   = nn.Linear(hidden, NUM_SLOTS)

    def forward(self, input_ids, attention_mask):
        out     = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = out.last_hidden_state
        cls_out = seq_out[:, 0, :]
        return (self.intent_head(self.dropout(cls_out)),
                self.slot_head(self.dropout(seq_out)))


# ── Lazy-loaded singletons ─────────────────────────────────────────────────────
_model     = None
_tokenizer = None
_device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load():
    global _model, _tokenizer
    if _model is not None:
        return
    _tokenizer = DistilBertTokenizerFast.from_pretrained(
        os.path.join(_MODEL_DIR, "tokenizer")
    )
    _model = DocEditBERT().to(_device)
    _model.load_state_dict(
        torch.load(os.path.join(_MODEL_DIR, "pytorch_model.bin"),
                   map_location=_device)
    )
    _model.eval()


# ── Public API ─────────────────────────────────────────────────────────────────
def predict(command: str) -> dict:
    """
    Run DocEditBERT inference on a free-form editing command.

    Returns dict: {action, from, to, confidence, method}
    """
    _load()

    words    = command.strip().split()
    encoding = _tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        max_length=64,
        padding="max_length",
        truncation=True,
    )
    input_ids      = encoding["input_ids"].to(_device)
    attention_mask = encoding["attention_mask"].to(_device)

    with torch.no_grad():
        intent_logits, slot_logits = _model(input_ids, attention_mask)

    # Intent
    intent_probs = torch.softmax(intent_logits, dim=-1)[0]
    intent_id    = intent_logits.argmax(-1).item()
    action       = INTENT_LABELS[intent_id]
    confidence   = round(intent_probs[intent_id].item(), 4)

    # Slots — aggregate multi-word spans via B-/I- tags
    word_ids   = encoding.word_ids()
    slot_preds = slot_logits[0].argmax(-1).tolist()

    from_tokens, to_tokens = [], []
    prev_wid = None
    for wid, slot_id in zip(word_ids, slot_preds):
        if wid is None or wid == prev_wid:
            prev_wid = wid
            continue
        label = SLOT_LABELS[slot_id]
        if label in ("B-FROM", "I-FROM"):
            from_tokens.append(words[wid])
        elif label in ("B-TO", "I-TO"):
            to_tokens.append(words[wid])
        prev_wid = wid

    return {
        "action":     action,
        "from":       " ".join(from_tokens) if from_tokens else None,
        "to":         " ".join(to_tokens)   if to_tokens   else None,
        "confidence": confidence,
        "method":     "doceditbert",
    }
