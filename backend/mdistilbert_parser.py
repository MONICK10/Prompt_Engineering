"""
mDistilBERT (distilbert-base-multilingual-cased) fine-tuned on docEdit_tanglish_500.json.
Lazy-loaded on first call to predict().
"""
import os
import json
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

_ROOT      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_MAPS_FILE = os.path.join(_ROOT, "models", "doceditbert", "label_maps.json")
_MODEL_DIR = os.path.join(_ROOT, "models", "mdistilbert_tanglish")

with open(_MAPS_FILE) as _f:
    _maps = json.load(_f)

INTENT_LABELS = _maps["intent_labels"]
SLOT_LABELS   = _maps["slot_labels"]
NUM_INTENTS   = len(INTENT_LABELS)
NUM_SLOTS     = len(SLOT_LABELS)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MDistilBERTDocEdit(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone    = backbone
        hidden           = backbone.config.hidden_size
        self.dropout     = nn.Dropout(0.1)
        self.intent_head = nn.Linear(hidden, NUM_INTENTS)
        self.slot_head   = nn.Linear(hidden, NUM_SLOTS)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out     = self.backbone(**kwargs)
        seq_out = out.last_hidden_state
        cls_out = seq_out[:, 0, :]
        return (self.intent_head(self.dropout(cls_out)),
                self.slot_head(self.dropout(seq_out)))


_model     = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is not None:
        return
    weights = os.path.join(_MODEL_DIR, "pytorch_model.bin")
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"mDistilBERT model not found at {weights}. "
            "Run scripts/train_indicbert_tanglish.py first."
        )
    print("[mdistilbert] Loading fine-tuned mDistilBERT...", flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(os.path.join(_MODEL_DIR, "tokenizer"))
    backbone   = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
    _model     = MDistilBERTDocEdit(backbone).to(_device)
    _model.load_state_dict(torch.load(weights, map_location=_device))
    _model.eval()


def predict(command: str) -> dict:
    try:
        _load()
    except FileNotFoundError as e:
        return {"action": None, "from": None, "to": None,
                "confidence": None, "method": "mdistilbert", "error": str(e)}

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
    token_type_ids = encoding.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(_device)

    with torch.no_grad():
        intent_logits, slot_logits = _model(input_ids, attention_mask, token_type_ids)

    intent_probs = torch.softmax(intent_logits, dim=-1)[0]
    intent_id    = intent_logits.argmax(-1).item()
    confidence   = round(intent_probs[intent_id].item(), 4)

    word_ids   = encoding.word_ids()
    slot_preds = slot_logits[0].argmax(-1).tolist()

    from_tokens, to_tokens = [], []
    prev_wid = None
    for wid, sid in zip(word_ids, slot_preds):
        if wid is None or wid == prev_wid:
            prev_wid = wid
            continue
        label = SLOT_LABELS[sid]
        if label in ("B-FROM", "I-FROM"):
            from_tokens.append(words[wid])
        elif label in ("B-TO", "I-TO"):
            to_tokens.append(words[wid])
        prev_wid = wid

    return {
        "action":     INTENT_LABELS[intent_id],
        "from":       " ".join(from_tokens) or None,
        "to":         " ".join(to_tokens)   or None,
        "confidence": confidence,
        "method":     "mdistilbert",
    }
