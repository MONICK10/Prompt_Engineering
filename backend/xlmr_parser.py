"""
XLM-RoBERTa baselines for Tanglish evaluation.

  predict_zero(command)   — zero-shot: xlm-roberta-base encoder + random heads
  predict_tuned(command)  — fine-tuned on docEdit_tanglish_500.json
                            (errors gracefully if model not yet trained)
"""
import os
import json
import torch
from torch import nn
from transformers import XLMRobertaModel, XLMRobertaTokenizerFast

_ROOT      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_MAPS_FILE = os.path.join(_ROOT, "models", "doceditbert", "label_maps.json")

with open(_MAPS_FILE) as _f:
    _maps = json.load(_f)

INTENT_LABELS = _maps["intent_labels"]
SLOT_LABELS   = _maps["slot_labels"]
NUM_INTENTS   = len(INTENT_LABELS)
NUM_SLOTS     = len(SLOT_LABELS)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Shared architecture (mirrors DocEditBERT, XLM-R backbone) ─────────────────
class XLMRDocEdit(nn.Module):
    def __init__(self):
        super().__init__()
        self.xlmr        = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        hidden           = self.xlmr.config.hidden_size   # 768
        self.dropout     = nn.Dropout(0.1)
        self.intent_head = nn.Linear(hidden, NUM_INTENTS)
        self.slot_head   = nn.Linear(hidden, NUM_SLOTS)

    def forward(self, input_ids, attention_mask):
        out     = self.xlmr(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = out.last_hidden_state
        cls_out = seq_out[:, 0, :]
        return (self.intent_head(self.dropout(cls_out)),
                self.slot_head(self.dropout(seq_out)))


# ── Lazy-loaded singletons ────────────────────────────────────────────────────
_zero_model = _zero_tok = None


def _load_zero():
    global _zero_model, _zero_tok
    if _zero_model is not None:
        return
    print("[xlmr] Loading xlm-roberta-base (zero-shot)…", flush=True)
    _zero_tok   = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
    _zero_model = XLMRDocEdit().to(_device)
    _zero_model.eval()   # random classification heads — no training



# ── Core inference ─────────────────────────────────────────────────────────────
def _infer(model, tokenizer, command: str) -> dict:
    words    = command.strip().split()
    encoding = tokenizer(
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
        intent_logits, slot_logits = model(input_ids, attention_mask)

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
    }


# ── Public API ─────────────────────────────────────────────────────────────────
def predict_zero(command: str) -> dict:
    _load_zero()
    r = _infer(_zero_model, _zero_tok, command)
    r["method"] = "xlmr-zero"
    return r


