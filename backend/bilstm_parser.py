"""
BiLSTM inference — loads the from-scratch BiLSTM model once and exposes predict().
"""
import os
import json
import torch
from torch import nn

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_MODEL_DIR = os.path.join(_ROOT, "models", "bilstm")

PAD_IDX = 0
UNK_IDX = 1


# ── Model architecture (identical to train_bilstm.py) ─────────────────────────
class BiLSTMJointModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_layers, num_intents, num_slot_labels, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.bilstm    = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout,
        )
        lstm_out         = hidden_dim * 2
        self.dropout     = nn.Dropout(dropout)
        self.intent_head = nn.Linear(lstm_out, num_intents)
        self.slot_head   = nn.Linear(lstm_out, num_slot_labels)

    def forward(self, token_ids, lengths):
        emb    = self.dropout(self.embedding(token_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, (hidden, _) = self.bilstm(packed)
        lstm_out, _           = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True
        )
        cls = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return (self.intent_head(self.dropout(cls)),
                self.slot_head(self.dropout(lstm_out)))


# ── Lazy-loaded singletons ─────────────────────────────────────────────────────
_model        = None
_vocab        = None
_config       = None
_intent_labels = None
_slot_labels   = None
_device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load():
    global _model, _vocab, _config, _intent_labels, _slot_labels
    if _model is not None:
        return

    with open(os.path.join(_MODEL_DIR, "vocab.json"))  as f:
        _vocab = json.load(f)
    with open(os.path.join(_MODEL_DIR, "config.json")) as f:
        _config = json.load(f)

    _intent_labels = _config["intent_labels"]
    _slot_labels   = _config["slot_labels"]

    _model = BiLSTMJointModel(
        vocab_size     = _config["vocab_size"],
        embed_dim      = _config["embed_dim"],
        hidden_dim     = _config["hidden_dim"],
        num_layers     = _config["num_layers"],
        num_intents    = _config["num_intents"],
        num_slot_labels= _config["num_slot_labels"],
        dropout        = _config["dropout"],
    ).to(_device)
    _model.load_state_dict(
        torch.load(os.path.join(_MODEL_DIR, "pytorch_model.bin"),
                   map_location=_device)
    )
    _model.eval()


# ── Public API ─────────────────────────────────────────────────────────────────
def predict(command: str) -> dict:
    """
    Run BiLSTM inference on a free-form editing command.

    Returns dict: {action, from, to, confidence, method}
    """
    _load()

    words      = command.strip().lower().split()
    token_ids  = [_vocab.get(w, UNK_IDX) for w in words]

    ids_tensor = torch.tensor([token_ids], dtype=torch.long).to(_device)
    len_tensor = torch.tensor([len(token_ids)], dtype=torch.long)

    with torch.no_grad():
        intent_logits, slot_logits = _model(ids_tensor, len_tensor)

    # Intent
    intent_probs = torch.softmax(intent_logits, dim=-1)[0]
    intent_id    = intent_logits.argmax(-1).item()
    action       = _intent_labels[intent_id]
    confidence   = round(intent_probs[intent_id].item(), 4)

    # Slots — aggregate multi-word spans
    slot_preds = slot_logits[0, :len(words), :].argmax(-1).tolist()
    from_tokens, to_tokens = [], []
    for word, slot_id in zip(words, slot_preds):
        label = _slot_labels[slot_id]
        if label in ("B-FROM", "I-FROM"):
            from_tokens.append(word)
        elif label in ("B-TO", "I-TO"):
            to_tokens.append(word)

    return {
        "action":     action,
        "from":       " ".join(from_tokens) if from_tokens else None,
        "to":         " ".join(to_tokens)   if to_tokens   else None,
        "confidence": confidence,
        "method":     "bilstm",
    }
