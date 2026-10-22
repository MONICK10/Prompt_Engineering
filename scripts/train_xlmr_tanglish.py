"""
Baseline B: Fine-tune XLM-RoBERTa on docEdit_tanglish_500.json.

  - 15 epochs, AdamW lr=2e-5, batch 16
  - Early stopping (patience 5) on val slot F1
  - Saves to models/xlmr_tanglish/
    ├── pytorch_model.bin
    └── tokenizer/
"""
import os
import sys
import json
import math
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    XLMRobertaModel,
    XLMRobertaTokenizerFast,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score as seq_f1

ROOT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_F   = os.path.join(ROOT, "data",   "docEdit_tanglish_500.json")
MAPS_F   = os.path.join(ROOT, "models", "doceditbert", "label_maps.json")
SAVE_DIR = os.path.join(ROOT, "models", "xlmr_tanglish")

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Label maps (shared with DocEditBERT) ─────────────────────────────────────
with open(MAPS_F) as f:
    maps = json.load(f)

INTENT_LABELS = maps["intent_labels"]
SLOT_LABELS   = maps["slot_labels"]
NUM_INTENTS   = len(INTENT_LABELS)
NUM_SLOTS     = len(SLOT_LABELS)
intent2id     = {l: i for i, l in enumerate(INTENT_LABELS)}
slot2id       = {l: i for i, l in enumerate(SLOT_LABELS)}

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {_device}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class TanglishDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples  = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex     = self.examples[idx]
        words  = ex["tokens"]
        labels = ex["labels"]

        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        word_ids   = enc.word_ids()
        slot_ids   = []
        prev_wid   = None
        for wid in word_ids:
            if wid is None:
                slot_ids.append(-100)
            elif wid == prev_wid:
                # continue BIO: B- → I- for continuations, I- stays I-
                prev_label = labels[wid]
                if prev_label.startswith("B-"):
                    cont = "I-" + prev_label[2:]
                    slot_ids.append(slot2id.get(cont, slot2id.get(prev_label, 0)))
                else:
                    slot_ids.append(slot2id.get(prev_label, 0))
            else:
                slot_ids.append(slot2id.get(labels[wid], 0))
            prev_wid = wid

        intent_id = intent2id.get(ex["intent"], 0)

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "intent_label":   torch.tensor(intent_id, dtype=torch.long),
            "slot_labels":    torch.tensor(slot_ids,  dtype=torch.long),
        }


# ── Model (same as xlmr_parser.XLMRDocEdit) ──────────────────────────────────
class XLMRDocEdit(nn.Module):
    def __init__(self):
        super().__init__()
        self.xlmr        = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        hidden           = self.xlmr.config.hidden_size
        self.dropout     = nn.Dropout(0.1)
        self.intent_head = nn.Linear(hidden, NUM_INTENTS)
        self.slot_head   = nn.Linear(hidden, NUM_SLOTS)

    def forward(self, input_ids, attention_mask):
        out     = self.xlmr(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = out.last_hidden_state
        cls_out = seq_out[:, 0, :]
        return (self.intent_head(self.dropout(cls_out)),
                self.slot_head(self.dropout(seq_out)))


# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate(model, loader, tokenizer):
    model.eval()
    intent_correct = 0
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in loader:
            iids  = batch["input_ids"].to(_device)
            amask = batch["attention_mask"].to(_device)
            ilabs = batch["intent_label"].to(_device)
            slabs = batch["slot_labels"].to(_device)

            i_logits, s_logits = model(iids, amask)
            intent_correct += (i_logits.argmax(-1) == ilabs).sum().item()

            for b in range(iids.size(0)):
                t_seq, p_seq = [], []
                for pos in range(slabs.size(1)):
                    true_id = slabs[b, pos].item()
                    pred_id = s_logits[b, pos].argmax(-1).item()
                    if true_id == -100:
                        continue
                    t_seq.append(SLOT_LABELS[true_id])
                    p_seq.append(SLOT_LABELS[pred_id])
                all_true.append(t_seq)
                all_pred.append(p_seq)

    n = sum(len(loader.dataset) for loader in [loader])
    n = len(loader.dataset)
    return {
        "intent_acc": intent_correct / n,
        "slot_f1":    seq_f1(all_true, all_pred, zero_division=0),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    with open(DATA_F) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} Tanglish examples")

    intents = [ex["intent"] for ex in data]
    train_ex, val_ex = train_test_split(
        data, test_size=0.2, random_state=42, stratify=intents
    )
    print(f"Train: {len(train_ex)}  |  Val: {len(val_ex)}")

    tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")

    train_ds = TanglishDataset(train_ex, tokenizer)
    val_ds   = TanglishDataset(val_ex,   tokenizer)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32)

    model = XLMRDocEdit().to(_device)

    EPOCHS     = 15
    LR         = 2e-5
    PATIENCE   = 5
    WARMUP     = 0.1

    total_steps   = len(train_loader) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    intent_loss_fn = nn.CrossEntropyLoss()
    slot_loss_fn   = nn.CrossEntropyLoss(ignore_index=-100)

    best_f1    = 0.0
    best_state = None
    no_improve = 0

    print()
    print("=" * 70)
    print(f" {'Ep':>4}  {'TrLoss':>8}  {'VLoss':>8}  {'IntAcc':>8}  {'SlotF1':>8}")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0

        for batch in train_loader:
            iids  = batch["input_ids"].to(_device)
            amask = batch["attention_mask"].to(_device)
            ilabs = batch["intent_label"].to(_device)
            slabs = batch["slot_labels"].to(_device)

            optimizer.zero_grad()
            i_logits, s_logits = model(iids, amask)

            loss = (intent_loss_fn(i_logits, ilabs) +
                    slot_loss_fn(s_logits.view(-1, NUM_SLOTS), slabs.view(-1)))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()

        tr_loss /= len(train_loader)

        # Validation loss
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                iids  = batch["input_ids"].to(_device)
                amask = batch["attention_mask"].to(_device)
                ilabs = batch["intent_label"].to(_device)
                slabs = batch["slot_labels"].to(_device)
                i_logits, s_logits = model(iids, amask)
                v_loss += (intent_loss_fn(i_logits, ilabs) +
                           slot_loss_fn(s_logits.view(-1, NUM_SLOTS), slabs.view(-1))).item()
        v_loss /= len(val_loader)

        metrics = evaluate(model, val_loader, tokenizer)
        improved = metrics["slot_f1"] > best_f1

        tag = " <-- BEST" if improved else ""
        print(f" {epoch:>4}  {tr_loss:>8.4f}  {v_loss:>8.4f}  "
              f"{metrics['intent_acc']:>8.4f}  {metrics['slot_f1']:>8.4f}{tag}")

        if improved:
            best_f1    = metrics["slot_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
                break

    print("=" * 70)
    print(f"\nBest slot F1: {best_f1:.4f}")

    # ── Save best checkpoint ──────────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "pytorch_model.bin"))
    tokenizer.save_pretrained(os.path.join(SAVE_DIR, "tokenizer"))

    # ── Final evaluation ──────────────────────────────────────────────────────
    final = evaluate(model, val_loader, tokenizer)
    print(f"\nFinal val  intent_acc={final['intent_acc']:.4f}  slot_f1={final['slot_f1']:.4f}  (n={len(val_ds)})")
    print(f"\nModel saved to: {os.path.abspath(SAVE_DIR)}")


if __name__ == "__main__":
    main()
