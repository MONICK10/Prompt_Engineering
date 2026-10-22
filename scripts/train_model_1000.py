"""
Retrain DocEditBERT on docEdit1000.json (500 English + 500 Tanglish).
Train/val split: 800 / 200.
Reports intent accuracy and slot F1 separately for English and Tanglish examples
on the validation set.
"""
import json
import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

INTENT_LABELS = [
    "bold", "italic", "delete", "underline", "heading2", "heading3",
    "replace", "insert-after", "font-increase", "highlight",
]
SLOT_LABELS = [
    "O",
    "B-ACTION", "I-ACTION",
    "B-FROM",   "I-FROM",
    "B-TO",     "I-TO",
    "B-ANCHOR", "I-ANCHOR",
]

INTENT2ID    = {l: i for i, l in enumerate(INTENT_LABELS)}
SLOT2ID      = {l: i for i, l in enumerate(SLOT_LABELS)}
ID2INTENT    = {i: l for l, i in INTENT2ID.items()}
ID2SLOT      = {i: l for l, i in SLOT2ID.items()}
NUM_INTENTS  = len(INTENT_LABELS)
NUM_SLOTS    = len(SLOT_LABELS)
IGNORE_INDEX = -100


class DocEditDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=64):
        self.examples   = examples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex          = self.examples[idx]
        word_labels = [SLOT2ID[l] for l in ex["labels"]]
        intent_id   = INTENT2ID[ex["intent"]]

        encoding = self.tokenizer(
            ex["tokens"],
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        word_ids       = encoding.word_ids()
        aligned_labels = []
        prev_word_id   = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(IGNORE_INDEX)
            elif wid != prev_word_id:
                aligned_labels.append(word_labels[wid])
            else:
                aligned_labels.append(IGNORE_INDEX)
            prev_word_id = wid

        return {
            "input_ids":      torch.tensor(encoding["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"],  dtype=torch.long),
            "intent_label":   torch.tensor(intent_id,                  dtype=torch.long),
            "slot_labels":    torch.tensor(aligned_labels,             dtype=torch.long),
            "lang":           ex.get("lang", "en"),
        }


class DocEditBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.distilbert  = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden           = self.distilbert.config.dim
        self.dropout     = nn.Dropout(0.1)
        self.intent_head = nn.Linear(hidden, NUM_INTENTS)
        self.slot_head   = nn.Linear(hidden, NUM_SLOTS)

    def forward(self, input_ids, attention_mask):
        out           = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out       = out.last_hidden_state
        cls_out       = seq_out[:, 0, :]
        intent_logits = self.intent_head(self.dropout(cls_out))
        slot_logits   = self.slot_head(self.dropout(seq_out))
        return intent_logits, slot_logits


def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels):
    intent_acc = accuracy_score(intent_labels, intent_preds)

    flat_preds, flat_labels = [], []
    for ps, ls in zip(slot_preds, slot_labels):
        for p, l in zip(ps, ls):
            if l != IGNORE_INDEX:
                flat_preds.append(p)
                flat_labels.append(l)

    present = sorted(set(flat_labels))
    non_o   = [i for i in present if SLOT_LABELS[i] != "O"]
    slot_f1 = f1_score(flat_labels, flat_preds,
                       labels=non_o, average="macro", zero_division=0) if non_o else 0.0
    return intent_acc, slot_f1


def evaluate_split(model, examples, tokenizer, intent_loss_fn, slot_loss_fn, device, batch_size=16):
    """Evaluate on a list of examples; returns (avg_loss, intent_acc, slot_f1)."""
    dataset = DocEditDataset(examples, tokenizer)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    total_loss = 0.0
    all_ip, all_il, all_sp, all_sl = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            iids  = batch["input_ids"].to(device)
            amask = batch["attention_mask"].to(device)
            il    = batch["intent_label"].to(device)
            sl    = batch["slot_labels"].to(device)

            logits_i, logits_s = model(iids, amask)
            loss = intent_loss_fn(logits_i, il) + slot_loss_fn(
                logits_s.view(-1, NUM_SLOTS), sl.view(-1))
            total_loss += loss.item()

            all_ip.extend(logits_i.argmax(-1).cpu().tolist())
            all_il.extend(il.cpu().tolist())
            all_sp.extend(logits_s.argmax(-1).cpu().tolist())
            all_sl.extend(sl.cpu().tolist())

    avg_loss                = total_loss / len(loader)
    intent_acc, slot_f1    = compute_metrics(all_ip, all_il, all_sp, all_sl)
    return avg_loss, intent_acc, slot_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "docEdit1000.json")
    with open(data_path, encoding="utf-8") as f:
        examples = json.load(f)

    # Stratified 800/200 split
    train_ex, val_ex = train_test_split(
        examples, test_size=200, random_state=SEED,
        stratify=[ex["intent"] for ex in examples],
    )
    print(f"Train: {len(train_ex)}  |  Val: {len(val_ex)}")

    # Language breakdown in val
    val_en  = [ex for ex in val_ex if ex.get("lang", "en") == "en"]
    val_tan = [ex for ex in val_ex if ex.get("lang") == "tanglish"]
    print(f"Val English: {len(val_en)}  |  Val Tanglish: {len(val_tan)}")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    train_dataset = DocEditDataset(train_ex, tokenizer)
    train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = DocEditBERT().to(device)

    intent_loss_fn = nn.CrossEntropyLoss()
    slot_loss_fn   = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    EPOCHS      = 15
    LR          = 2e-5
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = max(1, total_steps // 10)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    save_dir = os.path.join(os.path.dirname(__file__), "..", "models", "doceditbert_1000")
    os.makedirs(save_dir, exist_ok=True)

    best_slot_f1 = 0.0
    best_epoch   = 0

    print("\n" + "=" * 70)
    print(f"{'Ep':>3}  {'TrLoss':>8}  {'VLoss':>7}  {'IntAcc':>8}  {'SlotF1':>7}  {'Best?':>6}")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            iids  = batch["input_ids"].to(device)
            amask = batch["attention_mask"].to(device)
            il    = batch["intent_label"].to(device)
            sl    = batch["slot_labels"].to(device)

            optimizer.zero_grad()
            logits_i, logits_s = model(iids, amask)
            loss = intent_loss_fn(logits_i, il) + slot_loss_fn(
                logits_s.view(-1, NUM_SLOTS), sl.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_loader)
        val_loss, intent_acc, slot_f1 = evaluate_split(
            model, val_ex, tokenizer, intent_loss_fn, slot_loss_fn, device)

        is_best = slot_f1 > best_slot_f1
        if is_best:
            best_slot_f1 = slot_f1
            best_epoch   = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
            tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
            with open(os.path.join(save_dir, "label_maps.json"), "w") as f:
                json.dump({"intent_labels": INTENT_LABELS, "slot_labels": SLOT_LABELS,
                           "intent2id": INTENT2ID, "slot2id": SLOT2ID}, f, indent=2)

        print(f"{epoch:>3}  {avg_train:>8.4f}  {val_loss:>7.4f}  "
              f"{intent_acc:>8.4f}  {slot_f1:>7.4f}  "
              f"{'<-- BEST' if is_best else ''}", flush=True)

    print("=" * 70)
    print(f"\nBest slot F1: {best_slot_f1:.4f}  (epoch {best_epoch})")

    # Reload best checkpoint for final per-language evaluation
    model.load_state_dict(torch.load(
        os.path.join(save_dir, "pytorch_model.bin"), map_location=device))

    print("\n" + "=" * 70)
    print("FINAL VALIDATION METRICS (best checkpoint)")
    print("=" * 70)

    _, acc_all, f1_all = evaluate_split(
        model, val_ex, tokenizer, intent_loss_fn, slot_loss_fn, device)
    print(f"  All val      intent_acc={acc_all:.4f}  slot_f1={f1_all:.4f}")

    if val_en:
        _, acc_en, f1_en = evaluate_split(
            model, val_en, tokenizer, intent_loss_fn, slot_loss_fn, device)
        print(f"  English      intent_acc={acc_en:.4f}  slot_f1={f1_en:.4f}  (n={len(val_en)})")

    if val_tan:
        _, acc_tan, f1_tan = evaluate_split(
            model, val_tan, tokenizer, intent_loss_fn, slot_loss_fn, device)
        print(f"  Tanglish     intent_acc={acc_tan:.4f}  slot_f1={f1_tan:.4f}  (n={len(val_tan)})")

    print(f"\nModel saved to: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    main()
