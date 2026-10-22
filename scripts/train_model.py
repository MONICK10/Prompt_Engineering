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

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Label maps ───────────────────────────────────────────────────────────────
INTENT_LABELS = [
    "bold", "italic", "delete", "underline", "heading2", "heading3",
    "replace", "insert-after", "font-increase", "highlight",
]
# 9-class slot set as specified in the brief
SLOT_LABELS = [
    "O",
    "B-ACTION", "I-ACTION",
    "B-FROM",   "I-FROM",
    "B-TO",     "I-TO",
    "B-ANCHOR", "I-ANCHOR",
]

INTENT2ID = {l: i for i, l in enumerate(INTENT_LABELS)}
SLOT2ID   = {l: i for i, l in enumerate(SLOT_LABELS)}
ID2INTENT = {i: l for l, i in INTENT2ID.items()}
ID2SLOT   = {i: l for l, i in SLOT2ID.items()}

NUM_INTENTS    = len(INTENT_LABELS)   # 6
NUM_SLOT_LABELS = len(SLOT_LABELS)    # 9

IGNORE_INDEX = -100  # ignored by CrossEntropyLoss for subword continuations

# ── Dataset ──────────────────────────────────────────────────────────────────
class DocEditDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=64):
        self.examples   = examples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex          = self.examples[idx]
        word_tokens = ex["tokens"]
        word_labels = [SLOT2ID[l] for l in ex["labels"]]
        intent_id   = INTENT2ID[ex["intent"]]

        encoding = self.tokenizer(
            word_tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        # Align word-level BIO labels to subword token positions.
        # Only the first subword of each word keeps its label; the rest get IGNORE_INDEX
        # so they don't contribute to the loss or metrics.
        word_ids       = encoding.word_ids()
        aligned_labels = []
        prev_word_id   = None
        for wid in word_ids:
            if wid is None:                     # [CLS] / [SEP] / [PAD]
                aligned_labels.append(IGNORE_INDEX)
            elif wid != prev_word_id:           # first subword of a word
                aligned_labels.append(word_labels[wid])
            else:                               # continuation subword
                aligned_labels.append(IGNORE_INDEX)
            prev_word_id = wid

        return {
            "input_ids":      torch.tensor(encoding["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"],  dtype=torch.long),
            "intent_label":   torch.tensor(intent_id,                  dtype=torch.long),
            "slot_labels":    torch.tensor(aligned_labels,             dtype=torch.long),
        }


# ── Model ────────────────────────────────────────────────────────────────────
class DocEditBERT(nn.Module):
    """DistilBERT + intent classification head + slot labelling head."""

    def __init__(self):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden = self.distilbert.config.dim   # 768
        self.dropout     = nn.Dropout(0.1)
        self.intent_head = nn.Linear(hidden, NUM_INTENTS)
        self.slot_head   = nn.Linear(hidden, NUM_SLOT_LABELS)

    def forward(self, input_ids, attention_mask):
        out     = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = out.last_hidden_state          # (B, T, H)
        cls_out = seq_out[:, 0, :]              # [CLS] → intent

        intent_logits = self.intent_head(self.dropout(cls_out))    # (B, 6)
        slot_logits   = self.slot_head(self.dropout(seq_out))      # (B, T, 9)
        return intent_logits, slot_logits


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(all_intent_preds, all_intent_labels,
                    all_slot_preds,   all_slot_labels):
    """
    Returns:
        intent_acc  – fraction of correct intent predictions
        slot_f1     – macro-averaged token-level F1 over all 9 slot classes
        per_slot_f1 – dict mapping each slot label to its F1 score
    """
    intent_acc = accuracy_score(all_intent_labels, all_intent_preds)

    # Flatten slot predictions; skip IGNORE_INDEX positions
    flat_preds, flat_labels = [], []
    for preds_seq, labels_seq in zip(all_slot_preds, all_slot_labels):
        for p, l in zip(preds_seq, labels_seq):
            if l != IGNORE_INDEX:
                flat_preds.append(p)
                flat_labels.append(l)

    # Macro F1 over slot classes that actually appear in ground truth, excluding O.
    # Standard NLU practice: don't penalise for zero-support classes that never
    # occur in the dataset (e.g. I-* when all anchors are single words).
    present_in_gt = sorted(set(flat_labels))
    slot_classes  = [i for i in present_in_gt if SLOT_LABELS[i] != "O"]

    slot_f1 = f1_score(flat_labels, flat_preds,
                       labels=slot_classes, average="macro", zero_division=0)

    # Per-class F1 only for classes present in ground truth
    per_class = f1_score(flat_labels, flat_preds,
                         labels=present_in_gt, average=None, zero_division=0)
    per_slot_f1 = {SLOT_LABELS[present_in_gt[i]]: round(float(per_class[i]), 4)
                   for i in range(len(present_in_gt))}

    return intent_acc, slot_f1, per_slot_f1


# ── Evaluate one split ────────────────────────────────────────────────────────
def evaluate(model, loader, intent_loss_fn, slot_loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_intent_preds, all_intent_labels = [], []
    all_slot_preds,   all_slot_labels   = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_labels  = batch["intent_label"].to(device)
            slot_labels    = batch["slot_labels"].to(device)

            intent_logits, slot_logits = model(input_ids, attention_mask)

            loss_i = intent_loss_fn(intent_logits, intent_labels)
            # slot_logits: (B, T, 9) → (B*T, 9);  slot_labels: (B, T) → (B*T,)
            loss_s = slot_loss_fn(
                slot_logits.view(-1, NUM_SLOT_LABELS),
                slot_labels.view(-1),
            )
            total_loss += (loss_i + loss_s).item()

            intent_preds = intent_logits.argmax(dim=-1).cpu().tolist()
            slot_preds   = slot_logits.argmax(dim=-1).cpu().tolist()

            all_intent_preds.extend(intent_preds)
            all_intent_labels.extend(intent_labels.cpu().tolist())
            all_slot_preds.extend(slot_preds)
            all_slot_labels.extend(slot_labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    intent_acc, slot_f1, per_slot_f1 = compute_metrics(
        all_intent_preds, all_intent_labels,
        all_slot_preds,   all_slot_labels,
    )
    return avg_loss, intent_acc, slot_f1, per_slot_f1


# ── Main training routine ─────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load dataset ─────────────────────────────────────────────────────────
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "docEdit500.json")
    with open(data_path) as f:
        examples = json.load(f)

    train_examples, val_examples = train_test_split(
        examples, test_size=0.2, random_state=SEED,
        stratify=[ex["intent"] for ex in examples],
    )
    print(f"Train: {len(train_examples)}  |  Val: {len(val_examples)}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    train_dataset = DocEditDataset(train_examples, tokenizer)
    val_dataset   = DocEditDataset(val_examples,   tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DocEditBERT().to(device)

    # ── Losses ────────────────────────────────────────────────────────────────
    intent_loss_fn = nn.CrossEntropyLoss()
    slot_loss_fn   = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    EPOCHS     = 15
    LR         = 2e-5
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = max(1, total_steps // 10)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_slot_f1   = 0.0
    best_epoch     = 0
    save_dir       = os.path.join(os.path.dirname(__file__), "..", "models", "doceditbert")
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*70, flush=True)
    print(f"{'Ep':>3}  {'Train Loss':>10}  {'Val Loss':>8}  "
          f"{'Intent Acc':>10}  {'Slot F1':>8}  {'Best?':>6}", flush=True)
    print("="*70, flush=True)

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_labels  = batch["intent_label"].to(device)
            slot_labels    = batch["slot_labels"].to(device)

            optimizer.zero_grad()
            intent_logits, slot_logits = model(input_ids, attention_mask)

            loss_i = intent_loss_fn(intent_logits, intent_labels)
            loss_s = slot_loss_fn(
                slot_logits.view(-1, NUM_SLOT_LABELS),
                slot_labels.view(-1),
            )
            loss = loss_i + loss_s   # equal weighting as specified
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────────
        val_loss, intent_acc, slot_f1, per_slot_f1 = evaluate(
            model, val_loader, intent_loss_fn, slot_loss_fn, device
        )

        is_best = slot_f1 > best_slot_f1
        if is_best:
            best_slot_f1 = slot_f1
            best_epoch   = epoch
            # Save model weights + tokenizer + label maps
            torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
            tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
            label_maps = {
                "intent_labels": INTENT_LABELS,
                "slot_labels":   SLOT_LABELS,
                "intent2id":     INTENT2ID,
                "slot2id":       SLOT2ID,
            }
            with open(os.path.join(save_dir, "label_maps.json"), "w") as f:
                json.dump(label_maps, f, indent=2)

        print(f"{epoch:>3}  {avg_train_loss:>10.4f}  {val_loss:>8.4f}  "
              f"{intent_acc:>10.4f}  {slot_f1:>8.4f}  "
              f"{'<-- BEST' if is_best else ''}",
              flush=True)

    # ── Final report ──────────────────────────────────────────────────────────
    print("="*70)
    print(f"\nBest slot F1: {best_slot_f1:.4f}  (epoch {best_epoch})")

    # Reload best checkpoint and report per-slot breakdown
    model.load_state_dict(torch.load(os.path.join(save_dir, "pytorch_model.bin"),
                                     map_location=device))
    _, final_intent_acc, final_slot_f1, final_per_slot = evaluate(
        model, val_loader, intent_loss_fn, slot_loss_fn, device
    )

    print(f"\nFinal metrics on validation set (best checkpoint):")
    print(f"  Intent accuracy : {final_intent_acc:.4f}")
    print(f"  Slot F1 (macro) : {final_slot_f1:.4f}")
    print(f"\n  Per-slot F1:")
    for label, f1 in final_per_slot.items():
        bar = "#" * int(f1 * 20)
        print(f"    {label:<12}  {f1:.4f}  {bar}")

    print(f"\nModel saved to: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    main()
