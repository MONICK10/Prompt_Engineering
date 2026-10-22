import json
import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Label maps (identical to DocEditBERT for fair comparison) ─────────────────
INTENT_LABELS = ["bold", "italic", "delete", "underline", "heading2", "heading3"]
SLOT_LABELS = [
    "O",
    "B-ACTION", "I-ACTION",
    "B-FROM",   "I-FROM",
    "B-TO",     "I-TO",
    "B-ANCHOR", "I-ANCHOR",
]

INTENT2ID    = {l: i for i, l in enumerate(INTENT_LABELS)}
SLOT2ID      = {l: i for i, l in enumerate(SLOT_LABELS)}
NUM_INTENTS  = len(INTENT_LABELS)   # 6
NUM_SLOTS    = len(SLOT_LABELS)     # 9

PAD_TOKEN  = "<PAD>"
UNK_TOKEN  = "<UNK>"
PAD_IDX    = 0
UNK_IDX    = 1
IGNORE_IDX = -100    # slot positions to skip in loss/metrics


# ── Vocabulary (built from training split only to avoid leakage) ──────────────
def build_vocab(examples):
    tokens = [tok for ex in examples for tok in ex["tokens"]]
    vocab  = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for tok in sorted(set(tokens)):
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


# ── Dataset ──────────────────────────────────────────────────────────────────
class BiLSTMDataset(Dataset):
    def __init__(self, examples, vocab):
        self.examples = examples
        self.vocab    = vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex          = self.examples[idx]
        token_ids   = [self.vocab.get(t, UNK_IDX) for t in ex["tokens"]]
        slot_ids    = [SLOT2ID[l] for l in ex["labels"]]
        intent_id   = INTENT2ID[ex["intent"]]
        return {
            "token_ids":    torch.tensor(token_ids, dtype=torch.long),
            "slot_labels":  torch.tensor(slot_ids,  dtype=torch.long),
            "intent_label": torch.tensor(intent_id, dtype=torch.long),
            "length":       torch.tensor(len(token_ids), dtype=torch.long),
        }


def collate_fn(batch):
    """Pad token_ids and slot_labels to the longest sequence in the batch."""
    max_len = max(item["length"].item() for item in batch)

    token_ids_padded   = []
    slot_labels_padded = []
    intent_labels      = []
    lengths            = []

    for item in batch:
        seq_len  = item["length"].item()
        pad_size = max_len - seq_len

        token_ids_padded.append(
            torch.cat([item["token_ids"],
                       torch.full((pad_size,), PAD_IDX, dtype=torch.long)])
        )
        # Padding positions get IGNORE_IDX so they don't affect loss or F1
        slot_labels_padded.append(
            torch.cat([item["slot_labels"],
                       torch.full((pad_size,), IGNORE_IDX, dtype=torch.long)])
        )
        intent_labels.append(item["intent_label"])
        lengths.append(item["length"])

    return {
        "token_ids":    torch.stack(token_ids_padded),
        "slot_labels":  torch.stack(slot_labels_padded),
        "intent_label": torch.stack(intent_labels),
        "lengths":      torch.stack(lengths),
    }


# ── Model ────────────────────────────────────────────────────────────────────
class BiLSTMJointModel(nn.Module):
    """
    From-scratch 2-layer BiLSTM with random embeddings.
    No pretrained weights anywhere.
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128,
                 num_layers=2, num_intents=NUM_INTENTS,
                 num_slot_labels=NUM_SLOTS, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,   # applied between layers (only when num_layers > 1)
        )
        lstm_out = hidden_dim * 2   # bidirectional

        self.dropout     = nn.Dropout(dropout)
        self.intent_head = nn.Linear(lstm_out, num_intents)
        self.slot_head   = nn.Linear(lstm_out, num_slot_labels)

    def forward(self, token_ids, lengths):
        # token_ids : (B, T)
        emb = self.dropout(self.embedding(token_ids))   # (B, T, E)

        packed     = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, (hidden, _) = self.bilstm(packed)
        lstm_out, _           = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True
        )
        # lstm_out : (B, T, hidden*2)

        # Intent: concat last-layer forward + backward final states
        # hidden shape: (num_layers * 2, B, hidden_dim)
        fwd = hidden[-2]                            # (B, hidden_dim)
        bwd = hidden[-1]                            # (B, hidden_dim)
        cls = torch.cat([fwd, bwd], dim=-1)        # (B, hidden_dim*2)

        intent_logits = self.intent_head(self.dropout(cls))       # (B, 6)
        slot_logits   = self.slot_head(self.dropout(lstm_out))    # (B, T, 9)
        return intent_logits, slot_logits


# ── Metrics (same corrected logic as DocEditBERT) ─────────────────────────────
def compute_metrics(all_intent_preds, all_intent_labels,
                    all_slot_preds,   all_slot_labels):
    intent_acc = accuracy_score(all_intent_labels, all_intent_preds)

    flat_preds, flat_labels = [], []
    for preds_seq, labels_seq in zip(all_slot_preds, all_slot_labels):
        for p, l in zip(preds_seq, labels_seq):
            if l != IGNORE_IDX:
                flat_preds.append(p)
                flat_labels.append(l)

    present_in_gt = sorted(set(flat_labels))
    slot_classes  = [i for i in present_in_gt if SLOT_LABELS[i] != "O"]

    slot_f1  = f1_score(flat_labels, flat_preds,
                        labels=slot_classes, average="macro", zero_division=0)
    per_cls  = f1_score(flat_labels, flat_preds,
                        labels=present_in_gt, average=None, zero_division=0)
    per_slot = {SLOT_LABELS[present_in_gt[i]]: round(float(per_cls[i]), 4)
                for i in range(len(present_in_gt))}
    return intent_acc, slot_f1, per_slot


# ── Evaluate one split ────────────────────────────────────────────────────────
def evaluate(model, loader, intent_loss_fn, slot_loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_intent_preds, all_intent_labels = [], []
    all_slot_preds,   all_slot_labels   = [], []

    with torch.no_grad():
        for batch in loader:
            token_ids     = batch["token_ids"].to(device)
            lengths       = batch["lengths"].to(device)
            intent_labels = batch["intent_label"].to(device)
            slot_labels   = batch["slot_labels"].to(device)

            intent_logits, slot_logits = model(token_ids, lengths)

            loss_i = intent_loss_fn(intent_logits, intent_labels)
            loss_s = slot_loss_fn(
                slot_logits.view(-1, NUM_SLOTS),
                slot_labels.view(-1),
            )
            total_loss += (loss_i + loss_s).item()

            all_intent_preds.extend(intent_logits.argmax(-1).cpu().tolist())
            all_intent_labels.extend(intent_labels.cpu().tolist())

            # Trim slot preds to match the padded label sequence length
            B, T = slot_labels.shape
            slot_preds_trimmed = slot_logits[:, :T, :].argmax(-1)
            all_slot_preds.extend(slot_preds_trimmed.cpu().tolist())
            all_slot_labels.extend(slot_labels.cpu().tolist())

    avg_loss                          = total_loss / len(loader)
    intent_acc, slot_f1, per_slot_f1 = compute_metrics(
        all_intent_preds, all_intent_labels,
        all_slot_preds,   all_slot_labels,
    )
    return avg_loss, intent_acc, slot_f1, per_slot_f1


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "docEdit300.json")
    with open(data_path) as f:
        examples = json.load(f)

    train_examples, val_examples = train_test_split(
        examples, test_size=0.2, random_state=SEED,
        stratify=[ex["intent"] for ex in examples],
    )
    print(f"Train: {len(train_examples)}  |  Val: {len(val_examples)}")

    # ── Vocabulary (built from train split only) ───────────────────────────────
    vocab      = build_vocab(train_examples)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    train_dataset = BiLSTMDataset(train_examples, vocab)
    val_dataset   = BiLSTMDataset(val_examples,   vocab)

    train_loader = DataLoader(train_dataset, batch_size=16,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=16,
                              shuffle=False, collate_fn=collate_fn)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BiLSTMJointModel(vocab_size=vocab_size).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ── Loss, optimizer ───────────────────────────────────────────────────────
    EPOCHS = 15
    LR     = 1e-3   # standard for LSTMs trained from scratch (vs 2e-5 for BERT)

    intent_loss_fn = nn.CrossEntropyLoss()
    slot_loss_fn   = nn.CrossEntropyLoss(ignore_index=IGNORE_IDX)
    optimizer      = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Cosine annealing — works better than linear schedule for small LSTMs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_slot_f1 = 0.0
    best_epoch   = 0
    save_dir     = os.path.join(os.path.dirname(__file__), "..", "models", "bilstm")
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*70, flush=True)
    print(f"{'Ep':>3}  {'Train Loss':>10}  {'Val Loss':>8}  "
          f"{'Intent Acc':>10}  {'Slot F1':>8}  {'Best?':>8}", flush=True)
    print("="*70, flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            token_ids     = batch["token_ids"].to(device)
            lengths       = batch["lengths"].to(device)
            intent_labels = batch["intent_label"].to(device)
            slot_labels   = batch["slot_labels"].to(device)

            optimizer.zero_grad()
            intent_logits, slot_logits = model(token_ids, lengths)

            loss_i = intent_loss_fn(intent_logits, intent_labels)
            loss_s = slot_loss_fn(
                slot_logits.view(-1, NUM_SLOTS),
                slot_labels.view(-1),
            )
            loss = loss_i + loss_s
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)

        val_loss, intent_acc, slot_f1, _ = evaluate(
            model, val_loader, intent_loss_fn, slot_loss_fn, device
        )

        is_best = slot_f1 > best_slot_f1
        if is_best:
            best_slot_f1 = slot_f1
            best_epoch   = epoch
            torch.save(model.state_dict(),
                       os.path.join(save_dir, "pytorch_model.bin"))
            with open(os.path.join(save_dir, "vocab.json"), "w") as f:
                json.dump(vocab, f)
            config = {
                "vocab_size": vocab_size,
                "embed_dim": 128, "hidden_dim": 128,
                "num_layers": 2, "dropout": 0.3,
                "num_intents": NUM_INTENTS, "num_slot_labels": NUM_SLOTS,
                "intent_labels": INTENT_LABELS, "slot_labels": SLOT_LABELS,
            }
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

        print(f"{epoch:>3}  {avg_train_loss:>10.4f}  {val_loss:>8.4f}  "
              f"{intent_acc:>10.4f}  {slot_f1:>8.4f}  "
              f"{'<-- BEST' if is_best else ''}", flush=True)

    # ── Final report ──────────────────────────────────────────────────────────
    print("="*70, flush=True)
    print(f"\nBest slot F1: {best_slot_f1:.4f}  (epoch {best_epoch})", flush=True)

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
        bar = "#" * int(f1 * 30)
        print(f"    {label:<12}  {f1:.4f}  {bar}")
    print(f"\nModel saved to: {os.path.abspath(save_dir)}", flush=True)


if __name__ == "__main__":
    main()
