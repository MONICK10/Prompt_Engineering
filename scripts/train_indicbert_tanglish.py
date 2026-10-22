"""
Fine-tune IndicBERT (ai4bharat/indic-bert) on docEdit_tanglish_500.json.

  - 15 epochs, AdamW lr=2e-5, batch 16, early stopping patience=3
  - Saves to models/indicbert_tanglish/
      pytorch_model.bin
      tokenizer/
"""
import os
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score as seq_f1

ROOT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_F   = os.path.join(ROOT, "data",   "docEdit_tanglish_500.json")
MAPS_F   = os.path.join(ROOT, "models", "doceditbert", "label_maps.json")
SAVE_DIR = os.path.join(ROOT, "models", "indicbert_tanglish")
MODEL_ID = "distilbert-base-multilingual-cased"

os.makedirs(SAVE_DIR, exist_ok=True)

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

        word_ids = enc.word_ids()
        slot_ids = []
        prev_wid = None
        for wid in word_ids:
            if wid is None:
                slot_ids.append(-100)
            elif wid == prev_wid:
                prev_label = labels[wid]
                cont = ("I-" + prev_label[2:]) if prev_label.startswith("B-") else prev_label
                slot_ids.append(slot2id.get(cont, slot2id.get(prev_label, 0)))
            else:
                slot_ids.append(slot2id.get(labels[wid], 0))
            prev_wid = wid

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros(64, dtype=torch.long)).squeeze(0) if "token_type_ids" in enc else torch.zeros(64, dtype=torch.long),
            "intent_label":   torch.tensor(intent2id.get(ex["intent"], 0), dtype=torch.long),
            "slot_labels":    torch.tensor(slot_ids, dtype=torch.long),
        }


# ── Model ─────────────────────────────────────────────────────────────────────
class IndicBERTDocEdit(nn.Module):
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


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    intent_correct = 0
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in loader:
            iids  = batch["input_ids"].to(_device)
            amask = batch["attention_mask"].to(_device)
            ttype = batch["token_type_ids"].to(_device)
            ilabs = batch["intent_label"].to(_device)
            slabs = batch["slot_labels"].to(_device)

            i_logits, s_logits = model(iids, amask, ttype)
            intent_correct += (i_logits.argmax(-1) == ilabs).sum().item()

            for b in range(iids.size(0)):
                t_seq, p_seq = [], []
                for pos in range(slabs.size(1)):
                    true_id = slabs[b, pos].item()
                    if true_id == -100:
                        continue
                    t_seq.append(SLOT_LABELS[true_id])
                    p_seq.append(SLOT_LABELS[s_logits[b, pos].argmax(-1).item()])
                all_true.append(t_seq)
                all_pred.append(p_seq)

    return {
        "intent_acc": intent_correct / len(loader.dataset),
        "slot_f1":    seq_f1(all_true, all_pred, zero_division=0),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    with open(DATA_F) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} Tanglish examples from {DATA_F}")

    train_ex, val_ex = train_test_split(
        data, test_size=0.2, random_state=42,
        stratify=[ex["intent"] for ex in data],
    )
    print(f"Train: {len(train_ex)}  |  Val: {len(val_ex)}")

    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    train_ds = TanglishDataset(train_ex, tokenizer)
    val_ds   = TanglishDataset(val_ex,   tokenizer)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32)

    print(f"Loading model: {MODEL_ID}  (multilingual DistilBERT, 135M params, 104 languages)")
    backbone = AutoModel.from_pretrained(MODEL_ID)
    model    = IndicBERTDocEdit(backbone).to(_device)

    EPOCHS   = 15
    LR       = 2e-5
    PATIENCE = 3

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * 0.1)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
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
            ttype = batch["token_type_ids"].to(_device)
            ilabs = batch["intent_label"].to(_device)
            slabs = batch["slot_labels"].to(_device)

            optimizer.zero_grad()
            i_logits, s_logits = model(iids, amask, ttype)
            loss = (intent_loss_fn(i_logits, ilabs) +
                    slot_loss_fn(s_logits.view(-1, NUM_SLOTS), slabs.view(-1)))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                iids  = batch["input_ids"].to(_device)
                amask = batch["attention_mask"].to(_device)
                ttype = batch["token_type_ids"].to(_device)
                ilabs = batch["intent_label"].to(_device)
                slabs = batch["slot_labels"].to(_device)
                i_logits, s_logits = model(iids, amask, ttype)
                v_loss += (intent_loss_fn(i_logits, ilabs) +
                           slot_loss_fn(s_logits.view(-1, NUM_SLOTS), slabs.view(-1))).item()
        v_loss /= len(val_loader)

        metrics  = evaluate(model, val_loader)
        improved = metrics["slot_f1"] > best_f1
        tag      = " <-- BEST" if improved else ""
        print(f" {epoch:>4}  {tr_loss:>8.4f}  {v_loss:>8.4f}  "
              f"{metrics['intent_acc']:>8.4f}  {metrics['slot_f1']:>8.4f}{tag}", flush=True)

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

    if best_state:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "pytorch_model.bin"))
    tokenizer.save_pretrained(os.path.join(SAVE_DIR, "tokenizer"))

    final = evaluate(model, val_loader)
    print(f"\nFinal val  intent_acc={final['intent_acc']:.4f}  slot_f1={final['slot_f1']:.4f}  (n={len(val_ds)})")
    print(f"Model saved to: {os.path.abspath(SAVE_DIR)}")


if __name__ == "__main__":
    main()
