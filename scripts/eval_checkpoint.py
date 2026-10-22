"""
Quick re-evaluation of the saved DocEditBERT checkpoint with the corrected
slot F1 metric (macro over ground-truth slot classes only, O excluded).
Run from the project root: python scripts/eval_checkpoint.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast
from sklearn.model_selection import train_test_split

# Re-use everything from train_model
from train_model import (
    DocEditDataset, DocEditBERT,
    INTENT2ID, SLOT_LABELS, SEED,
    NUM_SLOT_LABELS, IGNORE_INDEX,
    compute_metrics,
)
from torch import nn

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = os.path.join(os.path.dirname(__file__), "..", "models", "doceditbert")

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "docEdit300.json")
with open(data_path) as f:
    examples = json.load(f)

_, val_examples = train_test_split(
    examples, test_size=0.2, random_state=SEED,
    stratify=[ex["intent"] for ex in examples],
)

tokenizer   = DistilBertTokenizerFast.from_pretrained(os.path.join(save_dir, "tokenizer"))
val_dataset = DocEditDataset(val_examples, tokenizer)
val_loader  = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = DocEditBERT().to(device)
model.load_state_dict(torch.load(os.path.join(save_dir, "pytorch_model.bin"),
                                 map_location=device))
model.eval()

intent_loss_fn = nn.CrossEntropyLoss()
slot_loss_fn   = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

all_intent_preds, all_intent_labels = [], []
all_slot_preds,   all_slot_labels   = [], []

with torch.no_grad():
    for batch in val_loader:
        iids  = batch["input_ids"].to(device)
        amask = batch["attention_mask"].to(device)
        il    = batch["intent_label"].to(device)
        sl    = batch["slot_labels"].to(device)

        intent_logits, slot_logits = model(iids, amask)

        all_intent_preds.extend(intent_logits.argmax(-1).cpu().tolist())
        all_intent_labels.extend(il.cpu().tolist())
        all_slot_preds.extend(slot_logits.argmax(-1).cpu().tolist())
        all_slot_labels.extend(sl.cpu().tolist())

intent_acc, slot_f1, per_slot_f1 = compute_metrics(
    all_intent_preds, all_intent_labels,
    all_slot_preds,   all_slot_labels,
)

print("\n" + "="*50)
print("DocEditBERT  --  Validation set (60 examples)")
print("="*50)
print(f"  Intent accuracy : {intent_acc:.4f}")
print(f"  Slot F1 (macro) : {slot_f1:.4f}")
print(f"\n  Per-slot F1 (present classes only):")
for label, f1 in per_slot_f1.items():
    bar = "#" * int(f1 * 30)
    print(f"    {label:<12}  {f1:.4f}  {bar}")
print("="*50)
