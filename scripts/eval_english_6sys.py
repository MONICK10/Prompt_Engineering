"""
English 500 evaluation — 6 systems:
Rule-based, BiLSTM, XLM-R Zero-shot, mDistilBERT (Tanglish-only), DocEditBERT-1000
(Groq excluded — rate limited; use frontend toggle if needed)
"""
import json
import urllib.request

with open("data/docEdit500.json") as f:
    data = json.load(f)

BATCH = 50
rule_ok = bilstm_ok = bert_ok = xlmr_ok = mdist_ok = 0
total = 0

print(f"Evaluating {len(data)} English commands across 6 systems...")
print("(mDistilBERT + XLM-R load on first batch — may take ~30s)\n")

for start in range(0, len(data), BATCH):
    batch = [
        {
            "text":        ex["text"],
            "intent":      ex["intent"],
            "from_anchor": ex["from_anchor"],
            "to_anchor":   ex["to_anchor"],
        }
        for ex in data[start : start + BATCH]
    ]
    body = json.dumps({
        "commands":          batch,
        "include_groq":      False,
        "include_xlmr_zero": True,
        "include_mdistilbert": True,
    }).encode()
    req  = urllib.request.Request(
        "http://localhost:5050/batch-evaluate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=120)
    d    = json.loads(resp.read())

    n = d["total"]
    rule_ok   += round(d["rule_accuracy"]                      * n)
    bilstm_ok += round(d["bilstm_accuracy"]                    * n)
    bert_ok   += round(d["doceditbert_accuracy"]               * n)
    xlmr_ok   += round((d.get("xlmr_zero_accuracy")   or 0)   * n)
    mdist_ok  += round((d.get("mdistilbert_accuracy")  or 0)   * n)
    total     += n

    print(
        f"  [{start+n:>4}/{len(data)}]  "
        f"rule={d['rule_accuracy']:.3f}  "
        f"bilstm={d['bilstm_accuracy']:.3f}  "
        f"bert={d['doceditbert_accuracy']:.3f}  "
        f"xlmr={d.get('xlmr_zero_accuracy',0):.3f}  "
        f"mdist={d.get('mdistilbert_accuracy',0):.3f}",
        flush=True,
    )

print()
print("=" * 65)
print(f"ENGLISH 500 -- FINAL RESULTS ({total} commands)")
print("=" * 65)
rows = [
    ("Rule-based",                rule_ok),
    ("BiLSTM",                    bilstm_ok),
    ("XLM-R Zero-shot",           xlmr_ok),
    ("mDistilBERT (Tanglish-only)", mdist_ok),
    ("DocEditBERT-1000",           bert_ok),
]
for name, ok in rows:
    print(f"  {name:<28}: {ok:>4}/{total}  ({ok/total*100:.1f}%)")
print("=" * 65)
