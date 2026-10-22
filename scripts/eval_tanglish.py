import json
import urllib.request

with open("data/docEdit_tanglish_500.json") as f:
    data = json.load(f)

BATCH = 50
rule_ok = bilstm_ok = bert_ok = 0
total = 0

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
    body = json.dumps({"commands": batch}).encode()
    req  = urllib.request.Request(
        "http://localhost:5050/batch-evaluate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=60)
    d    = json.loads(resp.read())

    n = d["total"]
    rule_ok   += round(d["rule_accuracy"]        * n)
    bilstm_ok += round(d["bilstm_accuracy"]      * n)
    bert_ok   += round(d["doceditbert_accuracy"] * n)
    total     += n
    pct_done   = (start + n) / len(data) * 100
    print(
        f"  [{start+n:>4}/{len(data)}]  "
        f"rule={d['rule_accuracy']:.3f}  "
        f"bilstm={d['bilstm_accuracy']:.3f}  "
        f"bert={d['doceditbert_accuracy']:.3f}",
        flush=True,
    )

print()
print("=" * 55)
print(f"TANGLISH 500 -- FINAL RESULTS ({total} commands)")
print("=" * 55)
print(f"  Rule-based  : {rule_ok}/{total}  ({rule_ok/total*100:.1f}%)")
print(f"  BiLSTM      : {bilstm_ok}/{total}  ({bilstm_ok/total*100:.1f}%)")
print(f"  DocEditBERT : {bert_ok}/{total}  ({bert_ok/total*100:.1f}%)  <-- bilingual model (doceditbert_1000)")
