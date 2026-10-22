"""
Tanglish 500 — 5 systems including Groq LLM.
Groq uses 5-command sub-batches with 1s delay to stay within rate limits.
"""
import json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

with open("data/docEdit_tanglish_500.json") as f:
    data = json.load(f)

BATCH      = 5    # small batches because Groq is in each one
GROQ_DELAY = 1.0  # seconds between batches

rule_ok = bilstm_ok = bert_ok = groq_ok = xlmr_ok = 0
total = 0

print(f"Evaluating {len(data)} Tanglish commands (5 systems, Groq included)...\n")

for start in range(0, len(data), BATCH):
    batch = [
        {"text": ex["text"], "intent": ex["intent"],
         "from_anchor": ex["from_anchor"], "to_anchor": ex["to_anchor"]}
        for ex in data[start:start+BATCH]
    ]
    body = json.dumps({
        "commands":          batch,
        "include_groq":      True,
        "include_xlmr_zero": True,
        "include_mdistilbert": False,
    }).encode()
    req  = urllib.request.Request(
        "http://localhost:5050/batch-evaluate",
        data=body, headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        d    = json.loads(resp.read())
        n    = d["total"]
        rule_ok  += round(d["rule_accuracy"]                    * n)
        bilstm_ok+= round(d["bilstm_accuracy"]                  * n)
        bert_ok  += round(d["doceditbert_accuracy"]             * n)
        groq_ok  += round((d.get("groq_accuracy")      or 0)   * n)
        xlmr_ok  += round((d.get("xlmr_zero_accuracy") or 0)   * n)
        total    += n
    except Exception as e:
        print(f"  Batch {start} failed: {e}")
        total += len(batch)

    done_pct = (start + len(batch)) / len(data) * 100
    print(f"  [{start+len(batch):>4}/{len(data)}]  ({done_pct:.0f}%)  "
          f"rule={rule_ok/max(total,1)*100:.1f}%  bilstm={bilstm_ok/max(total,1)*100:.1f}%  "
          f"bert={bert_ok/max(total,1)*100:.1f}%  groq={groq_ok/max(total,1)*100:.1f}%  "
          f"xlmr={xlmr_ok/max(total,1)*100:.1f}%", flush=True)

    if start + BATCH < len(data):
        time.sleep(GROQ_DELAY)

print()
print("=" * 65)
print(f"TANGLISH 500 -- FINAL RESULTS ({total} commands)")
print("=" * 65)
rows = [
    ("Rule-based",        rule_ok),
    ("XLM-R Zero-shot",   xlmr_ok),
    ("Groq LLM",          groq_ok),
    ("BiLSTM",            bilstm_ok),
    ("DocEditBERT-1000",  bert_ok),
]
for name, ok in rows:
    print(f"  {name:<22}: {ok:>4}/{total}  ({ok/total*100:.1f}%)")
print("=" * 65)

# Save results for graph generation
results = {name: round(ok/total*100, 1) for name, ok in rows}
with open("data/tanglish_results.json", "w") as f:
    json.dump(results, f)
print("\nResults saved to data/tanglish_results.json")
