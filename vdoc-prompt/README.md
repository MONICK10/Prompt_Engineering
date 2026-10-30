# vdoc-prompt

Which prompt technique, and which LLM, best turns a spoken document-editing
command into a JSON action? The commands are English, Tanglish (Tamil written in
English letters), or a mix of both.

```
"last line-a delete pannu"
-> {"action":"delete","target":{"type":"line","index":-1},"value":null,"question":null}
```

## What gets compared

**Six techniques** (`prompts/*.txt`, `{command}` is swapped in at run time):

| | technique | idea |
|---|---|---|
| p1 | zero shot | just ask |
| p2 | role | tell the model it is VDOC's command engine |
| p3 | structured | spell out the JSON schema and the number rules |
| p4 | few shot | p3 plus six worked examples |
| p5 | chain of thought | p4 plus "think in 3 steps inside `<think></think>`" |
| p6 | self-consistency | run p5 three times at temperature 0.7, keep the most common answer |

**Three hand-tuned versions of p4**, the prompt-engineering track. Same
technique, improved by hand, no retraining:

| | version | change |
|---|---|---|
| v1 | `p4_few_shot_v1` | today's p4, the baseline |
| v2 | `p4_few_shot_v2` | adds `eduthudu` = delete and two `maathu` -> replace examples |
| v3 | `p4_few_shot_v3` | v2 plus the rule that separates formatting from replacing |

These are not in the default run. Name them with `--techniques` to run them.

**Three models** (`src/models.py`): `gemini`, `groq`, `ollama-local`.

| name | package | default model | set with |
|---|---|---|---|
| `gemini` | `google-generativeai` | `gemini-2.5-flash` | `GEMINI_MODEL` |
| `groq` | `groq` | `openai/gpt-oss-20b` | `GROQ_MODEL` |
| `ollama-local` | `ollama` | `llama3.2:3b` | `OLLAMA_MODEL` |

Check which ones your keys can actually reach before a long run:

```powershell
python -m src.models
```

Model names go stale fast. Groq free keys no longer serve a Llama 3 chat model,
which is why the default there is `openai/gpt-oss-20b`. To see your own list:

```powershell
$k = (Get-Content .env | Select-String '^GROQ_API_KEY=').ToString().Split('=',2)[1].Trim()
(Invoke-RestMethod "https://api.groq.com/openai/v1/models" -Headers @{Authorization="Bearer $k"}).data.id
```

p1-p5 run at temperature 0, so they are repeatable.

## Action schema

```json
{
  "action": "bold|italic|underline|highlight|font-increase|delete|insert|rewrite|replace|move|undo|clarify",
  "target": {"type": "title|paragraph|line|word|selection|document", "index": null},
  "value": null,
  "question": null
}
```

`value` holds the text for `insert`, the tone word for `rewrite`, and a pair
`{"from": "<old word>", "to": "<new word>"}` for `replace`.
`question` is filled only when the action is `clarify`.

`replace` is the action for swapping one word for another, which the schema
had no way to express before: "3rd line-la reported-a car-nu maathu" had no
right answer available to it. `highlight` and `font-increase` were added for
the same reason — 17 commands in the set need them, and VDOC 2.0's own model
has both as intents, so leaving them out would have handicapped the prompts
in the DocEditBERT comparison.

`maathu` is the hard one: it covers both "change the tone" (rewrite) and
"change this word into that word" (replace). What follows it decides which.
That ambiguity is what the v2 and v3 prompts are aimed at.

Tanglish hints the prompts rely on: *pannu/aakku* = do/make, *eduthudu* = delete,
*maathu* = change (rewrite), *serkku* = add (insert), *nagarthu* = move,
*mudhal* = 1, *rendavathu* = 2, *kadaisi/last* = -1.

## Scoring

`src/evaluate.py` marks each reply three ways:

- **valid_json** - a JSON object could be pulled out of the reply
- **action_ok** - the action verb matches
- **exact_ok** - action **and** `target.type` **and** `target.index` all match;
  `value` is also checked for `insert`, `rewrite` and `replace`

For `replace`, both halves of the value have to be right: swapping the wrong
word out, or the wrong word in, is a different edit from the one asked for.
A reply that puts a bare string there ("reported -> car") has not followed
the schema and does not count as exact.

For an `unclear` command, `clarify` is the only right answer. For every clear
command, `clarify` counts as wrong.

## Setup (Windows PowerShell)

```powershell
cd E:\prompt_project\vdoc-prompt
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `Activate.ps1` is blocked, allow scripts for this session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Add your keys:

```powershell
Copy-Item .env.example .env
notepad .env
```

Pull the local model (needs [Ollama](https://ollama.com) installed and running):

```powershell
ollama pull llama3.2:3b
```

Any small instruct model works. To use one you already have instead of
downloading another, put its name in `.env` — `ollama list` shows them:

```
OLLAMA_MODEL=qwen2.5:3b
```

## Run

```powershell
python -m src.evaluate            # every model x technique on split=test
python -m src.report              # tables + results\accuracy.png, results\category.png
streamlit run app.py              # Try it / Results tabs
```

`src.evaluate` appends to `results\runs.csv` after every single call and skips
rows that are already there, so you can stop it with Ctrl+C and run it again to
carry on. Delete `results\runs.csv` to start fresh.

Run only part of the grid:

```powershell
python -m src.evaluate --models groq --techniques p4_few_shot p5_cot
python -m src.evaluate --split tune
```

The order to run things in, given that Gemini is capped at 20 calls a day:

```powershell
# 1. the six techniques on the two models with room (groq first, it is faster)
python -m src.evaluate --models groq
python -m src.evaluate --models ollama-local

# 2. the tuning track, v1 -> v2 -> v3, on the same two models
python -m src.evaluate --models groq ollama-local `
    --techniques p4_few_shot_v1 p4_few_shot_v2 p4_few_shot_v3

# 3. only once the tables say which technique wins, spend Gemini on it
python -m src.evaluate --models gemini --techniques p4_few_shot_v3

python -m src.report
```

Step 1 is 6 x 130 = 780 calls per model; step 2 is another 390 per model.
Groq pauses 1 s between calls, so budget roughly 20 minutes per model for
step 1 and 10 for step 2. `src.evaluate` appends as it goes and skips rows it
already has, so Ctrl+C and rerun is safe.

## Rate limits

The pause between calls is set per model in `src/models.py`:

```python
PAUSE_SECONDS = {"gemini": 13.0, "groq": 1.0, "ollama-local": 1.0}
```

If a call is rate limited anyway, `call_model` reads the wait the server asks
for ("retry in 7.5s", `retry_delay { seconds: 7 }`), sleeps that long and tries
again, up to 3 times. The waiting is not counted in `latency_s`, so the latency
table still measures the model rather than the queue. If the server asks for
longer than `MAX_RETRY_WAIT` (2 minutes) the quota is spent for the day, so it
gives up straight away instead of sleeping for hours.

**Gemini's free tier is capped at 20 generate-content requests per day, per
model.** That is the binding limit, not the per-minute one. A full grid needs
80 Gemini calls (10 test commands x 5 techniques, plus 3 calls each for p6),
so on a free key it takes about four days. The practical options:

- spread it out: rerun the same command daily, the resume logic fills the gaps
- use a second Gemini model as a separate 20-a-day budget, e.g.
  `GEMINI_MODEL=gemini-2.5-flash-lite`
- enable billing on the Google Cloud project for a much higher limit
- drop Gemini and compare `groq` against `ollama-local`, which have room

A call that fails for good is saved with an empty `output` and
`valid_json=False`. To clear those out and run them again:

```powershell
python -m src.evaluate --retry-failed
```

## Data

`data/commands.csv` - 150 commands with columns `id, command, category,
expected_json, split`. `split` is `tune` (20 rows, for writing prompts) or
`test` (130 rows, for the numbers you report).

The test split is mixed like this:

| category | rows |
|---|---|
| english | 30 |
| tanglish | 50 |
| tamil_number | 20 |
| replace | 15 |
| unclear | 15 |

Tanglish phrasing is reused from the VDOC 2.0 hard cases and
`docEdit1000.json` rather than invented. `rewrite` is no longer a category of
its own — a rewrite command written in Tanglish counts as `tanglish` — so that
the five categories match the mix above.

**Every few-shot example in `prompts/` is a `tune` row.** No test command is
ever shown to a model inside its own prompt. There is a check for this:

```powershell
python -c "import re,pathlib; from src.evaluate import load_commands; t={r['command'] for r in load_commands('test')}; print([c for f in pathlib.Path('prompts').glob('*.txt') for c in re.findall(r'^Command: (.+)$', f.read_text(encoding='utf-8'), re.M) if c in t] or 'no leakage')"
```

`results/runs_v0_15commands.csv` holds the first results, from back when the
set was 15 commands. It is kept only as a record: those ids mean different
commands now, so it must not be used to resume a run.

## Layout

```
data/commands.csv      the test set
prompts/p1..p5.txt     one file per technique (p6 reuses p5)
src/models.py          call_model(model, prompt, temperature) -> (text, seconds)
src/prompts.py         load_template, build, self_consistency
src/evaluate.py        parse_json, score, run_all
src/report.py          tables and the two charts
app.py                 Streamlit UI
results/runs.csv       one row per model x technique x command
```
