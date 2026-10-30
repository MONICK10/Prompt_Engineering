"""Streamlit UI: try one command live, read the saved results, or walk the workflow."""

import json

import pandas as pd
import streamlit as st

from src import report
from src.evaluate import load_commands, parse_json
from src.models import MODELS
from src.prompts import ALL_TECHNIQUES, build, run_technique

st.set_page_config(page_title="vdoc-prompt", layout="wide")
st.title("VDOC prompt comparison")

WORKFLOW_STEPS = [
    ("Voice", "the student speaks, in English or Tanglish"),
    ("Speech-to-text", "Whisper writes the words down"),
    ("Prompt + Model", "a template is filled in, one model is called"),
    ("JSON", "one action object comes back"),
    ("VDOC edit", "the document changes"),
]

# The pair the side-by-side section contrasts, and where it reads them from.
COMPARE = ("p1_zero_shot", "p4_few_shot")
COMPARE_MODEL = "groq"
DEFAULT_COMMAND = "kadaisi paragraph-a eduthudu"


def with_commands(df):
    """runs.csv stores only the command id, so bring the text back alongside."""
    rows = {int(r["id"]): r for r in load_commands("test")}
    df = df.copy()
    df["command"] = df["id"].map(lambda i: rows.get(int(i), {}).get("command"))
    df["expected_json"] = df["id"].map(lambda i: rows.get(int(i), {}).get("expected_json"))
    df["expected action"] = df["expected_json"].map(
        lambda e: json.loads(e)["action"] if isinstance(e, str) else None)
    return df.dropna(subset=["command"])


def draw_workflow():
    columns = st.columns([3, 1] * (len(WORKFLOW_STEPS) - 1) + [3])
    for i, (name, caption) in enumerate(WORKFLOW_STEPS):
        with columns[i * 2]:
            with st.container(border=True):
                st.markdown(f"**{name}**")
            st.caption(caption)
        if i < len(WORKFLOW_STEPS) - 1:
            columns[i * 2 + 1].markdown(
                "<div style='text-align:center;font-size:1.5rem;opacity:.4;"
                "padding-top:.4rem'>&rarr;</div>", unsafe_allow_html=True)


def side_by_side(df, command):
    """The same command under two techniques, read from runs.csv -- no new calls."""
    for column, technique in zip(st.columns(2), COMPARE):
        with column:
            st.markdown(f"**{technique}** on {COMPARE_MODEL}")
            hit = df[(df["model"] == COMPARE_MODEL)
                     & (df["technique"] == technique)
                     & (df["command"] == command)]
            if hit.empty:
                st.caption(f"not run for {COMPARE_MODEL} yet")
                continue
            row = hit.iloc[0]
            st.markdown("✅ exact match" if row["exact_ok"] else "❌ not an exact match")
            with st.expander("Prompt sent"):
                st.code(build(technique, command))
            st.caption("Raw model output")
            st.code(str(row["output"] or "(empty)")[:1500])
            parsed = parse_json(str(row["output"] or ""))
            st.caption("Parsed JSON")
            if parsed is None:
                st.error("no JSON in the reply")
            else:
                st.json(parsed)
            st.caption("Expected JSON")
            st.json(json.loads(row["expected_json"]))


def key_findings(df):
    """Every number here is read off the table, never written down."""
    accuracy, latency = report.accuracy_table(df), report.latency_table(df)
    notes = []

    if COMPARE_MODEL in accuracy.index:
        row = accuracy.loc[COMPARE_MODEL]
        if all(t in row.index for t in COMPARE) and row[list(COMPARE)].notna().all():
            zero, few = row[COMPARE[0]], row[COMPARE[1]]
            notes.append(f"**{COMPARE_MODEL}**: zero-shot {zero:.1f}% → "
                         f"few-shot {few:.1f}% ({few - zero:+.1f} points).")

    if accuracy.notna().any().any():
        model = accuracy.max(axis=1).idxmax()
        technique = accuracy.loc[model].idxmax()
        seconds = (latency.loc[model, technique]
                   if technique in latency.columns else None)
        tail = f", median {seconds:.2f} s" if pd.notna(seconds) else ""
        notes.append(f"**Best**: {technique} on {model}, "
                     f"{accuracy.loc[model, technique]:.1f}%{tail}.")

    if {"p4_few_shot", "p5_cot"} <= set(accuracy.columns):
        drop = (accuracy["p4_few_shot"] - accuracy["p5_cot"]).dropna()
        if not drop.empty and drop.max() > 0:
            notes.append(f"**Chain of thought cost the most on {drop.idxmax()}**: "
                         f"{drop.max():.1f} points below few-shot.")
        elif not drop.empty:
            notes.append("**Chain of thought** did not lower accuracy on any model.")
    return notes


def partial_runs(df):
    """Say plainly which cells are missing, so a gap is never read as a zero."""
    every = sorted(df["technique"].unique())
    for model in sorted(df["model"].unique()):
        missing = [t for t in every if t not in set(df[df["model"] == model]["technique"])]
        if not missing:
            continue
        reason = " — free tier limit of 20 requests/day" if model == "gemini" else ""
        st.info(f"{model}: partial results{reason}. Not run: {', '.join(missing)}.")


def workflow_sections(raw):
    df = with_commands(raw)

    st.subheader("One command, two prompts")
    available = df.sort_values("id")["command"].drop_duplicates().tolist()
    if not available:
        st.caption("No test commands in runs.csv yet.")
        return
    start = available.index(DEFAULT_COMMAND) if DEFAULT_COMMAND in available else 0
    side_by_side(df, st.selectbox("Command", available, index=start))

    st.subheader("Exact-match accuracy %  (model × technique)")
    accuracy = report.accuracy_table(df)
    st.dataframe(accuracy.style.format("{:.1f}", na_rep="not run")
                 .highlight_max(axis=None, color="#cdebd9"))
    partial_runs(df)

    st.subheader("Charts")
    st.image(str(report.accuracy_chart(df)))
    st.image(str(report.latency_chart(df)))

    st.subheader("Key findings")
    for note in key_findings(df) or ["Not enough results yet."]:
        st.markdown(f"- {note}")

    st.subheader(f"Where {COMPARE[1]} went wrong")
    wrong = df[(df["technique"] == COMPARE[1]) & (~df["exact_ok"])]
    if wrong.empty:
        st.caption(f"No exact-match failures recorded for {COMPARE[1]}.")
    else:
        st.dataframe(wrong[["command", "expected action", "output",
                            "model", "technique"]]
                     .rename(columns={"output": "model output"})
                     .head(10), hide_index=True)


try_it, results, workflow = st.tabs(["Try it", "Results", "Workflow & Results"])

with try_it:
    command = st.text_input("Voice command", "last line-a delete pannu")
    left, right = st.columns(2)
    model = left.selectbox("Model", MODELS)
    technique = right.selectbox("Technique", ALL_TECHNIQUES, index=3)

    if st.button("Run", type="primary"):
        with st.spinner(f"asking {model}..."):
            output, latency = run_technique(model, technique, command)
        st.caption(f"took {latency:.2f} s")
        parsed = parse_json(output)
        if parsed is None:
            st.error("No valid JSON in the reply.")
        else:
            st.json(parsed)
        with st.expander("Raw model output"):
            st.code(output or "(empty)")

with results:
    try:
        df = report.load()
    except FileNotFoundError as err:
        st.warning(f"{err}")
    else:
        st.subheader("Exact-match accuracy %  (model x technique)")
        st.dataframe(report.accuracy_table(df))

        st.subheader("Exact-match accuracy %  (category x model)")
        st.dataframe(report.category_table(df))

        st.subheader("Median latency in seconds  (model x technique)")
        st.dataframe(report.latency_table(df))

        st.image(str(report.accuracy_chart(df)))
        st.image(str(report.category_chart(df)))

        with st.expander("Every single run"):
            st.dataframe(df)

with workflow:
    st.caption(f"{len(load_commands('test'))} test commands in data/commands.csv. "
               "Everything below is read from results/runs.csv.")

    st.subheader("How a spoken command becomes an edit")
    draw_workflow()

    try:
        workflow_sections(report.load())
    except FileNotFoundError as err:
        st.warning(f"{err}")
