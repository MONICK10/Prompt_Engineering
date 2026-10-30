"""Tables and charts built only from results/runs.csv."""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # save files, never open a window
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS_FILE = ROOT / "results" / "runs.csv"
ACCURACY_PNG = ROOT / "results" / "accuracy.png"
CATEGORY_PNG = ROOT / "results" / "category.png"
LATENCY_PNG = ROOT / "results" / "latency.png"

# A spoken command that takes longer than this to come back stops feeling
# like speech and starts feeling like a form submission.
VOICE_LIMIT_S = 2.0

# One fixed colour per model, always in this order so a chart never
# repaints a model just because another one is missing.
MODEL_COLORS = {"gemini": "#2a78d6", "groq": "#eb6834", "ollama-local": "#1baf7a"}
SPARE_COLOR = "#eda100"
SURFACE, INK, INK_SOFT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BOOL_COLS = ["valid_json", "action_ok", "exact_ok"]


def load():
    """Read runs.csv into a DataFrame with real True/False columns."""
    if not RUNS_FILE.exists():
        raise FileNotFoundError(f"{RUNS_FILE} not found - run the evaluation first.")
    df = pd.read_csv(RUNS_FILE)
    for col in BOOL_COLS:
        df[col] = df[col].astype(str).str.strip().str.lower().isin(["true", "1"])
    return df


def accuracy_table(df):
    """Exact-match accuracy %, rows = model, columns = technique."""
    return (df.pivot_table(index="model", columns="technique",
                           values="exact_ok", aggfunc="mean") * 100).round(1)


def category_table(df):
    """Exact-match accuracy %, rows = category, columns = model."""
    return (df.pivot_table(index="category", columns="model",
                           values="exact_ok", aggfunc="mean") * 100).round(1)


def latency_table(df):
    """Median seconds per call, rows = model, columns = technique."""
    return df.pivot_table(index="model", columns="technique",
                          values="latency_s", aggfunc="median").round(2)


def _grouped_bars(table, title, ylabel, path, ylim=None, rule=None, rule_label="",
                  label_fmt="%.0f"):
    """One group per table row, one coloured bar per table column."""
    groups = [str(g).replace("_", "\n", 1) for g in table.index]
    # Keep our fixed model order first, then anything unexpected.
    series = [c for c in MODEL_COLORS if c in table.columns]
    series += [c for c in table.columns if c not in MODEL_COLORS]

    slot = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(1.5 * len(groups) + 3, 5.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for i, name in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * slot
        # A cell that was never run stays blank. Drawing it as 0 would read
        # as "this model scored zero", which is a different claim entirely.
        column = table[name]
        spots = [(x + offset, v) for x, v in enumerate(column) if pd.notna(v)]
        if not spots:
            continue
        positions, values = zip(*spots)
        bars = ax.bar(positions, values, slot * 0.9, label=name,
                      color=MODEL_COLORS.get(name, SPARE_COLOR), linewidth=0)
        # Every bar is labelled: one hue sits below 3:1 on this surface,
        # so the number, not the colour, carries the value.
        ax.bar_label(bars, fmt=label_fmt, padding=2, fontsize=7.5, color=INK_SOFT)

    if rule is not None:
        # A budget line, not a data series, so it is dashed and unlabelled in
        # the legend -- the text sits on the line itself instead.
        ax.axhline(rule, color=INK_SOFT, linewidth=1.0, linestyle="--", zorder=1)
        ax.annotate(rule_label, xy=(1.0, rule), xycoords=("axes fraction", "data"),
                    xytext=(-4, 3), textcoords="offset points",
                    ha="right", va="bottom", fontsize=8, color=INK_SOFT)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_SOFT)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, length=0)
    # Fixed margins, so the title always sits clear above the legend.
    ax.legend(frameon=False, ncol=max(len(series), 1), loc="lower center",
              bbox_to_anchor=(0.5, 1.01), fontsize=9, labelcolor=INK_SOFT)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.14)
    fig.suptitle(title, x=0.08, y=0.95, ha="left", fontsize=12, color=INK)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def accuracy_chart(df, path=ACCURACY_PNG):
    """Technique on the x axis, one bar per model."""
    table = accuracy_table(df).T  # rows become techniques
    return _grouped_bars(table, "Exact-match accuracy by technique",
                         "accuracy (%)", path, ylim=(0, 105))


def latency_chart(df, path=LATENCY_PNG):
    """Technique on the x axis, one bar per model, with the voice budget drawn on."""
    table = latency_table(df).T  # rows become techniques
    return _grouped_bars(table, "Median latency by technique", "seconds", path,
                         rule=VOICE_LIMIT_S, rule_label="voice limit (2 s)",
                         label_fmt="%.1f")  # whole seconds is too coarse here


def category_chart(df, path=CATEGORY_PNG):
    """Command category on the x axis, one bar per model."""
    return _grouped_bars(category_table(df), "Exact-match accuracy by command category",
                         "accuracy (%)", path, ylim=(0, 105))


def main():
    df = load()
    print("\n=== Exact-match accuracy %  (model x technique) ===")
    print(accuracy_table(df).to_string())
    print("\n=== Exact-match accuracy %  (category x model) ===")
    print(category_table(df).to_string())
    print("\n=== Median latency in seconds  (model x technique) ===")
    print(latency_table(df).to_string())
    print(f"\nSaved {accuracy_chart(df)}")
    print(f"Saved {category_chart(df)}")


if __name__ == "__main__":
    main()
