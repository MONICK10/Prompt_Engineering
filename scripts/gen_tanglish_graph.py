"""
Generate data/graphs/tanglish_accuracy.png
Grayscale, 300 dpi, IEEE single-column width, hatch patterns, value labels.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'graphs', 'tanglish_accuracy.png')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        7,
    'axes.titlesize':   7,
    'axes.labelsize':   7,
    'xtick.labelsize':  6.5,
    'ytick.labelsize':  6.5,
    'axes.linewidth':   0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'pdf.fonttype': 42,
    'ps.fonttype':  42,
})

DPI  = 300
W_IN = 3.5
EDGE = 'black'
EW   = 0.5

systems = ['Rule-based', 'XLM-R\nZero-shot', 'Groq LLM', 'BiLSTM', 'DocEditBERT\n-1000']
values  = [0.0,          0.0,                  0.6,        8.0,      100.0]
fills   = ['0.92',       '0.78',               '0.62',     '0.42',   '0.12']
hatches = ['///',        '\\\\\\',             '...',      'xxx',    '---']

fig, ax = plt.subplots(figsize=(W_IN, 2.8))

bars = ax.bar(
    systems, values,
    color=fills, hatch=hatches,
    edgecolor=EDGE, linewidth=EW,
    width=0.55, zorder=3,
)

for bar, val in zip(bars, values):
    label = f'{val:.1f}%'
    y = bar.get_height() + 1.5
    ax.text(
        bar.get_x() + bar.get_width() / 2, y,
        label, ha='center', va='bottom',
        fontsize=6.5, fontweight='bold',
    )

ax.set_ylim(0, 115)
ax.set_ylabel('Accuracy (%)')
ax.set_title(
    'Figure 6: System Accuracy on DocEdit-Tanglish-500 (500 commands)',
    pad=5, fontsize=6.5,
)
ax.yaxis.grid(True, linewidth=0.4, linestyle=':', color='0.6', zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout(pad=0.4)
fig.savefig(OUT, dpi=DPI, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {OUT}')
