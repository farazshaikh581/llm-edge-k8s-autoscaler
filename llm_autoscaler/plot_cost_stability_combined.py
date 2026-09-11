"""Combined cost-vs-stability figure for the real cluster, both traces.

Reads results/real/results_richer_{trace}_business_case.csv and writes
business_case_real/fig_cost_stability_real_combined.{pdf,png} (repo root).

One display item, two panels (both workloads). x = annualized cost (data plane
plus control plane plus API tokens at list price), y = scaling actions per run.
Marker shape = controller family, fill = meets the 99 percent latency target,
black ring = one of the four deployable candidates (SLA at or above 99 and few
scaling actions on both traces).

Run from anywhere:  python llm_autoscaler/plot_cost_stability_combined.py
"""
import csv
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "real", "results_richer_{trace}_business_case.csv")
OUT = os.path.join(ROOT, "business_case_real", "fig_cost_stability_real_combined.{ext}")

SLA_BAR = 99.0  # percent of steps under the 200 ms p90 target

COL = {"LLM": "#0072B2", "RULE": "#D55E00", "RL": "#009E73"}  # Okabe-Ito, colsafe
SHAPE = {"LLM": "o", "RULE": "s", "RL": "^"}
FAMILY = {"hpa": "RULE", "keda": "RULE", "rl-dqn": "RL", "rl-ppo": "RL"}

PRETTY_V = {"zero_shot": "zero-shot", "domain": "domain", "cot": "CoT", "history_5": "history-5"}
PRETTY_M = {"gpt-oss-120b": "GPT-OSS-120B", "llama-70b": "Llama-70B",
            "llama-8b": "Llama-8B", "mistral-small4": "Mistral-Small"}
BASE_LABEL = {"hpa": "HPA", "keda": "KEDA", "rl-dqn": "DQN", "rl-ppo": "PPO"}

CANDIDATES = ["gpt-oss-120b|zero_shot", "mistral-small4|zero_shot",
              "mistral-small4|history_5", "llama-70b|zero_shot"]

# single-leader text callouts: key -> (text_x, text_y, ha)
CALLOUTS = {
    "cpu_bursty": {
        "gpt-oss-120b|cot": (640, 92, "left"),
        "hpa|baseline": (215, 46, "left"),
        "keda|baseline": (930, 74, "left"),
        "rl-dqn|baseline": (700, 9.0, "left"),
        "rl-ppo|baseline": (775, 0.55, "left"),
    },
    "wiki_diurnal": {
        "gpt-oss-120b|cot": (680, 92, "left"),
        "hpa|baseline": (120, 3.0, "left"),
        "rl-dqn|baseline": (900, 19, "left"),
        "keda|baseline": (1055, 46, "left"),
        "rl-ppo|baseline": (775, 3.8, "left"),
    },
}
# candidate names and costs live in the figure caption, not on the plot.


def load(trace):
    rows = []
    with open(CSV.format(trace=trace)) as fh:
        for r in csv.DictReader(fh):
            fam = FAMILY.get(r["model"], "LLM")
            rows.append({
                "model": r["model"], "variant": r["variant"], "fam": fam,
                "cost": float(r["cost_usd"]), "scales": float(r["scales"]),
                "sla": float(r["lat_sla"]), "key": f"{r['model']}|{r['variant']}",
            })
    return rows


def _leader(ax, xy, xytext):
    ax.annotate("", xy=xy, xytext=xytext, textcoords="data",
                arrowprops=dict(arrowstyle="-", color="0.55", linewidth=0.6,
                                shrinkA=0, shrinkB=3), zorder=4)


def draw(ax, rows, trace, title):
    ax.set_axisbelow(True)
    by_key = {d["key"]: d for d in rows}

    for d in rows:
        pass_sla = d["sla"] >= SLA_BAR
        ax.scatter(d["cost"], d["scales"], marker=SHAPE[d["fam"]],
                   s=46 if d["fam"] == "LLM" else 68,
                   facecolor=COL[d["fam"]] if pass_sla else "white",
                   edgecolor=COL[d["fam"]], linewidth=1.4, zorder=3, clip_on=False)
    for key in CANDIDATES:
        d = by_key[key]
        ax.scatter(d["cost"], d["scales"], marker="o", s=132, facecolor="none",
                   edgecolor="black", linewidth=1.2, zorder=5, clip_on=False)

    for key, (tx, ty, ha) in CALLOUTS[trace].items():
        d = by_key[key]
        lab = (BASE_LABEL[d["model"]] if d["fam"] != "LLM"
               else f"{PRETTY_M[d['model']]} / {PRETTY_V[d['variant']]}")
        ax.text(tx, ty, lab, fontsize=7, ha=ha, va="center", zorder=6)
        _leader(ax, (d["cost"], d["scales"]), (tx, ty))

    ax.set_yscale("symlog", linthresh=8, linscale=1.4)
    ax.set_yticks([0, 2, 4, 6, 8, 20, 40, 80])
    ax.set_yticklabels(["0", "2", "4", "6", "8", "20", "40", "80"])
    ax.set_ylim(-1.5, 135)
    ax.set_xlim(0, 1300)
    ax.set_xticks([0, 300, 600, 900, 1200])
    ax.set_xlabel("Annualized cost (USD per service-year)", fontsize=8)
    ax.set_title(title, fontsize=8.5, pad=6)
    ax.grid(axis="y", color="0.86", linewidth=0.6)
    ax.tick_params(labelsize=7.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.9), sharey=True)
    draw(axes[0], load("cpu_bursty"), "cpu_bursty", "(a) CPU-bursty trace")
    draw(axes[1], load("wiki_diurnal"), "wiki_diurnal", "(b) Wikipedia diurnal trace")
    axes[0].set_ylabel("Scaling actions per run (120 steps)", fontsize=8)

    def mk(marker, fc, ec, mew=1.4, ms=7.5):
        return Line2D([], [], marker=marker, linestyle="none", markersize=ms,
                      markerfacecolor=fc, markeredgecolor=ec, markeredgewidth=mew)
    handles = [mk("o", COL["LLM"], COL["LLM"]), mk("s", COL["RULE"], COL["RULE"]),
               mk("^", COL["RL"], COL["RL"]), mk("o", "white", "0.35"),
               mk("o", "none", "black", ms=10)]
    labels = ["Frozen LLM", "HPA / KEDA", "DQN / PPO",
              "open: misses 99% SLA", "ring: deployable candidate"]
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02), handletextpad=0.4,
               columnspacing=1.3)

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(OUT.format(ext=ext), dpi=200, bbox_inches="tight")
    print("wrote " + OUT.format(ext="{pdf,png}"))


if __name__ == "__main__":
    main()
