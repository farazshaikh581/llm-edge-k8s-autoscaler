"""p90 latency against running replicas on the real cluster (cpu_bursty trace).

One point per controller configuration: sixteen frozen-LLM configurations
(four models, four prompt families) and the four baselines.

  x  mean running replica count over the run (linear).
  y  p90 request latency in ms, measured by the load generator over every
     workload request, taken as the 90th percentile of the per-interval p90
     over the 120-step run with three repetitions pooled (linear).

The dashed line and the shaded band mark the 200 ms p90 target. Filled markers
meet it on at least 99 percent of intervals, open markers do not. A dashed grey
ring marks configurations whose model inference was so slow that the 60-second
control loop fell behind the trace (wall clock over 150 minutes for a 120-step,
120-minute run, or a median inference call over 10 seconds); those operating
points come from a controller that was decoupled from the workload it was
meant to track.

Reads  results/real/richer/cpu_bursty/rep*/k8s_cpu_<config>.csv   (replicas, timing)
       results/real/richer/cpu_bursty/rep*/load_cpu_<config>.csv  (workload latency)
Writes business_case_real/fig_latency_replicas_real_cpu_bursty.{pdf,png}

Run from anywhere:  ./venv/bin/python plot_latency_vs_replicas.py
"""
import csv
import glob
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(ROOT, "results", "real", "richer", "cpu_bursty")
OUT = os.path.join(ROOT, "business_case_real",
                   "fig_latency_replicas_real_cpu_bursty.{ext}")
REPS = ("rep1", "rep2", "rep3")

SLA_MS = 200.0        # p90 latency target
SLA_BAR = 99.0        # percent of intervals under SLA_MS to count as passing
TAIL_Q = 90           # percentile of interval latency reported per configuration
Y_LIM = (0.0, 244.0)
Y_CLIP = 232.0        # latency above this is drawn here with a text label
DESYNC_WALL_MIN = 150.0    # a 120-step, 120-minute run over this ran behind
DESYNC_CALL_MS = 10_000.0  # median inference call over this stalls the loop

# Okabe-Ito, colour-blind safe. One colour per prompt family.
FAM_COL = {"zero_shot": "#0072B2", "domain": "#E69F00",
           "cot": "#009E73", "history_5": "#CC79A7"}
FAM_LBL = {"zero_shot": "zero-shot", "domain": "domain",
           "cot": "chain-of-thought", "history_5": "history"}
MOD_MK = {"llama-8b": "o", "llama-70b": "s",
          "mistral-small4": "D", "gpt-oss-120b": "^"}
MOD_LBL = {"llama-8b": "Llama-8B", "llama-70b": "Llama-70B",
           "mistral-small4": "Mistral-Small", "gpt-oss-120b": "GPT-OSS-120B"}
BASE_MK = {"hpa": "X", "keda": "X", "rl-dqn": "P", "rl-ppo": "P"}
BASE_LBL = {"hpa": "HPA", "keda": "KEDA", "rl-dqn": "DQN", "rl-ppo": "PPO"}
BASE_COL = "0.45"

# free text with a leader line:  name -> (tx, ty, ha, label, target_xy)
LEADERS = {
    "cot": (2.7, 178, "left", "GPT-OSS-120B / CoT:  p90 ≈ 5.0 s", (4.5, 228)),
    "fast": (2.7, 47, "left", "GPT-OSS-120B: zero-shot, history", (1.3, 25)),
    "l8b": (9.4, 174, "left", "Llama-8B: domain, CoT, history", (12.1, 134)),
}
# small offset labels:  key -> (dx, dy in points, ha)
OFFSETS = {
    "hpa|baseline": (-7, 6, "right"),
    "keda|baseline": (3, 15, "left"),
    "rl-dqn|baseline": (0, -12, "center"),
    "rl-ppo|baseline": (0, -12, "center"),
}


def parse_name(path):
    stem = os.path.basename(path)[len("k8s_cpu_"):-len(".csv")]
    if stem.endswith("_baseline"):
        return stem[:-len("_baseline")], "baseline"
    for m in MOD_MK:
        if stem.startswith(m + "_"):
            return m, stem[len(m) + 1:]
    return stem, ""


def fcol(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        if v in ("", "nan", "None"):
            continue
        try:
            out.append(float(v))
        except ValueError:
            pass
    return out


def load():
    acc = {}
    for kf in sorted(glob.glob(os.path.join(DIR, "rep1", "k8s_cpu_*.csv"))):
        model, variant = parse_name(kf)
        name = f"{model}_{variant}" if variant != "baseline" else f"{model}_baseline"
        key = f"{model}|{variant}"
        rep_list, lat_list, sla_list, call_list, walls = [], [], [], [], []
        for rep in REPS:
            kp = os.path.join(DIR, rep, f"k8s_cpu_{name}.csv")
            lp = os.path.join(DIR, rep, f"load_cpu_{name}.csv")
            if os.path.exists(kp):
                kr = list(csv.DictReader(open(kp)))
                rep_list += fcol(kr, "ready_replicas")
                call_list += fcol(kr, "llm_latency_ms")
                ts = [datetime.fromisoformat(r["timestamp"]) for r in kr]
                walls.append((ts[-1] - ts[0]).total_seconds() / 60.0)
            if os.path.exists(lp):
                lr = list(csv.DictReader(open(lp)))
                xl = fcol(lr, "latency_p90_ms")
                lat_list += xl
                sla_list += [1.0 if v < SLA_MS else 0.0 for v in xl]
        lat = np.asarray(lat_list)
        calls = [c for c in call_list if c > 0]
        wall = float(np.mean(walls)) if walls else 0.0
        call_med = float(np.median(calls)) if calls else 0.0
        acc[key] = {
            "key": key, "model": model, "variant": variant,
            "x": float(np.mean(rep_list)),
            "y": float(np.percentile(lat, TAIL_Q)),
            "sla": 100.0 * float(np.mean(sla_list)),
            "desync": wall > DESYNC_WALL_MIN or call_med > DESYNC_CALL_MS,
        }
    return list(acc.values())


def label_of(d):
    if d["variant"] == "baseline":
        return BASE_LBL[d["model"]]
    return f'{MOD_LBL[d["model"]]} / {FAM_LBL[d["variant"]]}'


def main():
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "axes.linewidth": 0.8,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "pdf.compression": 9, "savefig.bbox": "tight"})
    rows = load()
    by = {d["key"]: d for d in rows}

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="0.9", linewidth=0.6)
    ax.grid(True, axis="x", color="0.955", linewidth=0.5)

    ax.axhspan(SLA_MS, Y_LIM[1], color="#D55E00", alpha=0.07, zorder=0)
    ax.axhline(SLA_MS, color="#D55E00", linestyle="--", linewidth=1.1, zorder=2)
    ax.text(13.8, SLA_MS + 5, "200 ms p90 target", color="#B24500",
            fontsize=7.5, ha="right", va="bottom", style="italic")

    for d in rows:
        base = d["variant"] == "baseline"
        yv = min(d["y"], Y_CLIP)
        passes = d["sla"] >= SLA_BAR
        mk = BASE_MK[d["model"]] if base else MOD_MK[d["model"]]
        col = BASE_COL if base else FAM_COL[d["variant"]]
        size = 42
        if d.get("desync"):
            ax.scatter(d["x"], yv, marker="o", s=size * 2.5, facecolors="none",
                       edgecolors="0.55", linewidths=0.9, linestyle=(0, (2, 2)),
                       zorder=3, clip_on=False)
        ax.scatter(d["x"], yv, marker=mk, s=size,
                   facecolor=col if passes else "white",
                   edgecolor=col, linewidth=1.5, zorder=4, clip_on=False)
        if passes and not base:
            ax.scatter(d["x"], yv, marker=mk, s=size, facecolor="none",
                       edgecolor="white", linewidth=0.5, zorder=5, clip_on=False)
        if d["y"] > Y_CLIP:
            ax.annotate("", xy=(d["x"], Y_CLIP + 6),
                        xytext=(d["x"], Y_CLIP - 20),
                        arrowprops=dict(arrowstyle="-|>", color=col,
                                        linewidth=1.3), zorder=5)

    ax.text(9.0, 96, "every synced controller: p90 $\\approx$ 130 ms",
            fontsize=7, style="italic", color="0.4", ha="center", zorder=6)

    for tx, ty, ha, lab, xy in LEADERS.values():
        ax.annotate("", xy=xy, xytext=(tx, ty), textcoords="data",
                    arrowprops=dict(arrowstyle="-", color="0.55", linewidth=0.6,
                                    shrinkA=0, shrinkB=4), zorder=3)
        ax.text(tx, ty, lab, fontsize=7, ha=ha, va="center", zorder=6)
    for key, (dx, dy, ha) in OFFSETS.items():
        d = by[key]
        col = BASE_COL if d["variant"] == "baseline" else FAM_COL[d["variant"]]
        ax.annotate(label_of(d), (d["x"], min(d["y"], Y_CLIP)),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=7, color=col, ha=ha, va="center")

    ax.set_xlim(0, 14)
    ax.set_ylim(*Y_LIM)
    ax.set_xticks(range(0, 15, 2))
    ax.set_yticks(range(0, 241, 40))
    ax.set_xlabel("Mean running replicas", fontsize=8.5)
    ax.set_ylabel("p90 request latency (ms)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)

    fam_h = [Line2D([0], [0], marker="o", linestyle="", markersize=6,
                    markerfacecolor=FAM_COL[v], markeredgecolor="white",
                    markeredgewidth=0.5, label=FAM_LBL[v]) for v in FAM_COL]
    mod_h = [Line2D([0], [0], marker=MOD_MK[m], linestyle="", markersize=6,
                    markerfacecolor="0.4", markeredgecolor="0.4",
                    label=MOD_LBL[m]) for m in MOD_MK]
    spacer = Line2D([0], [0], linestyle="", marker="", label=" ")
    extra_h = [
        Line2D([0], [0], marker="X", linestyle="", markersize=6,
               markerfacecolor=BASE_COL, markeredgecolor=BASE_COL,
               label="rule-based (HPA, KEDA)"),
        Line2D([0], [0], marker="P", linestyle="", markersize=6,
               markerfacecolor=BASE_COL, markeredgecolor=BASE_COL,
               label="trained RL (DQN, PPO)"),
        Line2D([0], [0], marker="o", linestyle=(0, (2, 2)), markersize=11,
               markerfacecolor="none", markeredgecolor="0.5",
               markeredgewidth=0.9, label="control loop fell behind the trace"),
    ]
    # interleave so the column-major legend reads models on row 1, rest on row 2
    combo = [mod_h[0], extra_h[0], mod_h[1], extra_h[1],
             mod_h[2], extra_h[2], mod_h[3], spacer]

    fig.legend(handles=fam_h, title="prompt encoding", loc="lower center",
               bbox_to_anchor=(0.5, 0.12), ncol=4, fontsize=7,
               title_fontsize=7.5, frameon=False, handletextpad=0.4,
               columnspacing=1.7)
    fig.legend(handles=combo, loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=4, fontsize=6.8, frameon=False, handletextpad=0.4,
               columnspacing=1.6)
    fig.text(0.5, 0.0,
             "Filled marker meets the 200 ms target on at least 99% of "
             "intervals; open marker misses it.",
             ha="center", fontsize=6.8, color="0.3")

    fig.tight_layout(rect=(0, 0.19, 1, 1))
    fig.savefig(OUT.format(ext="pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT.format(ext="png"), dpi=600, bbox_inches="tight",
                pad_inches=0.03)
    for ext in ("pdf", "png"):
        print("wrote", OUT.format(ext=ext))


if __name__ == "__main__":
    main()
