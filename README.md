# LLM Edge Kubernetes Autoscaler

Code and results for "Can a Language Model Drive Kubernetes Autoscaling at
the Edge? Lessons Learned from a Physical Testbed." This repository holds
only the artifacts the paper cites: the controller, the two plot scripts
behind Figures 2 and 3, and the real-cluster result files behind Table I
and those figures.

## Contents

- `k8s_autoscaler.py` — the controller: prompt construction for the four
  encodings, the HPA/KEDA/DQN/PPO baselines, and the guardrail layer.
- `llm_autoscaler/run_richer_real.sh` — runs one controller against one
  trace on the physical cluster.
- `plot_latency_vs_replicas.py` — builds Figure 2 (tail latency vs. mean
  replicas).
- `llm_autoscaler/plot_cost_stability_combined.py` — builds Figure 3 (cost
  vs. scaling actions, both traces).
- `results/real/richer/{cpu_bursty,wiki_diurnal}/rep{1,2,3}/` — raw
  per-interval CSVs from the three repetitions of each trace on the
  physical cluster.
- `results/real/results_richer_*_{business_case,percentiles,summary}.csv`
  — aggregated per-controller results, the direct source for Table I.
- `business_case_real/` — the two figures as generated (PDF and PNG).

## Reproducing the figures

```
pip install -r requirements.txt
python plot_latency_vs_replicas.py
python llm_autoscaler/plot_cost_stability_combined.py
```

Both scripts read the CSVs under `results/real/`.

## Not included

Simulation results, earlier (k8s_v1/k8s_v2) real-cluster runs, and the raw
Alibaba 2018 cluster trace are outside this paper's scope and not included
here. Provider API keys are never checked into this repository; set them
as environment variables before running `run_richer_real.sh`.
