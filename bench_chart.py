#!/usr/bin/env python3
"""
bench_chart.py — generate the DGX Spark benchmark charts.

Produces two PNGs in results/:
  - comparison.png        : vLLM vs SGLang on the shared bench_serving harness
  - sglang_workloads.png  : SGLang per-workload spread (bench.sh methodology)

Run:  python3 bench_chart.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VLLM_COLOR   = "#898781"   # neutral gray
SGLANG_COLOR = "#2a78d6"   # blue
PEAK_COLOR   = "#1baf7a"   # green

# ---------------------------------------------------------------------------
# Chart 1: vLLM vs SGLang on the identical bench_serving harness
# ---------------------------------------------------------------------------
workloads = ["Random 512/512\n(single)", "ShareGPT chat\n(single)", "ShareGPT\n(concurrent x4)"]
vllm   = [20.0, 18.0, 61.4]
sglang = [26.5, 23.1, 59.5]

x = np.arange(len(workloads))
width = 0.38
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
b1 = ax.bar(x - width/2, vllm,   width, label="vLLM + MTP",     color=VLLM_COLOR)
b2 = ax.bar(x + width/2, sglang, width, label="SGLang + DSpark", color=SGLANG_COLOR)
for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{r.get_height():.1f}", xy=(r.get_x()+r.get_width()/2, r.get_height()),
                    xytext=(0,3), textcoords="offset points", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Output throughput (tokens/sec)", fontsize=11)
ax.set_title("Qwen3.8-27B (NVFP4) on DGX Spark (GB10)\nvLLM + MTP  vs  SGLang + DSpark - identical bench_serving harness",
             fontsize=12, pad=14)
ax.set_xticks(x); ax.set_xticklabels(workloads, fontsize=10)
ax.legend(frameon=False, fontsize=10, loc="upper left")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_ylim(0, max(max(vllm), max(sglang))*1.18)
ax.grid(axis="y", color="#e1e0d9", linewidth=0.8); ax.set_axisbelow(True)
fig.text(0.5, -0.02,
         "General/prose regime (random + chat). SGLang ~28-33% faster single-stream; concurrent within ~3%. "
         "Two matched GB10 units (CUDA 13.2 / 13.0).",
         ha="center", fontsize=8, color="#52514e")
plt.tight_layout()
plt.savefig("results/comparison.png", bbox_inches="tight", facecolor="white")
print("wrote results/comparison.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Chart 2: SGLang + DSpark per-workload spread (bench.sh methodology)
# ---------------------------------------------------------------------------
labels = ["Reasoning\n(greedy)", "Math peak\n(temp 0.6)", "Code\n(greedy)",
          "Free prose\n(worst case)"]
vals   = [44.5, 52.9, 41.8, 19.6]
colors = [SGLANG_COLOR, PEAK_COLOR, SGLANG_COLOR, VLLM_COLOR]

fig2, ax2 = plt.subplots(figsize=(9, 5.0), dpi=150)
bars = ax2.bar(labels, vals, color=colors, width=0.6)
for r in bars:
    ax2.annotate(f"{r.get_height():.1f}", xy=(r.get_x()+r.get_width()/2, r.get_height()),
                 xytext=(0,3), textcoords="offset points", ha="center", va="bottom", fontsize=11)
median_line = 41.0
ax2.axhline(median_line, ls="--", lw=1, color="#52514e")
ax2.annotate(f"greedy median {median_line:.1f}", xy=(3.35, median_line),
             xytext=(0,4), textcoords="offset points", ha="right", fontsize=9, color="#52514e")
ax2.set_ylabel("Decode throughput (tokens/sec)", fontsize=11)
ax2.set_title("SGLang + DSpark - throughput is workload-dependent\nQwen3.8-27B (NVFP4) on DGX Spark (GB10), bench.sh methodology",
              fontsize=12, pad=14)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
ax2.set_ylim(0, max(vals)*1.18)
ax2.grid(axis="y", color="#e1e0d9", linewidth=0.8); ax2.set_axisbelow(True)
fig2.text(0.5, -0.02,
          "Acceptance length drives everything: math/code accept ~3.3-5.6 tokens/step, free prose ~1.5-2.2. "
          "Methodology & probes: hasso5703/dgx-spark-qwen38.",
          ha="center", fontsize=8, color="#52514e")
plt.tight_layout()
plt.savefig("results/sglang_workloads.png", bbox_inches="tight", facecolor="white")
print("wrote results/sglang_workloads.png")
