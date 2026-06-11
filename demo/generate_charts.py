#!/usr/bin/env python3
"""
Generate charts from demo/results.json produced by run_demo.py.

Saves PNGs to demo/charts/:
  1. memory_comparison.png     — float32 vs TurboQuant vs +int8 refine bar chart
  2. recall_at_k.png           — recall@k curve: plain vs rerank rf=4
  3. rerank_improvement.png    — R@1 gain per rerank_factor (bar)
  4. incremental_add.png       — add time vs corpus size (flat vs O(n) projection)
  5. scale_projection.png      — memory at scale (100K → 1B)
  6. bitwidth_tradeoff.png     — memory vs recall for 2-bit / 4-bit / +rerank
"""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DEMO_DIR   = os.path.dirname(__file__)
CHARTS_DIR = os.path.join(DEMO_DIR, "charts")
RESULTS    = os.path.join(DEMO_DIR, "results.json")

os.makedirs(CHARTS_DIR, exist_ok=True)

with open(RESULTS) as f:
    r = json.load(f)

# Shared style
BLUE   = "#2563EB"
ORANGE = "#EA580C"
GREEN  = "#16A34A"
PURPLE = "#7C3AED"
RED    = "#DC2626"
GREY   = "#6B7280"
BG     = "#F8FAFC"

def savefig(name):
    path = os.path.join(CHARTS_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved {path}")

# ── 1. Memory comparison ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
ax.set_facecolor(BG)

labels  = ["Float32\n(raw)", "TurboQuant\n4-bit", "4-bit +\nInt8 RefineStore"]
values  = [r["memory_mb"]["fp32"], r["memory_mb"]["tq4bit"], r["memory_mb"]["tq4bit_i8refine"]]
colors  = [RED, BLUE, PURPLE]
bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=1.5)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
            f"{val:.0f} MB", ha="center", va="bottom", fontsize=12, fontweight="bold")

ratios = ["1×\n(baseline)", f"{r['memory_mb']['compression_vs_fp32']:.1f}× smaller",
          f"{r['memory_mb']['fp32']/r['memory_mb']['tq4bit_i8refine']:.1f}× smaller"]
for bar, label in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
            label, ha="center", va="center", fontsize=10, color="white", fontweight="bold")

ax.set_ylabel("Memory (MB)", fontsize=12)
ax.set_title(f"Memory Footprint — {r['corpus']['n']//1000}K documents, "
             f"dim={r['corpus']['dim']}", fontsize=14, fontweight="bold", pad=15)
ax.set_ylim(0, max(values) * 1.2)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)
savefig("memory_comparison.png")

# ── 2. Recall@k curve ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
ax.set_facecolor(BG)
ks = list(range(1, 11))

ax.plot(ks, r["recall_curves"]["plain"],  color=BLUE,   marker="o", linewidth=2.5,
        markersize=7, label="Plain 4-bit (no rerank)")
ax.plot(ks, r["recall_curves"]["rerank_rf4"], color=GREEN, marker="s", linewidth=2.5,
        markersize=7, label="4-bit + Int8 rerank (rf=4)")

ax.axhline(1.0, color=GREY, linestyle="--", linewidth=1, alpha=0.5, label="Perfect recall (1.0)")
ax.fill_between(ks, r["recall_curves"]["plain"], r["recall_curves"]["rerank_rf4"],
                alpha=0.12, color=GREEN, label="Recall gain from reranking")

ax.set_xlabel("k  (number of results returned)", fontsize=12)
ax.set_ylabel("Recall@k  (fraction finding true top-1 in top-k)", fontsize=12)
ax.set_title("Recall@k Curves — Plain Search vs Cascade Re-ranking", fontsize=14,
             fontweight="bold", pad=15)
ax.set_xlim(1, 10); ax.set_ylim(0.6, 1.02)
ax.set_xticks(ks)
ax.legend(fontsize=10, loc="lower right")
ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)

# Annotate R@1 gap
plain_r1 = r["recall_curves"]["plain"][0]
rerank_r1 = r["recall_curves"]["rerank_rf4"][0]
ax.annotate("", xy=(1, rerank_r1), xytext=(1, plain_r1),
            arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=2))
ax.text(1.15, (plain_r1 + rerank_r1)/2,
        f"+{(rerank_r1-plain_r1)*100:.1f}pp", color=ORANGE, fontsize=10, fontweight="bold")
savefig("recall_at_k.png")

# ── 3. Rerank improvement bar ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
ax.set_facecolor(BG)

rfs    = [0, 1, 2, 4, 8]
r1s    = [r["plain_search"]["r1"]] + [r["rerank"][str(rf)]["r1"] for rf in [1,2,4,8]]
labels = ["Baseline\n(coarse only)"] + [f"rerank\nfactor={rf}" for rf in [1,2,4,8]]
colors = [BLUE] + [GREEN]*4
bars   = ax.bar(labels, r1s, color=colors, width=0.55, edgecolor="white", linewidth=1.5)

for bar, val in zip(bars, r1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.axhline(r["plain_search"]["r1"], color=BLUE, linestyle="--", linewidth=1.2, alpha=0.7)
ax.set_ylabel("Recall@1", fontsize=12)
ax.set_title("R@1 vs Rerank Factor — 4-bit + Int8 RefineStore", fontsize=14,
             fontweight="bold", pad=15)
ax.set_ylim(0.7, 1.02)
ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)

baseline_patch = mpatches.Patch(color=BLUE,  label="Baseline (coarse 4-bit)")
rerank_patch   = mpatches.Patch(color=GREEN, label="With int8 re-rank")
ax.legend(handles=[baseline_patch, rerank_patch], fontsize=10)
savefig("rerank_improvement.png")

# ── 4. Incremental add timing ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
ax.set_facecolor(BG)

add_data   = r["incremental_add_ms"]
ns         = [int(k) for k in add_data.keys()]
actual_ms  = list(add_data.values())

# Skip round 1 (TQ+ calibration overhead) for the projection
ref_n, ref_ms = ns[1], actual_ms[1]
on_projection = [ref_ms * (n / ref_n) for n in ns]  # what O(n) would look like

ax.plot(ns, actual_ms,    color=BLUE,   marker="o", linewidth=2.5, markersize=8,
        label=f"Actual (incremental, O(batch))")
ax.plot(ns, on_projection, color=RED,    linestyle="--", linewidth=2, marker="",
        label=f"Hypothetical O(n) full rebuild")
ax.fill_between(ns, actual_ms, on_projection, alpha=0.1, color=GREEN,
                label="Time saved by incremental packing")

ax.scatter([ns[0]], [actual_ms[0]], color=ORANGE, zorder=5, s=100,
           label=f"Round 1 includes TQ+ calibration")

ax.set_xlabel("Total vectors in index", fontsize=12)
ax.set_ylabel("add() time (ms)", fontsize=12)
ax.set_title("Incremental Add Time — O(batch) vs Hypothetical O(n)",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlim(0, max(ns)*1.05)
ax.set_ylim(0, max(max(on_projection), max(actual_ms)) * 1.25)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
ax.legend(fontsize=10)
ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)
savefig("incremental_add.png")

# ── 5. Scale projection ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=BG)
ax.set_facecolor(BG)

dim = r["corpus"]["dim"]
bw  = r["corpus"]["bit_width"]
ns_scale = [1e5, 1e6, 1e7, 1e8, 1e9]
labels_s  = ["100K\n(demo)", "1M", "10M", "100M", "1B"]

fp32_mbs  = [n * dim * 4 / 1e6 for n in ns_scale]
tq_mbs    = [n * (math.ceil(dim * bw / 8) + 4) / 1e6 for n in ns_scale]
i8_mbs    = [tq_mbs[i] + n * (dim + 4) / 1e6 for i, n in enumerate(ns_scale)]

x = np.arange(len(ns_scale))
w = 0.25
bars1 = ax.bar(x - w, fp32_mbs, width=w, color=RED,    label="Float32 raw",         edgecolor="white")
bars2 = ax.bar(x,     tq_mbs,   width=w, color=BLUE,   label="TurboQuant 4-bit",    edgecolor="white")
bars3 = ax.bar(x + w, i8_mbs,   width=w, color=PURPLE, label="4-bit + Int8 refine", edgecolor="white")

ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(labels_s, fontsize=11)
ax.set_ylabel("Memory (MB, log scale)", fontsize=12)
ax.set_title("Memory at Scale — from 100K to 1 Billion Documents",
             fontsize=14, fontweight="bold", pad=15)
ax.legend(fontsize=10)

# Annotate 1B bar values
for bars in [bars1, bars2, bars3]:
    bar = bars[-1]  # 1B bar
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.4,
            f"{bar.get_height()/1000:.0f} GB", ha="center", fontsize=8, rotation=0)

ax.yaxis.grid(True, alpha=0.3, which="both"); ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)
savefig("scale_projection.png")

# ── 6. Bit-width tradeoff (memory vs recall) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
ax.set_facecolor(BG)

configs = [
    ("Float32\nbaseline",  fp32_mbs[0], 1.000, RED,    "x", 14),
    ("2-bit\nonly",         37.0,        0.000, BLUE,   "o", 10),
    ("4-bit\nonly",         73.6,        r["plain_search"]["r1"], BLUE, "o", 10),
    ("2-bit +\nInt8 rf=4",  183.9,       0.945, PURPLE, "s", 10),
    ("4-bit +\nInt8 rf=4",  r["memory_mb"]["tq4bit_i8refine"],
                             r["rerank"]["4"]["r1"], GREEN, "s", 10),
]

for label, mem, rec, col, mk, ms in configs:
    ax.scatter(mem, rec, color=col, marker=mk, s=ms**2, zorder=5)
    offset = (8, -15) if "4-bit only" in label else (8, 8)
    ax.annotate(label, (mem, rec), xytext=offset, textcoords="offset points",
                fontsize=9, color=col, fontweight="bold")

ax.set_xlabel("Memory (MB)", fontsize=12)
ax.set_ylabel("Recall@1", fontsize=12)
ax.set_title("Memory vs Recall Tradeoff — Bit-width and Rerank Configurations",
             fontsize=14, fontweight="bold", pad=15)
ax.set_ylim(-0.05, 1.08)
ax.xaxis.grid(True, alpha=0.3); ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)

# ideal frontier arrow
ax.annotate("← less memory, more recall\n(top-left is better)",
            xy=(100, 0.95), fontsize=9, color=GREY,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=GREY, alpha=0.8))
savefig("bitwidth_tradeoff.png")

print("\nAll charts saved to demo/charts/")
print("  memory_comparison.png   — 3-way memory bar chart")
print("  recall_at_k.png         — recall@k curves")
print("  rerank_improvement.png  — R@1 by rerank_factor")
print("  incremental_add.png     — O(batch) vs O(n) add time")
print("  scale_projection.png    — memory at 100K→1B scale")
print("  bitwidth_tradeoff.png   — memory vs recall Pareto")
