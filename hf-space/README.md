---
title: turbovec — Quantized Vector Search
emoji: 🧭
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# turbovec — quantized vector search (live demo)

Interactive Hugging Face Space for [`turbovec`](https://github.com/aabhimittal/turbovec-test):
a data-oblivious **2–4-bit vector quantizer + SIMD approximate-nearest-neighbour
engine** for embeddings.

Pick a topic and tune **bit width**, **refine mode** (`int8` / **`float16`** / `float32`)
and **rerank factor** to see the memory ↔ recall trade-off, measured live from the real
Rust engine (built from source in the `Dockerfile`).

## What this fork adds
- **`float16` refine mode** — half the size of a float32 refine store, near-exact re-rank.
- **Cascade re-ranking** — coarse 2–4-bit scan → exact re-score of the shortlist.
- **Incremental (O-batch) packing** and **WebAssembly bindings** (see the repo).

## Deploy your own copy
```bash
# 1. Create a new Docker Space on huggingface.co, then:
git clone https://huggingface.co/spaces/<your-username>/turbovec-demo
cp hf-space/* turbovec-demo/          # from this repo
cd turbovec-demo && git add . && git commit -m "turbovec demo" && git push
```
The Space builds turbovec from source on first deploy (a few minutes), then serves
the Gradio app on port 7860.
