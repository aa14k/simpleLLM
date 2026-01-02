# Clean GPT Implementations

This repository contains **minimal, readable implementations of GPT-style transformers**, designed for learning, experimentation, and interview preparation.  
Each subdirectory implements a distinct architectural variant while keeping the training pipeline simple and transparent.

The goal is not maximum performance. The goal is **clarity, correctness, and modularity**.

---

## Repository Structure

```
.
├── simpleGPT/
│   └── Minimal GPT (GPT-2 style)
├── moeGPT/
│   └── GPT with Mixture of Experts
├── modernGPT/
│   └── GPT with modern components (RoPE, SwiGLU, Muon, etc.)
├── modernmoeGPT/
│   └── Modern GPT + Mixture of Experts
```

---

## Implementations

### simpleGPT
A clean GPT-2–style decoder-only transformer.  
Baseline reference implementation.

### moeGPT
Extends `simpleGPT` with Mixture-of-Experts layers.

### modernGPT
A modernized GPT variant using:
- RoPE positional embeddings
- SwiGLU feed-forward layers
- KV caching
- Muon optimizer
- RMSNorm and other modern design choices

### modernmoeGPT
Combines modern GPT components with Mixture-of-Experts.

---

## Dataset

All models train on **TinyStories**, a lightweight language-modeling dataset suitable for fast iteration.

### Download TinyStories

```bash
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt?download=true -O TinyStories-train.txt
```

---

## Installation

Install dependencies using `uv`:

```bash
uv pip install torch pandas tiktoken numpy grain tqdm
```

---

## Training

1. **Set the dataset path**  
   In each `train.py`, ensure `data_path` points to `TinyStories-train.txt`, regardless of subdirectory.

2. **Run training**

```bash
python sub_directory/train.py
```

Replace `sub_directory` with one of:
- `simpleGPT`
- `moeGPT`
- `modernGPT`
- `modernmoeGPT`

---

## Design Philosophy

- Minimal abstractions
- Easy to modify
- Minimal lines of code

This repo is intentionally **not a framework**.  
It is a collection of **clean reference implementations** you can read end-to-end.

---

## Intended Use

- Interview preparation
- Architecture comparison
- Rapid prototyping
- Teaching and self-study
- Sanity-checking transformer fundamentals

---

## Notes

- These implementations prioritize readability over raw throughput.
- Ideal for small-scale experiments and conceptual clarity.
- Easy to extend with custom attention, routing, or optimization logic.
