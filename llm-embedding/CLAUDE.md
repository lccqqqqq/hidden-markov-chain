# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **theoretical research project** on deriving word/token embeddings from co-occurrence statistics using statistical mechanics. The core idea: model word co-occurrence as a Boltzmann distribution, show that large context windows yield Gaussian fluctuations (CLT), and that low-rank factorization of the long-run covariance matrix recovers embedding geometry (dot products encode PMI).

The project extends this "bag-of-words" baseline by introducing sequential (word-order) dependence via perturbation theory, exploring three approaches:
- **Nearest-neighbor interaction** with CJT/2PI effective action (`local_interaction_note.tex`)
- **Dressed embeddings** as bigram-weighted averages (`sequential_dependence_note.tex`)
- **Mallows-penalized models** using Kendall tau distance (`mallows_note.tex`)

A parallel analysis thread studies **Hidden Markov Models** as a tractable test case for these ideas (`notes/hmm-analysis/`).

## Repository Structure

- `report/` — LaTeX notes and compiled PDFs. Each `.tex` file is a self-contained note with its own preamble (no shared style files or bibliography). The main overview document is `report.tex`.
- `doc/` — Markdown research notes reconstructed from conversations (Gaussian limit derivation, scaling laws analysis, data sparsity).
- `notes/` — Reference PDFs, handwritten page scans (`pages/`), and the HMM analysis series (`hmm-analysis/`, 8 numbered markdown notes).
- `model.py` — Currently empty.

## Building LaTeX

Each `.tex` file in `report/` is compiled independently with `pdflatex`:
```bash
cd report && pdflatex -interaction=nonstopmode report.tex
```
No BibTeX step is needed (references are defined inline with `\bibitem`). No shared `.bib` file exists.

## LaTeX Conventions

- Custom commands are defined per-file (e.g., `\vw`, `\ee`, `\KL`, `\PMI`, `\BoW`, `\Tr`).
- User comments to address are marked with `\lcq{...}` (renders as blue bold text). When editing, address these comments and summarize how they were resolved.
- Claude's inserted comments use `\claude{...}` (renders in red sans-serif).
- Superseded text is marked with `\old{...}` (renders in gray).

## MCP Servers

- **Zotero** is configured for literature reference lookup.
