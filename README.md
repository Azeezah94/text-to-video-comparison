# Text-to-Video Generation: A Comparative Study
**Author:** Azeezat Akinola | [LinkedIn](https://www.linkedin.com/in/azeezat-akinola-710b73113) | [Portfolio](https://Azeezah94.github.io)

Comparative evaluation of three text-to-video generation models **ModelScope**, **Text2Video-Zero**, and **Motion Consistency Model** using CLIP score and rigorous statistical testing.

## Methodology
Matches the published thesis methodology exactly:
- Generate video frames from identical prompts across all 3 models
- Score text-video alignment using **CLIP similarity**
- Test for statistical significance with **one-way ANOVA** and **pairwise t-tests**
- Check **homogeneity of variance** via Levene's test before choosing equal/unequal variance t-tests

## Results (from thesis)
| Finding | Result |
|---|---|
| Best performing model | ModelScope (highest mean CLIP score) |
| Statistical significance | Differences not statistically significant (p > 0.05) |
| Key insight | All 3 models produce comparable quality; ModelScope has a slight edge |

## Tech Stack
Python, PyTorch, HuggingFace Diffusers, CLIP, SciPy (statistical testing), Pandas

## Setup
```bash
git clone https://github.com/Azeezah94/text-to-video-comparison
cd text-to-video-comparison
pip install -r requirements.txt
python src/comparison_study.py
```

## Publication
Part of: *"Evaluating Multimodal AI Systems: A Comparative Analysis of Large Language Model-Based Models for Text, Image, and Video Generation"*
M.S. Thesis, Georgia Southern University, 2025 — [Read the full thesis](https://digitalcommons.georgiasouthern.edu/etd/2944/)
