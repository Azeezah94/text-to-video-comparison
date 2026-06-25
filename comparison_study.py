"""
Text-to-Video Generation: A Comparative Study
Author: Azeezat Akinola
Thesis: "Evaluating Multimodal AI Systems" — Georgia Southern University, 2025
        https://digitalcommons.georgiasouthern.edu/etd/2944/

Compares ModelScope, Text2Video-Zero, and Motion Consistency Model on
text-to-video generation quality, evaluated via CLIP score with statistical
significance testing (t-tests, homogeneity of variance).
"""

import time
import numpy as np
import pandas as pd
from scipy import stats
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIGS = {
    "ModelScope":         "damo-vilab/text-to-video-ms-1.7b",
    "Text2Video-Zero":    "runwayml/stable-diffusion-v1-5",   # used with zero-shot video adapter
    "MotionConsistency":  "ali-vilab/text-to-video-ms-1.7b",  # consistency-tuned variant
}


# ── CLIP-based Evaluation ──────────────────────────────────────────────────────
class CLIPScorer:
    """Computes CLIP similarity between a text prompt and generated video frames."""

    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @torch.no_grad()
    def score(self, prompt: str, frames: list[Image.Image]) -> list[float]:
        """Return per-frame CLIP similarity scores for a prompt against video frames."""
        scores = []
        for frame in frames:
            inputs = self.processor(text=[prompt], images=frame, return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            logits = self.model(**inputs).logits_per_image
            scores.append(float(logits.softmax(dim=1)[0][0]))
        return scores


# ── Generation Wrapper (per model) ────────────────────────────────────────────
def generate_video(model_name: str, prompt: str, num_frames: int = 8) -> dict:
    """
    Generates video frames for a given model and prompt.
    In a full run this loads the actual diffusion pipeline per MODEL_CONFIGS;
    here it's structured so each model can be swapped in independently.
    """
    from diffusers import DiffusionPipeline

    start = time.time()
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_CONFIGS[model_name],
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)
    pipe.enable_attention_slicing()

    output = pipe(prompt, num_frames=num_frames, num_inference_steps=25)
    elapsed = time.time() - start

    return {
        "model": model_name,
        "prompt": prompt,
        "frames": output.frames[0],
        "gen_time_sec": round(elapsed, 2),
    }


# ── Comparative Study Runner ───────────────────────────────────────────────────
def run_comparison(prompts: list[str], models: list[str] = None) -> pd.DataFrame:
    """
    Runs all models on all prompts, scores each with CLIP, and returns a
    long-format DataFrame ready for statistical testing.
    """
    models = models or list(MODEL_CONFIGS.keys())
    scorer = CLIPScorer()
    records = []

    for model_name in models:
        for prompt in prompts:
            result = generate_video(model_name, prompt)
            clip_scores = scorer.score(prompt, result["frames"])
            for frame_idx, score in enumerate(clip_scores):
                records.append({
                    "model": model_name,
                    "prompt": prompt,
                    "frame": frame_idx,
                    "clip_score": score,
                    "gen_time_sec": result["gen_time_sec"],
                })

    return pd.DataFrame(records)


# ── Statistical Analysis ───────────────────────────────────────────────────────
def statistical_comparison(df: pd.DataFrame) -> dict:
    """
    Runs the same statistical tests used in the thesis:
    - One-way ANOVA / pairwise t-tests across models
    - Levene's test for homogeneity of variance
    """
    models = df["model"].unique()
    groups = [df[df["model"] == m]["clip_score"].values for m in models]

    # ANOVA across all models
    f_stat, anova_p = stats.f_oneway(*groups)

    # Levene's test for homogeneity of variance (used in thesis)
    levene_stat, levene_p = stats.levene(*groups)

    # Pairwise t-tests
    pairwise = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            t_stat, p_val = stats.ttest_ind(groups[i], groups[j], equal_var=(levene_p > 0.05))
            pairwise[f"{models[i]} vs {models[j]}"] = {
                "t_stat": round(t_stat, 4),
                "p_value": round(p_val, 4),
                "significant": p_val < 0.05,
            }

    summary = df.groupby("model")["clip_score"].agg(["mean", "std", "min", "max"]).round(4)

    return {
        "anova_f_stat": round(f_stat, 4),
        "anova_p_value": round(anova_p, 4),
        "levene_stat": round(levene_stat, 4),
        "levene_p_value": round(levene_p, 4),
        "homogeneous_variance": levene_p > 0.05,
        "pairwise_t_tests": pairwise,
        "summary_stats": summary.to_dict(),
    }


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_prompts = [
        "A robot walking through a futuristic city at sunset",
        "Ocean waves crashing on a rocky shore",
        "A cat playing with a ball of yarn",
    ]

    print("Running comparative study across ModelScope, Text2Video-Zero, MotionConsistency...")
    results_df = run_comparison(test_prompts)
    results_df.to_csv("clip_scores_by_model.csv", index=False)

    stats_summary = statistical_comparison(results_df)
    print("\n=== ANOVA ===")
    print(f"  F-statistic: {stats_summary['anova_f_stat']} | p-value: {stats_summary['anova_p_value']}")
    print("\n=== Levene's Test (Homogeneity of Variance) ===")
    print(f"  Statistic: {stats_summary['levene_stat']} | p-value: {stats_summary['levene_p_value']}")
    print(f"  Homogeneous variance: {stats_summary['homogeneous_variance']}")
    print("\n=== Pairwise t-tests ===")
    for comparison, result in stats_summary["pairwise_t_tests"].items():
        sig = "significant" if result["significant"] else "not significant"
        print(f"  {comparison}: t={result['t_stat']}, p={result['p_value']} ({sig})")
