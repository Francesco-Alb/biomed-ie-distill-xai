import os
import gc
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from peft import get_peft_model, PeftModel
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


class WeightedLossTrainer(Trainer):
    """
    Custom Trainer that safely applies class weights to the loss function 
    across any single or multi-GPU environment.
    """
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep weights as a basic tensor in memory during initialization
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        
        # Pass inputs to the model to get the forward pass logits
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Dynamically map the class weights to whichever device the active batch is on
        device = labels.device
        weights = self.class_weights.to(device)
        
        # Calculate cross-entropy with the weights mapping
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
    

def _visualize_smoothing_effect(balanced_weights: np.ndarray, smoothed_weights: np.ndarray):
    """
    Visualizes the effect of weight smoothing using log1p transformation.
    Used in: check_weight_strategy() when visualize=True.
    
    Args:
        balanced_weights (np.ndarray): Balanced class weights from sklearn.
        smoothed_weights (np.ndarray): Smoothed class weights after transformation.
    """
    
    # Weights check (before vs after smoothing)
    print(f"Unsmoothed class weights range: {balanced_weights.min():.2f} → {balanced_weights.max():.2f}")
    print(f"Smoothed class weights range: {smoothed_weights.min():.2f} → {smoothed_weights.max():.2f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(sorted(balanced_weights), label='Original')
    plt.plot(sorted(smoothed_weights), label='Smoothed')
    plt.legend()
    plt.title("Class Weight Smoothing Effect")
    plt.xlabel("Class Index (sorted)")
    plt.ylabel("Weight")
    plt.show()
    
def check_weight_strategy(
    labels: list = [], 
    smoothing_factor: float = 1.0,
    threshold: float = 10.0,
    visualize: bool = False
):
    """
    Analyzes class distribution and recommends a weighting strategy.
    
    Args
    ----------
        labels (list): Training targets/labels.
        threshold (float): Imbalance ratio cutoff (default is 10:1).
        visualize (bool): Whether to visualize the smoothing effect (default is False).
        smoothing_factor (float): Controls the degree of smoothing (default is 1.0).

    Returns
    -------
        str: Weight strategy recommendation ('balanced', 'smoothed', or 'none').

    Examples:
    ----------
        labels = hf_dataset['train']['label']
        check_weight_strategy(labels, visualize=True)

        # Example 1: Severe imbalance
        print("Test Case A:")
        check_weight_strategy([0]*1000 + [1]*5, visualize=True) 

        # Example 2: Mild imbalance
        print("Test Case B:")
        check_weight_strategy([0]*100 + [1]*30)
    """
    classes, counts = np.unique(labels, return_counts=True)
    
    max_count = np.max(counts)
    min_count = np.min(counts)
    
    # Calculate how many times more frequent the majority class is
    imbalance_ratio = max_count / min_count
    
    print("--- IMBALANCE ANALYSIS ---")
    print(f"Majority class count: {max_count}")
    print(f"Minority class count: {min_count}")
    print(f"Imbalance ratio:      {imbalance_ratio:.2f}:1\n")
    
    # Compute balanced weights for visualization/reference
    balanced_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    # Normalize weights to have mean=1 before smoothing
    normalized_weights = balanced_weights / balanced_weights.mean()
    
    # Apply log1p transformation to compress extreme values
    smoothed_weights = np.log1p(normalized_weights * smoothing_factor) + 1.0
    
    print("--- RECOMMENDATION ---")
    if imbalance_ratio >= threshold:
        print("Use SMOOTHED weights.")
        print(f"Reason: Imbalance exceeds {threshold}:1.")
        print("Raw weights will destabilize BERT gradients.")
        if visualize:
            _visualize_smoothing_effect(balanced_weights, smoothed_weights)
        return "smoothed", smoothed_weights
    elif imbalance_ratio > 1.05:  # >5% variance
        print("Use NORMAL (balanced) weights.")
        print(f"Reason: Imbalance is mild (under {threshold}:1).")
        print("Model will safely handle raw scaling.")
        if visualize:
            balanced_weights = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=labels
            )
            _visualize_smoothing_effect(balanced_weights, smoothed_weights)
        return "balanced", balanced_weights
    else:
        print("Use NO weights (Uniform).")
        print("Reason: Dataset is perfectly balanced.")
        return "none", None


def make_training_args(base_args: TrainingArguments, seed: int) -> TrainingArguments:
    """Return a new TrainingArguments object with seed set and run_name updated.
    
    Args:
        base_args: The original TrainingArguments object to copy.
        seed: The random seed value to apply to the new arguments.
        
    Returns:
        A deep-copied TrainingArguments instance with the updated `seed`
        and modified keys [`run_name`, `output_dir`] appended with the seed suffix.
    """
    from copy import deepcopy
    
    args = deepcopy(base_args)
    args.seed = seed
    suffix = f"-seed{seed}"
    
    if suffix not in args.run_name:
        args.run_name += suffix

    if suffix not in str(args.output_dir):
        args.output_dir = f"{args.output_dir}{suffix}"

    return args


def _get_trainer_state_file(output_dir: Path) -> Optional[Path]:
    """Find trainer_state.json either directly in output_dir or inside a checkpoint directory."""
    root_state = output_dir / "trainer_state.json"
    if root_state.exists():
        return root_state

    ckpt_states = sorted(output_dir.glob("checkpoint-*/trainer_state.json"))
    if ckpt_states:
        return ckpt_states[-1]

    return None


def _extract_metric_and_checkpoint(
    state_data: dict[str, Any], eval_metric: str, default_dir: Path
) -> tuple[Optional[float], str]:
    """Extract best metric and checkpoint path directly from trainer_state JSON data."""
    score = state_data.get("best_metric")

    # Fallback to log_history if best_metric was not set
    if score is None and "log_history" in state_data:
        keys_to_check = {
            eval_metric,
            f"eval_{eval_metric}" if not eval_metric.startswith("eval_") else eval_metric.replace("eval_", ""),
        }
        scores = [
            entry[k]
            for entry in state_data["log_history"]
            for k in keys_to_check
            if k in entry
        ]
        if scores:
            score = max(scores)

    best_ckpt = state_data.get("best_model_checkpoint") or str(default_dir)
    return (float(score) if score is not None else None), str(best_ckpt)


def train_single_seed(
    seed: int,
    baseline_training_args: TrainingArguments,
    model_checkpoint: str,
    model_config: Any,
    peft_config: Any,
    train_dataset: Any,
    eval_dataset: Any,
    class_weights: Sequence[float],
    compute_metrics_fn: Callable[..., Any],
    data_collator: Any,
    early_stopping: Optional[Any] = None,
    eval_metric: str = "eval_f1",
    *,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    return_trainer: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train one seed, save state to disk, and return the run summary."""
    seed_args = make_training_args(baseline_training_args, seed)
    output_dir = Path(seed_args.output_dir)
    state_file = _get_trainer_state_file(output_dir)

    if state_file is not None:
        with open(state_file, "r") as f:
            state_data = json.load(f)
        metric_score, best_ckpt = _extract_metric_and_checkpoint(state_data, eval_metric, output_dir)
        if verbose:
            print(f"\n=== Skipping seed {seed}: Already completed (trainer_state found on disk) ===")
        return {
            "seed": seed,
            f"{eval_metric}": metric_score,
            "best_checkpoint": best_ckpt,
        }

    if verbose:
        print(f"\n\n=== Initializing training with seed: {seed} ===")

    seed_model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        config=model_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    seed_model = get_peft_model(seed_model, peft_config)

    trainer = WeightedLossTrainer(
        model=seed_model,
        args=seed_args,
        class_weights=class_weights,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fn,
        data_collator=data_collator,
        callbacks=[early_stopping] if early_stopping is not None else None,
    )

    start = time.time()
    trainer.train(resume_from_checkpoint=False)
    trainer.save_state()  # Saves trainer_state.json directly to output_dir
    end = time.time()

    if verbose:
        print(
            f"Seed {seed} finished in {(end - start) / 60:.2f} min"
        )

    if not return_trainer:
        del trainer
        del seed_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def aggregate_seed_results(
    seed_list: Sequence[int],
    baseline_training_args: Any,
    eval_metric: str = "eval_f1",
    plot: bool = False,
    save_plot_path: Path | str | None = None,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Collect metrics from trainer_state files of completed seeds.
    
    Args:
        seed_list: List of random seeds evaluated.
        baseline_training_args: Base TrainingArguments used for the experiment.
        eval_metric: Metric key to aggregate (default: 'eval_f1').
        plot: If True, renders a boxplot with individual seed data points.
        verbose: If True, prints formatted summary.
        
    Returns:
        dict containing 'completed_runs', 'mean_score', 'std_score', and 'best_run'.
    """
    completed_runs = []

    for seed in seed_list:
        seed_args = make_training_args(baseline_training_args, seed)
        output_dir = Path(seed_args.output_dir)
        state_file = _get_trainer_state_file(output_dir)

        if state_file is not None:
            with open(state_file, "r") as f:
                state_data = json.load(f)
            metric_score, best_ckpt = _extract_metric_and_checkpoint(state_data, eval_metric, output_dir)
            if metric_score is not None:
                completed_runs.append({
                    "seed": seed,
                    eval_metric: metric_score,
                    "best_checkpoint": best_ckpt,
                })

    if not completed_runs:
        if verbose:
            print("⚠️ No completed runs found.")
        return {
            "completed_runs": [],
            "mean_score": None,
            "std_score": None,
            "best_run": None,
        }

    scores = [run[eval_metric] for run in completed_runs]
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    best_run = max(completed_runs, key=lambda x: x[eval_metric])

    if verbose:
        print(f"\n📊 Results across {len(completed_runs)}/{len(seed_list)} completed seeds:")
        print(f"   {eval_metric}: {mean_score:.4f} ± {std_score:.4f}")
        print(
            f"   Best Seed: {best_run['seed']} ({eval_metric}: {best_run[eval_metric]:.4f} @ {best_run['best_checkpoint']})"
        )

    if plot and completed_runs:
        _plot_seed_distribution(scores, completed_runs, eval_metric, mean_score, std_score, save_plot_path)

    return {
        "completed_runs": completed_runs,
        "mean_score": mean_score,
        "std_score": std_score,
        "best_run": best_run,
    }


def _plot_seed_distribution(
    scores: list[float],
    completed_runs: list[dict[str, Any]],
    eval_metric: str,
    mean_score: float,
    std_score: float,
    save_plot_path: Path | str | None = None,
):
    """Helper function to plot score distribution across seeds."""
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Horizontal boxplot
    sns.boxplot(x=scores, ax=ax, color="lightblue", width=0.3, boxprops=dict(alpha=0.7))
    
    # Overlaid individual seed points (strip plot)
    seeds = [str(run["seed"]) for run in completed_runs]
    sns.stripplot(x=scores, ax=ax, color="darkblue", size=8, jitter=0.05)
    
    # Annotate seed numbers next to points
    for run in completed_runs:
        ax.annotate(
            f" seed {run['seed']}", 
            (run[eval_metric], 0.05), 
            fontsize=9, 
            va="center"
        )

    ax.axvline(mean_score, color="red", linestyle="--", label=f"Mean: {mean_score:.4f} (±{std_score:.4f})")
    
    ax.set_title(f"Multi-Seed Performance Distribution ({eval_metric})")
    ax.set_xlabel(eval_metric)
    ax.legend(loc="best")
    plt.tight_layout()
    
    if save_plot_path:
        save_plot_path = Path(save_plot_path)
        save_plot_path = save_plot_path.with_suffix(".png")
        save_plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
    
    plt.show()