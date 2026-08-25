import json
from logging import config
import re
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.special import softmax
import torch
from peft import PeftModel
import evaluate
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.modelling.training import (
    _get_trainer_state_file,
    _extract_metric_and_checkpoint,
)

seqeval = evaluate.load("seqeval")

def compute_metrics_ner(
        p: tuple[np.ndarray, np.ndarray], 
        label_list: list[str]
        ) -> dict[str, float]:
    """
    Computes evaluation metrics for NER predictions using the seqeval library.

    Args:
        p (tuple): A tuple containing the model's predictions and the true labels. 
                   - predictions: A numpy array of shape (batch_size, sequence_length, num_labels) containing the predicted probabilities for each label.
                   - labels: A numpy array of shape (batch_size, sequence_length) containing the true label indices.

        label_list (list[str]): A list of label names corresponding to the label indices.
    """
    
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def compute_metrics_re(
        eval_pred: tuple[np.ndarray, np.ndarray], 
        average: str = "binary"
        ) -> dict[str, float]:
    """
    Computes evaluation metrics for relation extraction predictions using the evaluate library.

    Args:
        eval_pred (tuple): A tuple containing the model's prediction logits and the true labels.
                           - logits: A numpy array containing the model's predicted relation logits.
                           - labels: A numpy array containing the true relation label indices.

        average (str): The type of averaging performed on precision, recall, and F1.
                       Options include 'binary', 'micro', 'macro', 'weighted', etc. Default is 'binary'.

    Returns:
        dict[str, float]: A dictionary containing 'accuracy', 'recall', 'precision', and 'f1' scores.
    """
    accuracy_metric = evaluate.load("accuracy")
    recall_metric = evaluate.load("recall")
    precision_metric = evaluate.load("precision")
    f1_metric = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "recall": recall_metric.compute(predictions=predictions, references=labels, average=average)["recall"],
        "precision": precision_metric.compute(predictions=predictions, references=labels, average=average)["precision"],
        "f1": f1_metric.compute(predictions=predictions, references=labels, average=average)["f1"]
    }


def aggregate_seed_results(
    output_dir: Path | str,
    eval_metric: str = "eval_f1",
    plot: bool = False,
    save_plot_path: Path | str | None = None,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Collect metrics from completed relation extraction runs inside `output_dir`.

    Args:
        output_dir: Directory containing seed-specific model output folders.
        eval_metric: Metric key to aggregate (default: 'eval_f1').
        plot: If True, renders a boxplot with individual seed data points.
        save_plot_path: Optional path to save the generated plot.
        verbose: If True, prints formatted summary.

    Returns:
        dict containing 'completed_runs', 'mean_score', 'std_score', and 'best_run'.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists() or not output_dir.is_dir():
        raise ValueError(f"output_dir does not exist or is not a directory: {output_dir}")

    completed_runs = []
    run_dirs = [d for d in output_dir.iterdir() if d.is_dir()]

    for run_dir in run_dirs:
        seed = _parse_seed_from_dir_name(run_dir.name)
        state_file = _get_trainer_state_file(run_dir)

        if state_file is None:
            if verbose:
                print(f"⚠️ Skipping {run_dir.name}: no trainer_state.json found.")
            continue

        with open(state_file, "r") as f:
            state_data = json.load(f)

        metric_score, best_ckpt = _extract_metric_and_checkpoint(state_data, eval_metric, run_dir)
        if metric_score is None:
            if verbose:
                print(f"⚠️ Skipping {run_dir.name}: metric '{eval_metric}' not found.")
            continue

        completed_runs.append({
            "seed": seed,
            "run_dir": str(run_dir),
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
        print(f"\n📊 Results across {len(completed_runs)} completed runs:")
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


def _parse_seed_from_dir_name(dir_name: str) -> Optional[int]:
    """Extract seed integer from a directory name like '...-seed1' or '...-seed42'."""
    match = re.search(r"(?:^|[-_])seed(\d+)(?:$|[-_])", dir_name, re.IGNORECASE)
    return int(match.group(1)) if match else None


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


def load_best_trainer(
    best_run: dict,
    model_configs: Any,
    compute_metrics_fn: Callable[..., Any],
    args: Optional[TrainingArguments] = None,
) -> Trainer:
    """
    Reconstructs a Trainer from a best_run dict to perform post-training tasks.
    """

    auto_config = AutoConfig.from_pretrained(
        model_configs.re_model_checkpoint,
        num_labels=len(model_configs.re_label_names_binary),
        id2label={i: label for i, label in enumerate(model_configs.re_label_names_binary)},
        label2id={label: i for i, label in enumerate(model_configs.re_label_names_binary)}
    )
    
    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_configs.re_model_checkpoint, config=auto_config
    )

    #  Load tokenizer and datacollator
    tokenizer = AutoTokenizer.from_pretrained(model_configs.re_model_checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer)
    
    # Load best checkpoint as PEFT model
    checkpoint_path = best_run["best_checkpoint"]
    model = PeftModel.from_pretrained(model, checkpoint_path)
    
    # Re-initialize trainer
    best_trainer = Trainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )
    
    return best_trainer


def get_validation_probabilities(trainer: Trainer, eval_dataset: Any) -> np.ndarray:
    """
    Extracts positive-class probabilities from a Trainer prediction output.
    """
    predictions = trainer.predict(eval_dataset)
    probs = softmax(predictions.predictions, axis=-1)
    return probs[:, 1]


def find_optimal_threshold(val_probs: np.ndarray, val_labels: np.ndarray) -> float:
    """
    Finds the probability threshold that maximizes F1-score on validation data.
    """
    from sklearn.metrics import f1_score, precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(val_labels, val_probs)
    
    # Calculate F1 scores across all candidate thresholds
    # Avoid division by zero
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    
    # Thresholds array is length N-1, precisions/recalls are length N
    # The threshold at best_idx corresponds to the transition between best_idx and best_idx+1
    best_threshold = thresholds[best_idx]
    
    print(f"Default (0.50) Threshold F1: {f1_score(val_labels, (val_probs > 0.5).astype(int)):.4f}")
    print(f"Optimal ({best_threshold:.4f}) Threshold F1: {f1_scores[best_idx]:.4f}")
    
    return float(best_threshold)


def compute_metrics_at_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    average: str = "binary"
) -> dict[str, float]:
    """
    Computes evaluation metrics at a specific classification threshold.
    
    Args:
        probabilities: Array of positive-class probabilities (shape: (n_samples,)).
        labels: Array of true binary labels (shape: (n_samples,)).
        threshold: Classification threshold to apply.
        average: Averaging method for metrics ('binary', 'micro', 'macro', 'weighted').
    
    Returns:
        dict containing 'accuracy', 'precision', 'recall', and 'f1' scores.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    predictions = (probabilities > threshold).astype(int)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average=average, zero_division=0),
        "recall": recall_score(labels, predictions, average=average, zero_division=0),
        "f1": f1_score(labels, predictions, average=average, zero_division=0)
    }