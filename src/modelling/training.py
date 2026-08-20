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
from peft import get_peft_model
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


class WeightedLossTrainer(Trainer):
    """
    Custom Trainer that applies pre-computed class weights.
    """
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None
        self._loss_fct = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Lazy initialization of loss function on the correct device
        if self._loss_fct is None:
            if self.class_weights is not None:
                weights = self.class_weights.to(logits.device)
                self._loss_fct = nn.CrossEntropyLoss(weight=weights)
            else:
                self._loss_fct = nn.CrossEntropyLoss()
        
        loss = self._loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
    

def _visualize_smoothing_effect(weights: np.ndarray, smoothed_weights: np.ndarray):
    """
    Visualizes the effect of weight smoothing using log1p transformation.
    Used in: define_weight_strategy() when visualize=True.
    
    Args:
        weights (np.ndarray): Raw class weights.
        smoothed_weights (np.ndarray): Smoothed class weights after transformation.
    """
    
    # Weights check (before vs after smoothing)
    print(f"Unsmoothed class weights range: {weights.min():.2f} → {weights.max():.2f}")
    print(f"Smoothed class weights range: {smoothed_weights.min():.2f} → {smoothed_weights.max():.2f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(sorted(weights), label='Original')
    plt.plot(sorted(smoothed_weights), label='Smoothed')
    plt.legend()
    plt.title("Class Weight Smoothing Effect")
    plt.xlabel("Class Index (sorted)")
    plt.ylabel("Weight")
    plt.show()
    
def define_weight_strategy(
    labels: list, 
    threshold: float = 10.0,
    visualize: bool = False,
    power: float = 0.5  # Square Root Smoothing (best for Transformers)
):

    """
    Analyzes class distribution and recommends a weighting strategy.
    
    Args
    ----------
        labels (list): Training targets/labels.
        threshold (float): Imbalance ratio cutoff (default is 10:1).
        visualize (bool): Whether to visualize the smoothing effect (default is False).
        power (float): Square Root Smoothing (best for Transformers)


    Returns
    -------
        str: Weight strategy recommendation ('smoothed', or 'none').
    """
    classes, counts = np.unique(labels, return_counts=True)
    max_count = np.max(counts)
    min_count = np.min(counts)
    imbalance_ratio = max_count / min_count
    
    print("--- IMBALANCE ANALYSIS ---")
    print(f"Majority class count: {max_count}")
    print(f"Minority class count: {min_count}")
    print(f"Imbalance ratio:      {imbalance_ratio:.2f}:1\n")
    
    print("--- RECOMMENDATION ---")
    if imbalance_ratio >= threshold:
        # Compute smooth weights via power scaling (sqrt)
        weights = (max_count / counts) ** power
        # Normalize so mean weight == 1.0
        smoothed_weights = weights / np.mean(weights)
        
        print("Use SMOOTHED (Square-Root) weights.")
        print(f"Reason: Imbalance exceeds {threshold}:1.")
        print(f"Weight vector: {np.round(smoothed_weights, 3)}")
        if visualize:
            _visualize_smoothing_effect(weights, smoothed_weights)
        return "smoothed", smoothed_weights
    else:
        print("Use NO weights (Uniform).")
        print(f"Reason: Imbalance is mild ({imbalance_ratio:.2f}:1 < {threshold}:1).")
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

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()