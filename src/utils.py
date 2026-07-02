import sys
import importlib
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset

def dynamic_module_reloader(
        modules: str|list[str] = "",
        verbose: bool = False
        ) -> None:
    """
    Reload all relevant modules dynamically.
    
    Args:
        modules: The modules to reload. Can be a single module or a list of modules.
        verbose: Whether to print the names of the reloaded modules.
    """
    if isinstance(modules, str):
        modules = [modules]
    for name in modules:
        if name in sys.modules:
            try:
                importlib.reload(sys.modules[name])
                if verbose:
                    print(f"Reloaded: {name}")
            except Exception as e:
                print(f"Warning: Could not reload {name}: {e}")

def verbose_print(verbose: bool, *messages, sep: str = "\n"):
    if verbose:
        print(*messages, sep=sep)


def seed_everything(seed: int) -> None:
    """
    Seeds the random number generators.

    Args:
        seed: The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✅ Seed set to {seed}")


def plot_length_distribution_with_percentiles(
        lengths: list[int], 
        model_max_length: int|None=None, 
        bins: int=60): 
    """
    Plot a histogram of token lengths with annotated percentile lines and an optional
    model context window marker. This visualization clarifies the distribution of 
    sequence lengths and how they relate to contextual capacity.

    Args:
        lengths : list[int]
            A list of token lengths for all dataset examples.
        model_max_length : int, optional
            The maximum number of tokens the model can process.
            If included, draws a vertical line marking the model's context window. Defaults to None.
        bins : int, optional
            Number of histogram bins. Defaults to 60.

    Returns:
        None: Displays the histogram and prints how many examples exceed the model's context length.
    """
    
    p25, p50, p75, p90, p99, max_val = np.percentile(lengths, [25, 50, 75, 90, 99, 100])

    plt.figure(figsize=(12, 6))
    sns.histplot(lengths, bins=bins, color="skyblue", edgecolor="white", element="bars", alpha=0.5)

    # Percentile lines
    for val, label in [(p25, "25th"), (p50, "50th"), (p75, "75th"), (p90, "90th"), (p99, "99th"), (max_val, "max_val")]:
        plt.axvline(val, linestyle="--", linewidth=1.5, color="grey")
        plt.text(val, plt.ylim()[1]*0.9, label, rotation=90, va="top")

    # Context window 
    if model_max_length is not None:
        plt.axvline(model_max_length, color="red", linewidth=2, linestyle="--", label="Context window")

    plt.title("Token Length Distribution with Percentiles", fontsize=16)
    plt.xlabel("Token count", fontsize=13)
    plt.ylabel("Frequency", fontsize=13)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Check how many samples exceed the model's model_max_length
    if model_max_length is not None:
        exceeding = sum([l > model_max_length for l in lengths])
        print("_"*100)
        print(f"There's {exceeding} sample(s) exceeding model_max_length")


def estimate_optimization_steps(
        dataset: Dataset,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 1
        ) -> None:
    """
    Estimates the number of optimization steps per epoch based on dataset size,
    per-step batch size, and optional gradient accumulation.

    Args:
        dataset: A Hugging Face Dataset containing the training data.
        batch_size: The per-step batch size.
        gradient_accumulation_steps: Number of steps to accumulate gradients before
            updating model weights. Defaults to 1.
    
    Returns:
        None: Prints a formatted string with the estimated number of optimization
        steps per epoch.
    """
    dataset_size = dataset.shape[0]
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = dataset_size // effective_batch_size

    text = (
        f"Per-step batch size is: {batch_size}\n"
        f"Gradient accumulation steps: {gradient_accumulation_steps}\n"
        f"Effective batch size is: {effective_batch_size}\n"
        f"The dataset has {dataset_size} examples\n"
        f"One full pass (epoch) through the dataset corresponds to roughly\n"
        f"{dataset_size}/{effective_batch_size} ≈ {steps_per_epoch} optimization steps."
    )

    print(text)