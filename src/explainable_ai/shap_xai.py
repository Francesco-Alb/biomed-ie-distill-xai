import re
import numpy as np
import torch
import transformers
import shap
from pathlib import Path


MAX_SAFE_SAMPLES = 10


def build_explainer(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizerBase,
    device: str | torch.device,
    padding: bool | str = True,
    max_length: int = 500,
    truncation: bool | str = True,
) -> shap.Explainer:
    """
    Build a SHAP explainer for the given model and tokenizer.

    Args:
        model: The trained model to explain.
        tokenizer: The tokenizer used for the model.
        device: The device to run the model on (e.g., 'cuda' or 'cpu').
        padding: The padding strategy to use.
        max_length: The maximum length of the input sequences.
        truncation: Whether to truncate sequences longer than max_length.

    Returns:
        explainer: A SHAP explainer object.
    """ 

    model.to(device)
    model.eval()

    def f(texts: list[str] | np.ndarray) -> np.ndarray:
        inputs = tokenizer(
            list(texts),
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        return outputs.logits.detach().cpu().numpy()
    
    return shap.Explainer(f, tokenizer)


def save_shap_text_plot(
    shap_values: shap.Explanation,
    filename: str | Path = Path("shap_text_plot.html"),
    sample_index: int | str | None = 0,
    class_id: int | None = None
) -> None:
    """
    Save a SHAP text plot to an HTML file.

    Args:
        shap_values: The SHAP Explanation object.
        filename: Path to save the HTML output.
        sample_index: Index of the sample to plot (e.g. 0, 1, -1 for last).
                    * Set to 'all' or None to plot all samples in shap_values. Default is 0.
                    * MAX_SAFE_SAMPLES is set to 10 to avoid generating large files by accident.
        class_id: The class ID to isolate. If None, plots all classes for the sample(s).

    Examples:
        Plot one sample for all classes:
            save_shap_text_plot(
                shap_values,
                f"evals/images/xai/one_sample_all_classes_{model_type}.html",
                sample_index=3,
            )

        Plot all samples for all classes (up to MAX_SAFE_SAMPLES):
            save_shap_text_plot(
                shap_values,
                "evals/images/xai/all_samples_all_classes.html",
                sample_index="all",
            )

        Plot one sample for a single class:
            save_shap_text_plot(
                shap_values,
                "evals/images/xai/one_sample_one_class.html",
                sample_index=5,
                class_id=5,
            )
    """

    filename = Path(filename)

    # Select samples
    is_all_samples = (
        sample_index is None
        or (isinstance(sample_index, str) and sample_index.lower() == "all")
    )

    if is_all_samples:
        num_samples = len(shap_values)

        if num_samples > MAX_SAFE_SAMPLES:
            raise ValueError(
                f"Cannot plot all {num_samples} samples at once "
                f"(limit is {MAX_SAFE_SAMPLES}). "
                "Specify a sample_index to avoid generating excessively large HTML files."
            )
        
        explanation = shap_values

    else:
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise TypeError(
                "`sample_index` must be an integer, 'all', or None."
            )
        
        explanation = shap_values[int(sample_index)]

    # Select class, if requested
    if class_id is not None:
        explanation = explanation[..., class_id]

    # Generate HTML
    raw_html = shap.plots.text(explanation, display=False)

    # Clean NumPy scalar representations if required by the SHAP output
    cleaned_html = re.sub(
        r"np\.float\d+\(([^)]+)\)", 
        r"\1", 
        raw_html
    )

    # Explicitly bundle the required SHAP JavaScript headers
    final_html = (
        f"<!DOCTYPE html>"
        f"<html>"
        f"<head>{shap.getjs()}</head>"
        f"<body>{cleaned_html}</body>"
        f"</html>"
    )

    # Create output directory and save
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(final_html, encoding="utf-8")