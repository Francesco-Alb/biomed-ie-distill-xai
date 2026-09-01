from typing import Any
import numpy as np
import pandas as pd
from scipy.special import softmax


def analyze_prediction_errors(
    dataset: Any,
    prediction_output: Any,
    id_column: str = "pair_id",
    text_column: str = "masked_text",
    error_only: bool = False,
) -> pd.DataFrame:
    """
    Create an observation-level error-analysis DataFrame from Trainer.predict().

    For binary classification, adds TP/FP/TN/FN categories.
    """

    temp_df = dataset.to_pandas().copy()

    # -------------------------------------------- Extract logits and labels
    logits = prediction_output.predictions
    labels = prediction_output.label_ids

    # -------------------------------------------- Calculate softmax probabilities and predicted classes
    probs = softmax(logits, axis=-1)
    preds = np.argmax(probs, axis=-1)

    # -------------------------------------------- Create full error_df
    df_errors = pd.DataFrame({
        "index": temp_df[id_column],
        "text": temp_df[text_column],
        "true_label": labels,
        "pred_label": preds,
        "confidence": [
            p[pred] for p, pred in zip(probs, preds)
        ], # Prob of the chosen class
    })

    # -------------------------------------------- Add binary error categories
    if len(np.unique(labels)) == 2:
        df_errors["category"] = np.select(
            [
                (labels == 1) & (preds == 1),
                (labels == 0) & (preds == 1),
                (labels == 0) & (preds == 0),
                (labels == 1) & (preds == 0),
            ],
            ["TP", "FP", "TN", "FN"],
            # NOTE: A default value is required by numpy.select() to align dtypes 
            # with the string choice list (preventing a casting TypeError), even though 
            # all binary classification outcomes are fully covered by the 4 conditions above.
            default="Other",
        )

    # -------------------------------------------- Return only wrong rows error_df
    if error_only:
        df_errors = df_errors.query("true_label != pred_label")

    return df_errors