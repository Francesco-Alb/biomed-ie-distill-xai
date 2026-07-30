import evaluate
import numpy as np

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