from transformers import AutoTokenizer
import evaluate
import numpy as np

seqeval = evaluate.load("seqeval")

def tokenize_and_align_labels(
        examples: dict, 
        tokenizer: AutoTokenizer,
        padding: bool = True,
        truncate: bool = True
        ) -> dict:
    """
    Tokenizes the input text and aligns the entity labels with the tokenized output.
    
    Args:
        examples (dict): A dictionary containing the input text and entity annotations. 
                         Expected keys are "text" (the document text) and "entities" (a list of entity annotations).
        tokenizer (AutoTokenizer): A Hugging Face tokenizer instance to tokenize the input text.
        padding (bool): Whether to pad the tokenized sequences to a maximum length. Default is True.
        truncate (bool): Whether to truncate the tokenized sequences to a maximum length. Default is True.

    Returns:
        dict: A dictionary containing the tokenized inputs and aligned labels.
    """
    tokenized_inputs = tokenizer(
        examples["text"], 
        truncation=True,
        padding=padding,
        truncate=truncate,
        max_length=512, 
        return_offsets_mapping=True
    )
    
    labels = []
    # Loop through each document in the batch
    for i, labels_list in enumerate(examples["entities"]):
        doc_labels = [0] * len(tokenized_inputs["input_ids"][i])
        offset_mapping = tokenized_inputs["offset_mapping"][i]
        
        # Look at every true entity found in the data prep step
        for ent in labels_list:
            start_char, end_char = ent["offsets"]
            ent_type = ent["type"] # 'Chemical' or 'Disease'
            
            b_tag = 1 if ent_type == "Chemical" else 3
            i_tag = 2 if ent_type == "Chemical" else 4
            
            first_match = True
            for idx, (start_tok, end_tok) in enumerate(offset_mapping):
                # Ignore special tokens like [CLS] or [SEP]
                if start_tok == 0 and end_tok == 0:
                    doc_labels[idx] = -100 # PyTorch default to ignore crs-entropy ls
                    continue
                    
                # Check if the token falls within the character bounds of the entity
                if start_tok >= start_char and end_tok <= end_char and start_tok < end_tok:
                    if first_match:
                        doc_labels[idx] = b_tag
                        first_match = False
                    else:
                        doc_labels[idx] = i_tag
                        
        labels.append(doc_labels)
        
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

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