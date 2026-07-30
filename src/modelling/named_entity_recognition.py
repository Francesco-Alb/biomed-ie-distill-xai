from transformers import AutoTokenizer

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