import pandas as pd
from datasets import DatasetDict, Dataset

def ensure_validation_split(
    dataset: DatasetDict,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetDict:
    """Checks if a Hugging Face DatasetDict has a validation set. 
    
    If not, splits the train set to create one.
    """
    if "validation" in dataset.keys():
        print("Dataset has a validation split already: no action performed.")
        return dataset
        
    # Assuming it has a "train" key to split from
    train_test = dataset["train"].train_test_split(test_size=test_size, seed=random_state)
    
    dataset["train"] = train_test["train"]
    dataset["validation"] = train_test["test"]
    
    print("Validation split created.")
    return dataset


def flatten_bc5cdr(dataset_split: Dataset) -> Dataset:
    """
    Flattens the BC5CDR dataset split (train/validation/test) into a single DataFrame with one row per document.

    The resulting DataFrame has the following columns:
    - document_id: The unique identifier for the document (from the first passage).
    - text: The combined text of all passages in the document.
    - entities: A list of all entities across passages, where each entity is a dictionary containing:
        - offsets: The character offsets of the entity in the combined text.
        - text: The text of the entity.
        - type: The type of the entity (e.g., "Chemical", "Disease").
        - mesh_id: The MeSH ID of the entity if available, otherwise None.
    - relations: A list of unique relations across passages, where each relation is a dictionary containing:
        - chem_mesh: The MeSH ID of the chemical entity in the relation.
        - dis_mesh: The MeSH ID of the disease entity in the relation.

    Args:
        dataset_split (Dataset): A Hugging Face Dataset split (e.g., train, validation, or test) from the BC5CDR dataset.

    Returns:
        Dataset: A Hugging Face Dataset containing the flattened data.
    """
    flat_data = []
    
    for row in dataset_split:
        passages = row['passages']
        
        # 1. Combine title and abstract text
        # (This single space ensures offsets like [93, 105] match perfectly)
        combined_text = " ".join([p['text'] for p in passages])
        doc_id = passages[0]['document_id']
        
        # 2. Extract and flatten all entities across passages
        all_entities = []
        for p in passages:
            for ent in p['entities']:
                # Extract MeSH ID safely if it exists
                mesh_id = ent['normalized'][0]['db_id'] if ent['normalized'] else None
                
                all_entities.append({
                    "offsets": ent['offsets'][0], # Takes the [start, end] list
                    "text": ent['text'][0],
                    "type": ent['type'],
                    "mesh_id": mesh_id
                })
        
        # 3. Extract and flatten all unique relations across passages
        all_relations = []
        for p in passages:
            if 'relations' in p and p['relations']:
                for rel in p['relations']:
                    all_relations.append({
                        "chem_mesh": rel['arg1_id'],
                        "dis_mesh": rel['arg2_id']
                    })
        
        # Remove any duplicate relation definitions if they appear in multiple passages
        unique_relations = [dict(t) for t in {tuple(d.items()) for d in all_relations}]
        
        flat_data.append({
            "document_id": doc_id,
            "text": combined_text,
            "entities": all_entities,
            "relations": unique_relations
        })
        
    # return pd.DataFrame(flat_data)
    return Dataset.from_pandas(pd.DataFrame(flat_data))