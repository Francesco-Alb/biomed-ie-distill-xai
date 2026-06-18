import pandas as pd
import numpy as np
import itertools
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


def create_labeled_candidate_pairs(row: pd.Series) -> list[dict]:
    """
    Create labeled candidate pairs from a given row.

    Args:
    - row (pd.Series): A pandas Series containing the data for a single row.

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the labeled candidate pairs.

    Notes:
    - This function assumes that the input row contains the necessary information to create labeled candidate pairs.
    """
    text = row["text"]
    entities = row["entities"]
    
    # Create a quick lookup set of true ground-truth relations for this document
    true_relations = set()
    if "relations" in row and isinstance(row["relations"], (list, np.ndarray)):
        for rel in row["relations"]:
            true_relations.add((rel["chem_mesh"], rel["dis_mesh"]))

    # 2. Group entity mentions by their UNIQUE Concept IDs
    chem_groups = {}  # Format: { 'D003000': [mention1, mention2] }
    dis_groups = {}   # Format: { 'D006973': [mention1, mention2] }
    
    for e in entities:
        
        if e["type"] == "Chemical":
            chem_groups.setdefault(e["mesh_id"], []).append(e)
        elif e["type"] == "Disease":
            dis_groups.setdefault(e["mesh_id"], []).append(e)

    if not chem_groups or not dis_groups:
        return []
        
    candidate_rows = []
    
    # 3. Pair UNIQUE IDs instead of individual mentions
    for chem_id, dis_id in itertools.product(chem_groups.keys(), dis_groups.keys()):
        
        label = 1 if (chem_id, dis_id) in true_relations else 0
        
        # 4. Gather ALL tag insertions for ALL mentions of this specific pair
        insertions = []
        
        # Collect all chemical mention boundaries for this ID
        for chem_ent in chem_groups[chem_id]:
            c_start, c_end = chem_ent["offsets"]
            insertions.append((c_start, "<chemical> "))
            insertions.append((c_end, " </chemical>"))
            
        # Collect all disease mention boundaries for this ID
        for dis_ent in dis_groups[dis_id]:
            d_start, d_end = dis_ent["offsets"]
            insertions.append((d_start, "<disease> "))
            insertions.append((d_end, " </disease>"))
            
        # CRITICAL: Sort all collected insertions back-to-front
        insertions.sort(key=lambda x: x[0], reverse=True)
        
        # Inject all tags into a single text version
        text_list = list(text)
        for idx, tag in insertions:
            text_list.insert(idx, tag)
        masked_text = "".join(text_list)
        
        # Get standard string names for the final dataframe columns
        chem_name = chem_groups[chem_id][0]["text"].lower()
        dis_name = dis_groups[dis_id][0]["text"].lower()
        
        candidate_rows.append({
            "document_id": row["document_id"],
            "chemical": chem_name,
            "disease": dis_name,
            "chemical_id": chem_id,
            "disease_id": dis_id,
            "masked_text": masked_text,
            "label": label
        })
        
    return candidate_rows