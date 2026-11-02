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


def _tag_candidate_pair(
    text: str,
    chemical_spans: list[tuple[int, int]],
    disease_spans: list[tuple[int, int]],
) -> str:
    insertions = []
    for start, end in chemical_spans:
        insertions.append((start, "<chemical> "))
        insertions.append((end, " </chemical>"))
    for start, end in disease_spans:
        insertions.append((start, "<disease> "))
        insertions.append((end, " </disease>"))

    # Note: insert in reverse order, to prevent offset shifting
    insertions.sort(key=lambda item: item[0], reverse=True)
    text_list = list(text)
    for index, tag in insertions:
        text_list.insert(index, tag)
    return "".join(text_list)


def create_labeled_candidate_pairs(row: pd.Series) -> list[dict]:
    """
    Create MeSH-grouped, labeled candidate pairs from a gold-annotated row.

    Args:
        row: A row containing ``text``, ``entities`` with ``mesh_id`` and
            ``offsets``, and optionally gold ``relations``.

    Returns:
        Candidate records with tagged text, MeSH IDs, and binary gold labels.

    Mentions sharing a MeSH ID are grouped, and labels come from the row's
    gold relations. For unlabeled NER predictions without MeSH IDs, use
    :func:`make_inference_candidate_pairs` instead.
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
        
        chemical_spans = [tuple(entity["offsets"]) for entity in chem_groups[chem_id]]
        disease_spans = [tuple(entity["offsets"]) for entity in dis_groups[dis_id]]
        masked_text = _tag_candidate_pair(text, chemical_spans, disease_spans)
        
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


def make_inference_candidate_pairs(text: str, entities: list[dict]) -> list[dict]:
    """Create unlabeled, mention-level candidate pairs from predicted NER spans.

    Args:
        text: Source sentence or document whose offsets are used by the NER output.
        entities: Token-classification pipeline entities with ``entity_group``
            (``Chemical`` or ``Disease``), ``start``, and ``end`` fields.

    Returns:
        One tagged candidate record per predicted chemical-mention/disease-mention
        pair. Records contain the mention strings and ``masked_text`` but no label
        or MeSH IDs, since neither is available at inference time.

    Unlike :func:`create_labeled_candidate_pairs`, this function pairs individual
    predicted mentions and does not consult gold relations. Use the labeled helper
    when preparing annotated training or evaluation rows.
    """
    chemicals = [entity for entity in entities if entity["entity_group"] == "Chemical"]
    diseases = [entity for entity in entities if entity["entity_group"] == "Disease"]
    candidate_rows = []

    for chemical, disease in itertools.product(chemicals, diseases):
        masked_text = _tag_candidate_pair(
            text,
            [(chemical["start"], chemical["end"])],
            [(disease["start"], disease["end"])],
        )
        candidate_rows.append({
            "sentence": text,
            "chemical": text[chemical["start"]:chemical["end"]],
            "disease": text[disease["start"]:disease["end"]],
            "masked_text": masked_text,
        })

    return candidate_rows


def merge_datasets_on_key(
        dataset1: Dataset, 
        dataset2: Dataset, 
        on: str = "pair_id", 
        how: str = "inner"
) -> Dataset:
    """Merge two Dataset instances on `on` key using `pandas.DataFrame.merge()`."""
    df1 = dataset1.to_pandas()
    df2 = dataset2.to_pandas()

    # Merge on key
    merged_df = pd.merge(df1, df2, on=on, suffixes=("", "__drop__"))
    merged_df = merged_df.drop(columns=[col for col in merged_df if col.endswith("__drop__")])

    # Create new Dataset from merged DataFrame
    return Dataset.from_pandas(merged_df)


def create_balanced_stratified_subset(
    dataset_dict: DatasetDict,
    n_samples: int = 64,
    seed: int = 42,
    label_column: str = "labels"
) -> DatasetDict:
    """
    Downsamples a Hugging Face DatasetDict so that each split contains 
    an equal number of samples per label category.
    """
    downsampled_splits = {}
    for split, ds in dataset_dict.items():
        df = ds.to_pandas()
        
        # If the split is already smaller than or equal to n_samples, keep it as is
        if len(df) <= n_samples:
            downsampled_splits[split] = ds
            continue
            
        unique_labels = sorted(df[label_column].unique())
        n_labels = len(unique_labels)
        
        # Calculate exact samples per label (e.g., 64 / 4 = 16)
        samples_per_label = n_samples // n_labels
        
        sampled_dfs = []
        for label in unique_labels:
            df_label = df[df[label_column] == label]
            sampled_label_df = df_label.sample(
                n=min(len(df_label), samples_per_label), 
                random_state=seed
            )
            sampled_dfs.append(sampled_label_df)
            
        # Combine and shuffle the stratified subset
        df_subset = pd.concat(sampled_dfs).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        
        # Convert back to Hugging Face Dataset
        downsampled_splits[split] = Dataset.from_pandas(df_subset, preserve_index=False)
        
    return DatasetDict(downsampled_splits)