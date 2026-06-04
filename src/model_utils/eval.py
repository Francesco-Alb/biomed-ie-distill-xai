def calculate_f1(predicted_tuples, gold_tuples):
    """
    predicted_tuples: set of (abstract_id, chem_mesh, disease_mesh)
    gold_tuples: set of (abstract_id, chem_mesh, disease_mesh)
    """
    tp = len(predicted_tuples.intersection(gold_tuples))
    fp = len(predicted_tuples - gold_tuples)
    fn = len(gold_tuples - predicted_tuples)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1_score": f1}