"""Configuration class for the biomedical extraction pipeline."""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Literal

import torch

@dataclass
class EnvConfig:
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    smoke_test: bool = False
    push_to_hub: bool = True



@dataclass
class DataConfig:
    """Configuration for the data."""
    data_folder: Path = Path("data/documents")
    eval_folder: Path = Path("evals")
    dataset_name: str = "bigbio/bc5cdr"
    flattened_dataset_name: Path = Path("data/processed/bc5cdr_flattened")
    relations_dataset_name: Path = Path("data/processed/bc5cdr_relations")
    structured_file: Path = Path("data/processed/structured_data")
    structured_checkpoint_file: Path = Path("data/checkpoints/structured_data")


@dataclass
class ModelConfig:
    """Model configurations."""
    
    # --- NER ---
    ner_model_name: str = "BiomedNLP-BiomedBERT-base-uncased-ner-abstract-bc5cdr-LoRA-v1.1"
    ner_model_checkpoint: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    ner_label_names: list[str] = field(default_factory=lambda: [
        "O", 
        "B-Chemical", "I-Chemical", 
        "B-Disease", "I-Disease"
    ])
    
    # --- Structured extraction ---
    client_tagger_checkpoint: str = "groq/qwen/qwen3-32b"
    local_tagger_checkpoint_small: str = "Qwen/Qwen2.5-7B-Instruct-GGUF"
    local_tagger_checkpoint_medium: str = "Qwen/Qwen2.5-14B-Instruct-GGUF"
    local_tagger_checkpoint_large: str = "Qwen/Qwen2.5-32B-Instruct-GGUF"

    # --- RE ---
    align_labels: bool = True
    re_model_name: str = "Clinical-Longformer-re-abstract-bc5cdr-LoRA-v1.1"
    re_model_checkpoint: str = "yikuan8/Clinical-Longformer"
    re_label_names_binary: list[str] = field(default_factory=lambda: [
        "No-Relation",
        "Chemical-Disease-Relation"
    ])
    re_label_names_multiclass: list[str] = field(default_factory=lambda: [
        "No-Relation", # label 0 
        "Chemical-Disease-Relation", # label 1
        "Chemical-Treats-Disease", # label 2
        "Chemical-Antagonizes-Chemical", # label 3
    ])


@dataclass
class TrackingCongif:
    # set a literal with wandb, trackio or none
    tracker: Literal["tensorboard", "wandb", "trackio", "none"] = "trackio"
    project: str = "Biomed-IE"
    trackio_space_id: str = "Francesco-A/nlp-tracking"

@dataclass
class CFG:
    env: EnvConfig = field(default_factory=EnvConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingCongif = field(default_factory=TrackingCongif)

    def to_dict(self):
        return asdict(self)

def load_configs():
    return CFG()
