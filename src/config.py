"""Configuration class for the biomedical extraction pipeline."""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Literal

@dataclass
class EnvConfig:
    seed: int = 42


@dataclass
class DataConfig:
    """Configuration for the data."""
    dataset_name: str = "bigbio/bc5cdr"
    data_folder: Path = Path("data/documents")


@dataclass
class ModelConfig:
    """Configuration for the model."""
    # --- NER ---
    ner_model_name: str = "BiomedNLP-PubMedBERT-base-uncased-ner-abstract-bc5cdr-LoRA-v1.1"
    ner_model_checkpoint: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    ner_label_names: list[str] = field(default_factory=lambda: [
        "O", 
        "B-Chemical", "I-Chemical", 
        "B-Disease", "I-Disease"
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
