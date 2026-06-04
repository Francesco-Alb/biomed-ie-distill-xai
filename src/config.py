"""Configuration class for the biomedical extraction pipeline."""

from dataclasses import dataclass, asdict, field
from pathlib import Path

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
    ner_model_name: str = "BiomedNLP-PubMedBERT-base-uncased-ner-abstract-bc5cdr"
    ner_model_checkpoint: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased"


@dataclass
class RetrievalConfig:
    pass

@dataclass
class CFG:
    env: EnvConfig = field(default_factory=EnvConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    def to_dict(self):
        return asdict(self)

def load_configs():
    return CFG()
