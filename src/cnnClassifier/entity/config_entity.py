from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path # defined in config.yaml
    base_model_path: Path # defined in config.yaml
    updated_base_model_path: Path # defined in config.yaml
    params_image_size: list # defined in params.yaml
    params_learning_rate: float # defined in params.yaml
    params_include_top: bool # defined in params.yaml
    params_weights: str # defined in params.yaml

    params_classes: int # defined in params.yaml