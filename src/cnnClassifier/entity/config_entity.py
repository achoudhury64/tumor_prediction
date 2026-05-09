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


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path #from config.yaml
    trained_model_path: Path #from config.yaml
    updated_base_model_path: Path #the path is artifacts/prepare basemodel/base_model_updated.h5
    training_data: Path #The training data is in artifacts/data_ingestion. Will be provided when calling, 
    #from get_training_config(self) in this notebook
    params_epochs: int #from params.yaml 
    params_batch_size: int #from params.yaml 
    params_is_augmentation: bool #from params.yaml 
    params_image_size: list #from params.yaml 


@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict  #in earlier notebooks, params were taken separately, now the are taken at once as a dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int