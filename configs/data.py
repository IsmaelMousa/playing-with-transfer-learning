from dataclasses import dataclass
from enum import Enum

class ModelType(Enum):
    """
       Represents the special values for the model type.
    """
    RESNET50 = "resnet50"
    MOBILENET = "mobilenet"

@dataclass
class DataConfig:
    """
      Represents the interface of the data configuration.
    """
    img_size:tuple[int,int]
    validation_ratio:float
    test_ratio:float
    as_supervised:bool
    with_info:bool
    data_name:str
    model_type: ModelType
