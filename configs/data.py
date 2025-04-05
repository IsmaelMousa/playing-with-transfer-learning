from dataclasses import dataclass

@dataclass
class DataConfig:
    """
      Represents the interface of the data configuration.

      Attributes:
        - img_size (tuple[int, int]): The target size (height, width) for image resizing.
        - validation_ratio (float): The fraction of data to be used for validation.
        - test_ratio (float): The fraction of data to be used for testing.
        - as_supervised (bool): Whether to load the dataset in (image, label) format.
        - with_info (bool): Whether to include additional metadata when loading the dataset.
        - data_name (str): The name of the dataset to be loaded.
        - preprocess (Literal["resnet50", "mobilenet"]): The name of the preprocessing model to apply.
    """
    img_size:tuple[int,int]
    validation_ratio:float
    test_ratio:float
    as_supervised:bool
    with_info:bool
    data_name:str
    preprocess:str

    def __post_init__(self):
        """
        Check the preprocess name before create config object.
        """
        allowed = {"resnet50", "mobilenet"}
        if self.preprocess not in allowed:
            raise ValueError(f"Invalid preprocess: {self.preprocess} the preprocess name must be one of {allowed}")
