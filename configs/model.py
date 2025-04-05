from dataclasses import dataclass, field
from enum import Enum


class Type(Enum):
    """
    TODO
    """
    RESNET50  = "resnet50"
    MOBILENET = "mobilenet"


class By(Enum):
    """
    TODO
    """
    BLOCKS = "blocks"
    LAYERS = "layers"


@dataclass
class ModelConfig:
    """
    TODO
    """
    type                     : str | Type
    by                       : str | By
    classes                  : list
    convolutional_filters    : list        = field(default_factory=lambda: [])
    convolutional_kernels    : list        = field(default_factory=lambda: [])
    convolutional_activations: list        = field(default_factory=lambda: [])
    pooling_kernels          : list        = field(default_factory=lambda: [])
    hidden_neurons           : list        = field(default_factory=lambda: [])
    hidden_activations       : list        = field(default_factory=lambda: [])
    dropout_rates            : list        = field(default_factory=lambda: [0.2])
    metrics                  : list        = field(default_factory=lambda: ["accuracy", "recall", "precision", "f1_score"])
    epochs                   : list        = field(default_factory=lambda: [2, 2])
    input_shape              : tuple       = field(default_factory=lambda:(224, 224, 3))
    callbacks                : list | None = None
    loss                     : str         = "categorical_crossentropy"
    optimizer                : str         = "sgd"
    weights                  : str         = "imagenet"
    padding                  : str         = "same"
    output_activation        : str         = "softmax"
    convolutional_layers     : int         = 0
    pooling_layers           : int         = 0
    hidden_layers            : int         = 0
    dropout_layers           : int         = 1
    batch_normalization      : int         = 0
    blocks                   : int | None  = 2
    layers                   : int | None  = 56
    top                      : int | None  = 2
    patience                 : int         = 3
    batch_size               : int         = 32
    lr                       : float       = 0.01
    momentum                 : float       = 0.9
    global_average_pooling   : bool        = True
    info                     : bool        = True


    def __post_init__(self):
        self.type = Type(self.type).value
        self.by   = By(self.by).value

        if self.type not in {t.value for t in Type}: raise ValueError(f"Type: {self.type} is not supported.")
        if self.by   not in {t.value for t in By}  : raise ValueError(f"By: {self.by} is not supported.")
