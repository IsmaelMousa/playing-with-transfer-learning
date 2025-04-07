from dataclasses import dataclass, field
from enum import Enum


class Type(Enum):
    """
    Represents the available model architectures.
    """
    RESNET50  = "resnet50"
    MOBILENET = "mobilenet"


class By(Enum):
    """
    Represents the possible strategies for finetuning.
    """
    BLOCKS = "blocks"
    LAYERS = "layers"


@dataclass
class ModelConfig:
    """
    Configuration settings for the model architecture, building, training, and evaluation.

    - type: the model architecture type (e.g., ResNet50, MobileNet).
    - by: the finetuning strategy (e.g., 'blocks' or 'layers').
    - classes: the class names for classification.
    - convolutional_filters: the number of filters for each convolutional layer.
    - convolutional_kernels: the size of the kernels for each convolutional layer.
    - convolutional_activations: the activation functions for each convolutional layer.
    - pooling_kernels: the size of the kernels for each pooling layer.
    - hidden_neurons: the number of neurons in each hidden layer.
    - hidden_activations: the activation functions for each hidden layer.
    - dropout_rates: the dropout rates for each layer.
    - metrics: the evaluation metrics (e.g., accuracy, recall, precision, f1_score).
    - epochs: the number of training epochs, the first is for training phase, the second is for finetuning phase.
    - input_shape: the input shape of the data (e.g., (224, 224, 3) for images).
    - callbacks: the callback functions for model training.
    - loss: the loss function to be used during training (e.g., 'categorical_crossentropy').
    - optimizer: the optimizer to be used during training (e.g., 'sgd', 'adam').
    - weights: the pretrained weights to use (e.g. 'imagenet').
    - padding: the padding type for convolutional layers (e.g., 'same', 'valid').
    - output_activation: the activation function for the output layer (e.g., 'softmax').
    - convolutional_layers: the number of convolutional layers.
    - pooling_layers: the number of pooling layers.
    - hidden_layers: the number of hidden layers.
    - dropout_layers: the number of layers that use dropout.
    - batch_normalization: the number of layers with batch normalization.
    - blocks: the number of blocks to finetune when using block strategy.
    - layers: the number of layers to finetune when using layer strategy.
    - top: the number of top layers (from the end of the model) to finetune.
    - patience: the early stopping patience.
    - batch_size: the batch size for training.
    - lr: the learning rate for the optimizer.
    - momentum: the momentum for optimizers that support it (e.g., SGD).
    - global_average_pooling: whether to use global average pooling.
    - info: whether to print additional model information during training.
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
    epochs                   : list        = field(default_factory=lambda: [3, 5])
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
        self.by   = self.by

        assert self.type in {t.value for t in Type}
        assert self.by   in {b.value for b in By}
