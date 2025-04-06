import warnings; warnings.filterwarnings("ignore")
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import re

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import Model as KModel
from tensorflow.keras.applications import resnet as res, mobilenet_v2 as mob
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam, Ftrl
from tensorflow.keras.callbacks import History
from sklearn.metrics import  confusion_matrix

from configs import ModelConfig


class Model:
    """
    Represents the operation of the model, with methods for building, training, evaluating, and visualizing.
    Supports architectures like ResNet50 and MobileNetV2, offers configurable layers, optimizers, and training settings.
    """
    def __init__(self, config: ModelConfig):
        self.config    = config
        self.dropout   = config.dropout_layers      == config.convolutional_layers
        self.norm      = config.batch_normalization == config.convolutional_layers
        self.optimizer = {"sgd"     : SGD,      "adam"   : Adam,
                          "rmsprop" : RMSprop,  "adagrad": Adagrad,
                          "adadelta": Adadelta, "adamax" : Adamax,
                          "nadam"   : Nadam,    "ftrl"   : Ftrl}[config.optimizer]

        assert config.hidden_layers                 == len(config.hidden_neurons)
        assert config.hidden_layers                 == len(config.hidden_activations)
        assert config.convolutional_layers          == len(config.convolutional_filters)
        assert config.convolutional_layers          == len(config.convolutional_kernels)
        assert config.convolutional_layers          == len(config.convolutional_activations)
        assert config.pooling_layers                == len(config.pooling_kernels)
        assert config.dropout_layers                == len(config.dropout_rates)


    def build(self):
        """
        Builds the model based on the provided configuration.

        :return: the compiled keras model.
        """
        if self.config.type == "resnet50": self.base = res.ResNet50   (input_shape=self.config.input_shape,
                                                                       weights    =self.config.weights,
                                                                       include_top=False)

        else                             : self.base = mob.MobileNetV2(input_shape=self.config.input_shape,
                                                                       weights    =self.config.weights,
                                                                       include_top=False)

        self.base.trainable = False

        inputs              = Input(shape=self.config.input_shape, batch_size=self.config.batch_size)

        outputs             = self.base(inputs, training=False)

        for index in range(self.config.convolutional_layers):
            outputs = Conv2D(filters    =self.config.convolutional_filters[index],
                             kernel_size=self.config.convolutional_kernels[index],
                             activation =self.config.convolutional_activations[index],
                             padding    =self.config.padding)(outputs)

            if self.norm   : outputs = BatchNormalization()(outputs)

            if self.dropout: outputs = Dropout(rate=self.config.dropout_rates[index])(outputs)

            outputs = MaxPooling2D(pool_size=self.config.pooling_kernels[index])(outputs)


        dropout             = self.config.dropout_layers      == self.config.hidden_layers

        norm                = self.config.batch_normalization == self.config.hidden_layers

        if self.config.global_average_pooling: outputs = GlobalAveragePooling2D()(outputs)

        for index in range(self.config.hidden_layers):
            outputs = Dense(units=self.config.hidden_neurons[index], activation=self.config.hidden_activations[index])(outputs)

            if norm   : outputs = BatchNormalization()(outputs)

            if dropout: outputs = Dropout(rate=self.config.dropout_rates[index])(outputs)

        outputs             = Dense(units=len(self.config.classes), activation=self.config.output_activation)(outputs)

        self.model          = KModel(inputs=inputs, outputs=outputs)

        if self.config.info:
            print(f"Total layers in model: {len(self.model.layers)}")
            print(f"Base model trainable : {self.base.trainable}")
            print(f"Trainable weights    : {len(self.model.trainable_weights)}")
            print(f"Non-trainable weights: {len(self.model.non_trainable_weights)}")

        return self.model


    def summary(self):
        """
        Prints the summary of the model architecture.
        """
        self.model.summary()


    def compile(self):
        """
        Compiles the model with the specified optimizer, loss function, and metrics.
        """
        optimizer = self.optimizer(learning_rate=self.config.lr, momentum=self.config.momentum)

        self.model.compile(optimizer=optimizer, loss=self.config.loss, metrics=self.config.metrics)


    def train(self, splits: tuple):
        """
        Trains the model on the provided dataset splits.

        :param splits: two splits; one for training and one for validation sets
        :return: the training history.
        """
        x, y            = splits[0]
        validation_data = splits[1]

        return self.model.fit(x              =x,
                              y              =y,
                              epochs         =self.config.epochs[0],
                              batch_size     =self.config.batch_size,
                              validation_data=validation_data,
                              callbacks      =self.config.callbacks)


    def __get_block_number(self, layer: str):
        """
        Extracts the block number from the layer name for finetuning.

        :param layer: the layer name.
        :return: the block number.
        """
        if self.config.type == "resnet50":
            match = re.search(r"block(\d+)", layer)
            return int(match.group(1)) if match else None

        else:
            match = re.search(r"block_(\d+)", layer)
            return int(match.group(1)) if match else None


    def finetune(self, splits: tuple):
        """
        Performs finetuning on specific layers or blocks based on the configuration.

        :param splits: two splits; one for training and one for validation sets.
        :return: the training history.
        """
        if self.config.by == "blocks":
            self.model.trainable = False

            base = self.base

            block_layers = {}

            for layer in base.layers:
                block = self.__get_block_number(layer=layer.name)

                if block is not None: block_layers.setdefault(block, []).append(layer)

            sorted_blocks = sorted(block_layers.keys(), reverse=True)

            for block in sorted_blocks[:self.config.blocks]:

                for layer in block_layers[block]: layer.trainable = True

            for layer in self.model.layers[self.config.top:]: layer.trainable = True

            trainable = [layer.name for layer in self.model.layers if layer.trainable] + [layer.name for layer in self.base.layers if layer.trainable]

            if self.config.info:
                print(f"Trainable layers      : {trainable}")
                print(f"Total trainable layers: {len(trainable)}")

            self.compile()

            return self.train(splits)

        else:
            base_layers = [layer for layer in self.model.layers if layer.name.startswith("block")]

            if self.config.layers < len(base_layers):
                for layer in base_layers[self.config.layers:]: layer.trainable = True

            for layer in self.model.layers[-self.config.top:]: layer.trainable = True

            trainable = [layer.name for layer in self.model.layers if layer.trainable]

            if self.config.info:
                print(f"Trainable layers      : {trainable}")
                print(f"Total trainable layers: {len(trainable)}")

            self.compile()

            return self.train(splits)


    def evaluate(self, splits: tuple):
        """
        Evaluates the model on the provided dataset.

        :param splits: the evaluation or testing sets.
        :return: the evaluation or testing metrics.
        """
        x, y      = splits

        metrics   = self.model.evaluate(x=x, y=y)
        loss      = metrics[0]
        accuracy  = metrics[1]
        recall    = metrics[2]
        precision = metrics[3]
        f1        = np.mean(metrics[4])

        print("-" * 16)
        print(f"Loss     : {loss     :.2f}")
        print(f"Accuracy : {accuracy :.2f}")
        print(f"Recall   : {recall   :.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"F1 score : {f1       :.2f}")
        print("-" * 16)

        if self.config.info:
            predictions      = self.model.predict(x=x)
            predicted_labels = np.argmax(a=predictions, axis=1)
            actual_labels    = np.argmax(a=y, axis=1)
            indices          = np.random.choice(a=x.shape[0], size=9)
            classes          = self.config.classes

            plt.figure(figsize=(10, 10))
            for i, idx in enumerate(indices):
                plt.subplot(3, 3, i + 1)
                image     = (x[idx] + 1) / 2.0
                plt.imshow(image)
                actual    = classes[actual_labels[idx]]
                predicted = classes[predicted_labels[idx]]
                color     = "green" if actual == predicted else "red"
                plt.title(f"Actual: {actual}\nPredicted: {predicted}", color=color)
                plt.axis("off")
            plt.tight_layout()
            plt.show()

        return metrics


    def predict(self, inputs: np.ndarray):
        """
        Predicts the labels for the given input data.

        :param inputs: the input images.
        :return: the predicted labels.
        """
        return self.model.predict(inputs)


    @staticmethod
    def visualize_performance(history: History, metric: str):
        """
        Visualizes the performance of the model over epochs for a specific metric.

        :param history: the training history.
        :param metric: the metric to visualize.
        """
        if metric == "f1_score":
            train_f1 = [np.mean(values) for values in history.history[metric]]
            valid_f1 = [np.mean(values) for values in history.history[f"val_{metric}"]]

            plt.plot(train_f1, label="Training f1_score", color="green", linestyle="-", linewidth=2)
            plt.plot(valid_f1, label="Validation f1_score", color="red",linestyle="--", linewidth=2)

        else:
            train_f1 = history.history[metric]
            valid_f1 = history.history[f"val_{metric}"]

            plt.plot(train_f1, label=f"Training {metric} ", color="green", linestyle="-", linewidth=2)
            plt.plot(valid_f1, label=f"Validation {metric}", color="red", linestyle="--", linewidth=2)

        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(visible=True, linestyle="--", alpha=0.7)
        plt.title(f"{metric.capitalize()} Over Epochs")
        plt.tight_layout()
        plt.show()


    def visualize_misclassifications(self, splits: tuple):
        """
        Visualizes misclassified samples in a 3x3 grid.

        :param splits: the evaluation or testing sets.
        """
        x, y             = splits

        predictions      = self.model.predict(x=x)
        predicted_labels = np.argmax(a=predictions, axis=1)
        actual_labels    = np.argmax(a=y, axis=1)
        indices          = np.where(predicted_labels != actual_labels)[0]
        classes          = self.config.classes
        misclassified    = np.random.choice(a=indices, size=9, replace=False)

        plt.figure(figsize=(10, 10))
        for i, idx in enumerate(misclassified):
            plt.subplot(3, 3, i + 1)
            image     = (x[idx] + 1) / 2.0
            plt.imshow(image)
            actual    = classes[actual_labels[idx]]
            predicted = classes[predicted_labels[idx]]
            plt.title(f"Actual: {actual}\nPredicted: {predicted}", color="red")
            plt.axis("off")
        plt.tight_layout()
        plt.show()


    def visualize_confusion_matrix(self, splits: tuple):
        """
        Visualizes the confusion matrix for model predictions.

        :param splits: the evaluation or testing sets.
        """
        x, y        = splits

        predictions = self.model.predict(x=x)
        predicted   = np.argmax(a=predictions, axis=1)
        actual      = np.argmax(a=y, axis=1)
        cm          = confusion_matrix(actual, predicted)
        classes     = self.config.classes

        plt.figure(figsize=(8, 6))
        sns.heatmap(data=cm, annot=True, fmt="d", cmap="Reds", cbar=False, xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title ("Confusion Matrix")
        plt.tight_layout()
        plt.show()
