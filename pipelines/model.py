from warnings import filterwarnings; filterwarnings("ignore")
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model as KModel
from tensorflow.keras.applications import resnet as res, mobilenet_v2 as mob
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam, Ftrl
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from configs import ModelConfig

class Model:
    """
    TODO
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
        TODO
        """
        if self.config.type == "resnet50": base = res.ResNet50   (input_shape=self.config.input_shape,
                                                                  weights    =self.config.weights,
                                                                  include_top=False)

        else                             : base = mob.MobileNetV2(input_shape=self.config.input_shape,
                                                                  weights    =self.config.weights,
                                                                  include_top=False)

        base.trainable = False

        inputs         = Input(shape=self.config.input_shape, batch_size=self.config.batch_size)
        outputs        = base(inputs, training=False)

        for index in range(self.config.convolutional_layers):
            outputs = Conv2D(filters    =self.config.convolutional_filters[index],
                             kernel_size=self.config.convolutional_kernels[index],
                             activation =self.config.convolutional_activations[index],
                             padding    =self.config.padding)(outputs)

            if self.norm   : outputs = BatchNormalization()(outputs)
            if self.dropout: outputs = Dropout(rate=self.config.dropout_rates[index])(outputs)

            outputs = MaxPooling2D(pool_size=self.config.pooling_kernels[index])(outputs)


        dropout        = self.config.dropout_layers      == self.config.hidden_layers
        norm           = self.config.batch_normalization == self.config.hidden_layers

        if self.config.global_average_pooling: outputs = GlobalAveragePooling2D()(outputs)

        for index in range(self.config.hidden_layers):
            outputs = Dense(units     =self.config.hidden_neurons[index],
                            activation=self.config.hidden_activations[index])(outputs)

            if norm   : outputs = BatchNormalization()(outputs)
            if dropout: outputs = Dropout(rate=self.config.dropout_rates[index])(outputs)

        outputs        = Dense(units=len(self.config.classes), activation=self.config.output_activation)(outputs)

        self.model     = KModel(inputs=inputs, outputs=outputs)

        if self.config.info:
            print(f"Total layers in model: {len(self.model.layers)}")
            print(f"Base model trainable : {base.trainable}")
            print(f"Trainable weights    : {len(self.model.trainable_weights)}")
            print(f"Non-trainable weights: {len(self.model.non_trainable_weights)}")

        return self.model


    def summary(self):
        """
        TODO
        """
        self.model.summary()


    def compile(self):
        """
        TODO
        """
        optimizer = self.optimizer(learning_rate=self.config.lr, momentum=self.config.momentum)

        self.model.compile(optimizer=optimizer, loss=self.config.loss, metrics=self.config.metrics)


    def train(self, splits):
        """
        TODO
        """
        (train_images, train_labels), (valid_images, valid_labels), _ = splits

        return self.model.fit(x              =train_images,
                              y              =train_labels,
                              epochs         =self.config.epochs[0],
                              batch_size     =self.config.batch_size,
                              validation_data=(valid_images, valid_labels),
                              callbacks      =self.config.callbacks)


    def evaluate(self, splits):
        """
        TODO
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


    def predict(self, inputs):
        """
        TODO
        """
        return self.model.predict(inputs)


    @staticmethod
    def visualize_performance(history, metric):
        """
        TODO
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


    def visualize_misclassifications(self, splits):
        """
        TODO
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


    def visualize_confusion_matrix(self, splits):
        """
        TODO
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

# TODO
from PIL import Image
import tensorflow_datasets as tfds

def to_numpy(dataset, img_size):
    images, labels = [], []

    for img, label in dataset:
        img = img.numpy()
        img = np.array(Image.fromarray(img).resize(img_size))
        images.append(img)
        labels.append(label.numpy())

    labels = tf.keras.utils.to_categorical(np.array(labels)) # TODO

    return np.array(images), np.array(labels)

def load_and_prepare(img_size=(224, 224)):
    splits = ["train[:10%]", "train[10%:25%]", "train[25%:]"]

    (test_set_raw, valid_set_raw, train_set_raw), info = tfds.load("tf_flowers", split=splits, as_supervised=True, with_info=True)

    train_images, train_labels = to_numpy(train_set_raw, img_size)
    valid_images, valid_labels = to_numpy(valid_set_raw, img_size)
    test_images , test_labels  = to_numpy(test_set_raw , img_size)

    train_images = mob.preprocess_input(train_images)
    valid_images = mob.preprocess_input(valid_images)
    test_images  = mob.preprocess_input(test_images)

    class_names = info.features["label"].names

    return (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels), class_names

# TODO
np.random.seed(42)
tf.random.set_seed(42)

# (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels), classes = load_and_prepare()
#
# splits = (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels)
#
# config = ModelConfig(type="mobilenet", by="blocks", classes=classes)
# model  = Model(config=config)
#
# model.build()
# model.compile()
# history = model.train(splits=splits)
# metrics  = model.evaluate(splits=splits[-1])
# model.visualize_misclassifications(splits=splits[-1])
# model.visualize_confusion_matrix(splits=splits[-1])
# model.visualize_performance(history=history, metric="f1_score")