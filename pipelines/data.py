import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import tensorflow.keras.applications.resnet50 as resnet
import tensorflow.keras.applications.mobilenet as mobilenet
from configs.data import DataConfig

np.random.seed(42)

class Data:
    def __init__(self, data_config: DataConfig):
        self.data_config = data_config
        validation_ratio=f"{data_config.validation_ratio * 100:.0f}%"
        test_ratio=f"{(data_config.test_ratio+data_config.validation_ratio) * 100:.0f}%"
        self.splits=[f"train[:{validation_ratio}]", f"train[{validation_ratio}:{test_ratio}]", f"train[{test_ratio}:]"]

    def __to_numpy(self,dataset):
        """
        Converts the dataset to numpy array
        :param dataset: the img,label dataset
        :return: the two numpy arrays 1)images 2)labels
        """
        images, labels = [], []
        for img, label in dataset:
            img = img.numpy()
            img = np.array(Image.fromarray(img).resize(self.data_config.img_size))
            images.append(img)
            labels.append(label.numpy())
        return np.array(images), np.array(labels)


    def load_and_prepare(self):
        """
        Loads the data from a specific name and prepares the data for the pre-train model
        :return: the prepared data and labels
        """
        (test_set_raw, valid_set_raw, train_set_raw), info = tfds.load(
            self.data_config.data_name, split=self.splits,
            as_supervised=self.data_config.as_supervised, with_info=self.data_config.with_info)

        train_images, train_labels = self.__to_numpy(train_set_raw)
        valid_images, valid_labels = self.__to_numpy(valid_set_raw)
        test_images, test_labels = self.__to_numpy(test_set_raw)

        if self.data_config.model_type.value == "resnet50":
            train_images = resnet.preprocess_input(train_images)
            valid_images = resnet.preprocess_input(valid_images)
            test_images = resnet.preprocess_input(test_images)

        else:
            train_images = mobilenet.preprocess_input(train_images)
            valid_images = mobilenet.preprocess_input(valid_images)
            test_images = mobilenet.preprocess_input(test_images)

        class_names = info.features["label"].names

        return (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels), class_names