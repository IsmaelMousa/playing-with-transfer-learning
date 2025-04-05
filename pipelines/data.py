import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import tensorflow.keras.applications.resnet50 as resnet
import tensorflow.keras.applications.mobilenet as mobilenet
from configs.data import DataConfig

np.random.seed(42)

class Data:
    def __init__(self, config: DataConfig):
        self.config = config
        validation_ratio=f"{config.validation_ratio * 100:.0f}%"
        test_ratio=f"{(config.test_ratio+config.validation_ratio) * 100:.0f}%"
        self.splits=[f"train[:{validation_ratio}]", f"train[{validation_ratio}:{test_ratio}]", f"train[{test_ratio}:]"]
        if config.preprocess == "resnet50":
            self.preprocess=resnet.preprocess_input
        else:
            self.preprocess=mobilenet.preprocess_input
        self.is_load=False
        self.test_set_raw=None
        self.validation_set_raw=None
        self.train_set_raw=None
        self.info=None

    def __to_numpy(self,dataset):
        """
        Converts the dataset to numpy array
        :param dataset: the img,label dataset
        :return: the two numpy arrays 1)images 2)labels
        """
        images, labels = [], []
        for img, label in dataset:
            img = img.numpy()
            img = np.array(Image.fromarray(img).resize(self.config.img_size))
            images.append(img)
            labels.append(label.numpy())
        return np.array(images), np.array(labels)

    def load(self):
        """
        Loads the data from a specific name
        :return: the loaded data and info
        """
        (test_set_raw, valid_set_raw, train_set_raw), info = tfds.load(
            self.config.data_name, split=self.splits,
            as_supervised=True, with_info=True)

        self.is_load=True
        self.test_set_raw=test_set_raw
        self.validation_set_raw=valid_set_raw
        self.train_set_raw=train_set_raw
        self.info=info
        return test_set_raw, valid_set_raw, train_set_raw, info


    def prepare(self):
        """
        prepares the data for the pre-train model
        :return: the prepared data and labels
        """
        if not self.is_load:
            raise Exception("You need to load the data first => data.load()")

        train_images, train_labels = self.__to_numpy(self.train_set_raw)
        valid_images, valid_labels = self.__to_numpy(self.validation_set_raw)
        test_images, test_labels = self.__to_numpy(self.test_set_raw)

        train_images = self.preprocess(train_images)
        valid_images = self.preprocess(valid_images)
        test_images = self.preprocess(test_images)

        class_names = self.info.features["label"].names

        return (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels), class_names