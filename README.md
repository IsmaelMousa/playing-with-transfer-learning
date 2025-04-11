# Playing with Transfer Learning

Exploring the application of transfer learning using ResNet50 and MobileNetV2 for flower images classification task.

## Introduction
Transfer learning utilizes pre-trained models to adapt to new tasks with minimal data. This project evaluates the effectiveness of ResNet50 and MobileNetV2 in classifying flower images, aiming for high accuracy and efficient training.

## Methodology

Three main stages were followed: data preparation, model setup, and training stage.

### Data Preparation
The dataset was loaded and prepared with the following configurations: the input images were resized to 224×224. The data was split into 75% for training, 15% for testing, and 10% for validation. All images were converted to NumPy arrays, normalized, and one-hot encoded. Class names were dynamically extracted from the dataset metadata.

### Model Setup
The model architecture was built using a customizable OOP based system configured with ImageNet weights. The input shape was set to (224, 224, 3), and all backbone layers were frozen during the initial training phase. A custom head was added, consisting of global average pooling, configurable dense hidden layers, optional dropout and batch normalization, and a final output layer with softmax activation for five classes.


### Training
Training occurred in two phases. Phase 1 involved training only the classification head while keeping the pre-trained backbone frozen, using the SGD optimizer with a learning rate of 0.01 and momentum of 0.9. Metrics like accuracy, recall, precision, and F1-score were tracked over typically three epochs.
In Phase 2, fine-tuning was done by selectively unfreezing parts of the backbone, either by blocks or individual layers, with the top layers always unfrozen to adapt to the dataset.
## Results
After fine-tuning, the MobileNetV2 architecture achieved approximately 78% training accuracy and 70% testing accuracy. However, performance was relatively unstable, with fluctuations in loss and inconsistent precision and recall across classes. While some classes showed reasonable F1-scores, the overall model struggled with generalization, indicating limited effectiveness in this setup. Most errors occur between visually similar classes, such as daisy and dandelion.  

After fine-tuning, the ResNet50 architecture achieved approximately 90% accuracy on both the training and testing datasets.
However, when I applied a block-wise unfreezing strategy, the training process became unstable and the test accuracy dropped to 88%, with clear signs of overfitting.
In contrast, using a layer-wise unfreezing strategy resulted in stable training and maintained a 90% test accuracy without overfitting.


## Discussion
MobileNetV2 showed moderate adaptability to the TF-Flowers dataset, with transfer learning offering some benefits despite limited generalization. While ImageNet feature reuse helped speed up training and slightly reduce overfitting, performance remained unstable. Partial fine-tuning by layers gave slightly better results than block-based tuning, but the gains were minimal. Although the modular pipeline allowed flexible experimentation, MobileNetV2 may not be the best fit for this fine-grained task without further tuning or architectural changes.

ResNet50 demonstrated strong adaptability to the TF-Flowers dataset, with transfer learning significantly improving both training speed and accuracy. The reuse of ImageNet features contributed to faster convergence and reduced overfitting. Layer-wise fine-tuning provided more stable performance compared to block-wise tuning, which often led to instability and signs of overfitting. While block-based strategies yielded slightly lower test accuracy, the layer-based approach consistently reached around 90% without overfitting. Overall, ResNet50 proved to be a more robust choice for this fine-grained classification task, although further tuning may still enhance performance.

## Conclusion

This project highlights the strengths and limitations of transfer learning with ResNet50 and MobileNetV2. While both models offer efficient training, ResNet50's deeper architecture and residual connections provide better accuracy and stability for fine-grained classification tasks.