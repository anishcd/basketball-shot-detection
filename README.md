# Basketball Shooting Analysis using Faster R-CNN Object Detection Architecture
## Overview

This project aims to detect basketball shots in images/videos and classify them as makes and misses using object detection techniques. Specifically, the project employs the Faster R-CNN architecture with a ResNet-50 backbone and FPN for this purpose. The project is implemented using PyTorch.

## Data

The dataset is in COCO format and contains annotations for the following classes:
- Basketball
- Rim

The dataset is divided into 3 sets (Training set, Validation Set, Test set) for cross-validation while the model was being trained:

[Link to dataset download in COCO JSON Format](https://universe.roboflow.com/uc-berkely-w210-tracer/tracer-basketball/dataset/3/download)

Object detection models usually follow a two-step process. First, the model scans the input image for possible object locations (often referred to as "proposals"). In the second step, the model classifies each proposal into different categories and refines their bounding boxes. Before feeding the images into the model, they undergo preprocessing steps to convert them into a format that can be used for training. The processing for this case was done in the functions located in /utils.py.

- **Annotations**: Annotations are stored in COCO (Common Objects in Context) format, which is a widely used format for object detection.
- **Bounding Boxes**: The bounding box annotations in the COCO format are converted into the `[xmin, ymin, xmax, ymax]` format required for training the model.
- **Labels**: The labels (object classes) are also preprocessed to be compatible with the model.

## Model Architecture

The model architecture is based on Faster R-CNN with a ResNet-50 backbone and Feature Pyramid Network (FPN).

- **Faster R-CNN**: This is a two-stage object detection model that first proposes candidate object bounding boxes and then classifies them.
- **ResNet-50**: This is the backbone network used for feature extraction.
- **FPN**: Feature Pyramid Network is used to enhance the feature maps obtained from ResNet-50, making the model more robust to varying object sizes.

The model is trained using Stochastic Gradient Descent (SGD) with a learning rate of 0.0001, momentum of 0.9, and weight decay of 0.0005.

- **Loss Function**: The model is trained using a multi-task loss, which is a combination of classification loss and regression loss for the bounding boxes. Specifically, log loss is used for classification, and smooth L1 loss is used for bounding box regression.
- **Backpropagation**: The gradients are backpropagated through the network to update the weights.

## Configuration

Configuration options like data paths, model architecture, and hyperparameters are stored in the `config.yaml` file.

## Training

The model used was trained using the following hyperparameters:

- Learning rate: 0.0001
- Batch size: 5
- Epochs: 20
- Momentum: 0.9
- Weight decay: 0.0005

The training script uses early stopping based on validation loss to prevent overfitting.
