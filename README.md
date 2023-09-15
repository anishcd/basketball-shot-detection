# Basketball Shooting Analysis using Faster R-CNN Object Detection Architecture
## Overview

This project aims to detect basketball shots in images/videos and classify them as makes and misses using object detection techniques. Specifically, the project employs the Faster R-CNN architecture with a ResNet-50 backbone and FPN for this purpose. The project is implemented using PyTorch.

## Data

The dataset is in COCO format and contains annotations for the following classes:
- Basketball
- Rim

The dataset is divided into 3 sets for cross-validation after model was trained:
- Training set
- Validation set
- Test set

[Link to dataset download in COCO JSON Format](https://universe.roboflow.com/uc-berkely-w210-tracer/tracer-basketball/dataset/3/download)

## Configuration

Configuration options like data paths, model architecture, and hyperparameters are stored in a `config.yaml` file.

### Model Architecture

```yaml
model:
  backbone: ResNet50
  num_classes: 4
```

## Training

The model is trained using the following hyperparameters:

- Learning rate: 0.0001
- Batch size: 5
- Epochs: 20
- Momentum: 0.9
- Weight decay: 0.0005

The training script uses early stopping based on validation loss to prevent overfitting.
