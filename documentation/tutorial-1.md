---
title: Application Development Concepts
---

# Application Development Concepts

## Introduction



## Outline

- Model Retraining in Deep Learning
- MobileNet
- Object Recognition
- Bottlenecks to the Model Improving in Object Recognition

## Model Retraining in Deep Learning

### Training and Tensorboard

Once the script finishes generating all the bottleneck files, the actual training of the final layer of the network begins.

The script runs 4,000 training steps. With each step, 10 images at random are chosen from the training set, their bottlenecks are found from cache and they are fed into the final layer to get predictions. Those predictions are compared against the actual labels, and the results of this comparison is used to update the final layer’s weights through a backpropagation process.

As the script trains, you’ll see a series of step outputs, each showing training accuracy, validation accuracy and cross entropy:

- **Training accuracy**: shows percentage of images used in current training batch labeled with correct class
- **Validation accuracy**: is the precision (percentage of correctly-labelled images) on randomly-selected group of images from a different set
- **Cross entropy**: is a loss function that gives a glimpse into how well the learning process is progressing (lower numbers are better).

Show graph of training vs validation accuracy. If the training accuracy orange continues to increase while the validation accuracy decreases, the model is overfitting.

## MobileNet

## Object Recognition

## Bottlenecks to the Model Improving in Object Recognition

### Bottlenecks

ImageNet doesn’t include flower species we’re training on here. The kinds of info that make it possible for ImageNet to differentiate among 1000 classes are useful for distinguishing other objects. By using this pre-trained network, we are using that information as input to final or last classification layer that distinguishes our flower classes.

Bottleneck is a term in deep learning that signifies the layer prior to the final layer that does the classification. Calculating the layers behind the bottleneck for each image takes a significant amount of time. Since these lower layers of the network aren’t being modified their outputs can be cached and reused. So, the script runs the constant part of the network, everything below the node labeled Bottleneck above and caching the results.
