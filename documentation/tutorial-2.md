---
title: Performing Image Recognition
---

# Performing Image Recognition

## Introduction

You will explore setting up a development environment with TensorFlow, use the Deep Learning Framework to perform object classification in an image and present the results.

## Prerequisites

- Basic understanding of Linux commands
- Installed Docker on your local host machine
- [Installed git on Ubuntu Docker](https://stackoverflow.com/questions/29929534/docker-error-unable-to-locate-package-git)
- Installed wget on Ubuntu Docker

## Outline

- Setup Development Environment
- Perform Object Classification in Image

## Setup Development Environment

### Option 1: Install TensorFlow Docker Container

Open your terminal, linux shell or windows ubuntu shell. Run the command to download the docker **tensorflow** image:

~~~bash
git clone https://github.com/james94/tensorflow.git
~~~

~~~bash
cd ~/github_repos/james-repos/tensorflow/tensorflow/tools/dockerfiles
~~~

~~~bash
docker build -f dockerfiles/cpu.Dockerfile -t tensorflow .
~~~

Run the command to deploy the docker image as a container:

~~~bash
docker run -p 8888:8888 -p 6006:6006 -it tensorflow bash
~~~

Jump to **Clone TensorFlow for Poets Application**.

### Option 2: Install TensorFlow in Virtual Machine

On a virtual machine running linux, run the command to install the latest **tensorflow**:

~~~bash
pip install--upgrade "tensorflow>=1.7.*"
pip install tensorflow-hub
~~~

### Clone TensorFlow for Poets Application

~~~bash
git clone https://github.com/googlecodelabs/tensorflow-for-poets-2
cd tensorflow-for-poets-2
~~~

### Download the Training Images

~~~bash
curl -LO http://download.tensorflow.org/example_images/flower_photos.tgz
tar -xf flower_photos.tgz -C tf_files
~~~

~~~bash
ls tf_files/flower_photos
~~~

### Configure MobileNet Convolutional Neural Network

~~~bash
IMAGE_SIZE=224
ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"
~~~

### Monitor Training with TensorBoard

~~~bash
tensorboard --logdir tf_files/training_summaries &
~~~

> Note: to kill the tensorboard application, run the following command:

~~~bash
pkill -f "tensorboard"
~~~

## Perform Object Classification in Image

### Train the Final Layer of ImageNet Model

9:56PM - 9:59PM

~~~bash
python -m scripts.retrain \
	--bottleneck_dir=tf_files/bottlenecks \
	--how_many_training_steps=500 \
	--model_dir=tf_files/models/ \
	--summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
	--output_graph=tf_files/retrained_graph.pb \
	--output_labels=tf_files/retrained_labels.txt \
	--architecture="${ARCHITECTURE}" \
	--image_dir=tf_files/flower_photos
~~~

### Classify Flower in Image

Daisy?

~~~bash
python  -m scripts.label_image \
	--graph=tf_files/retrained_graph.pb \
	--image=tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg
~~~

Rose?

~~~bash
python -m scripts.label_image \
	--graph=tf_files/retrained_graph.pb \
	--image=tf_files/flower_photos/roses/2414954629_3708a1a04d.jpg
~~~

## Summary

Congratulations! You now know how to deploy tensorflow, use the deep learning framework to perform object classification in an image and present the results.

## Further Reading

- [How to Retrain an Image Classifier for New Categories](https://www.tensorflow.org/hub/tutorials/image_retraining)
- [TensorFlow For Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0)
