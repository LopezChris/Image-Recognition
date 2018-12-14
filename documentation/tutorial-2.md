---
title: Performing Image Recognition
---

# Performing Image Recognition

## Introduction

You will explore setting up a development environment with TensorFlow, use the Deep Learning Framework to perform object classification on an image of different elephant specifies and present the results.

## Prerequisites

- Basic understanding of Linux commands
- Installed Docker on your local host machine
- [Installed git on Ubuntu Docker](https://stackoverflow.com/questions/29929534/docker-error-unable-to-locate-package-git)
- Installed wget on Ubuntu Docker

## Outline

- Setup Development Environment
- Perform Object Classification in Image
- Summary
- Further Reading
- Appendix: Classify Flower in Image

## Setup Development Environment

### Option 1: Install TensorFlow Docker Container

Open your terminal, linux shell or windows ubuntu shell. Run the command to download the docker **tensorflow** image:

~~~bash
cd ~/Downloads
git clone https://github.com/james94/tensorflow.git
~~~

~~~bash
cd ~/Downloads/tensorflow/tensorflow/tools/dockerfiles
~~~

~~~bash
docker build -f dockerfiles/cpu.Dockerfile -t tensorflow .
~~~

### Deploy Docker Container Mounted to Elephant Images on Host Machine

Run the command to deploy the docker image as a container:

~~~bash
docker run -p 8888:8888 -p 6006:6006 -it -v ~/Desktop/elephant_photos:/tmp/tf_files/elephant_photos tensorflow bash
~~~

> Note: the volume mount to your elephant photos may be at a different location than
`~/Desktop/elephant_photos:`, so make sure to specify the path to your images.
Docker will create the directory path `/tmp/tf_files/elephant_photos` to connect
your photos between `host-dir:docker-container-dir`.

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

Train the final layer of ImageNet model using **elephant_photos**.

~~~bash
python -m scripts.retrain \
	--bottleneck_dir=tf_files/bottlenecks \
	--how_many_training_steps=500 \
	--model_dir=tf_files/models/ \
	--summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
	--output_graph=tf_files/retrained_graph.pb \
	--output_labels=tf_files/retrained_labels.txt \
	--architecture="${ARCHITECTURE}" \
	--image_dir=/tmp/tf_files/elephant_photos
~~~

### Classify Flower in Image

### Option 1: Classify Borneo Elephant in Image

~~~bash
python -m scripts.label_image \
	--graph=tf_files/retrained_graph.pb \
	--image=/tmp/tf_files/elephant_photos/borneo_elephant/yjLRf5A.jpg
~~~

Output example:

~~~bash
2018-12-14 18:13:14.734788: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2018-12-14 18:13:14.808851: W tensorflow/core/framework/allocator.cc:122] Allocation of 91835964 exceeds 10% of system memory.

Evaluation time (1-image): 0.124s

borneo elephant (score=0.69002)
african bush elephant (score=0.30986)
african forest elephant (score=0.00010)
sumatran elephant (score=0.00001)
indian elephant (score=0.00000)
~~~

### Option 2: Classify Sumatran Elephant in Image

~~~bash
python -m scripts.label_image \
	--graph=tf_files/retrained_graph.pb \
	--image=/tmp/tf_files/elephant_photos/sumatran_elephant/yongki2_3448838b.jpg
~~~

Output example:

~~~bash
2018-12-14 18:14:46.135042: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA

Evaluation time (1-image): 0.137s

sumatran elephant (score=0.87228)
borneo elephant (score=0.12618)
african forest elephant (score=0.00154)
african bush elephant (score=0.00000)
indian elephant (score=0.00000)
~~~

## Summary

Congratulations! You now know how to deploy tensorflow, use the deep learning framework to perform object classification in an image and present the results.

## Further Reading

- [How to Retrain an Image Classifier for New Categories](https://www.tensorflow.org/hub/tutorials/image_retraining)
- [TensorFlow For Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0)

## Appendix: Classify Flower in Image

### Train Final Layer with Flower Photos

Train the final layer of ImageNet model using flower_photos.

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

### Classify Daisy in Image

Daisy?

~~~bash
python  -m scripts.label_image \
	--graph=tf_files/retrained_graph.pb \
	--image=tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg
~~~

### Classify Rose in Image

Rose?

~~~bash
python -m scripts.label_image \
	--graph=tf_files/retrained_graph.pb \
	--image=tf_files/flower_photos/roses/2414954629_3708a1a04d.jpg
~~~
