# Deep Learning from Pre-Trained Models with Keras

## Abstract

ImageNet, an image recognition benchmark dataset*, helped trigger the modern AI explosion.  In 2012, the AlexNet architecture (a deep convolutional-neural-network) rocked the ImageNet benchmark competition, handily beating the next best entrant.  By 2014, all the leading competitors were deep learning based.  Since then, accuracy scores continued to improve, eventually surpassing human performance.

In this hands-on tutorial we will build on this pioneering work to create our own neural-network architecture for image recognition.  Participants will use the elegant Keras deep learning programming interface to build and train TensorFlow models for image classification tasks on the CIFAR-10 / MNIST datasets*.  We will demonstrate the use of transfer learning* (to give our networks a head-start by building on top of existing, ImageNet pre-trained, network layers*), and explore how to improve model performance for standard deep learning pipelines.  We will use cloud-based interactive Jupyter notebooks to work through our explorations step-by-step.  Once participants have successfully trained their custom model we will show them how to submit their model's predictions to Kaggle for scoring*.

This tutorial is designed as an introduction to the topic for a general, but technical audience. As a practical introduction, it will focus on tools and their application. Previous ML (Machine Learning) experience is not required; but, previous experience with scripting in Python will help. 

Participants are expected to provide their own laptops and sign-up for free online cloud services (e.g., Google Colab, Kaggle).  They may also need to download free, open-source software prior to arriving for the workshop.


## Runtime Environment

Running the example notebooks requires compute resources; preferably, including a GPU device to speed training.  These resource can be either in the cloud, or on a local computer.


### Launch Cloud

Launch this tutorial in the cloud using [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb) by clicking the buttons below:

* Tutorial: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaust-vislab/keras-tutorials-lite/blob/master/notebooks/keras-transfer-learning-tutorial.ipynb)
* Exercise: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaust-vislab/keras-tutorials-lite/blob/master/notebooks/keras-mnist-kaggle-exercise.ipynb)


### Launch Locally

To work locally on your own laptop or workstation, use the [Conda](https://docs.conda.io/en/latest/miniconda.html) package managment system to create a work environment with the required software. After installing miniconda (above), follow these steps to setup the work environment and run the tutorial:

Create the environment...

```bash
conda env create --prefix ./env --file environment-gpu.yml
```

...then activate the environment...

```bash
conda activate ./env
```

...then launch the Jupyter Notebook server.

```bash
jupyter notebook notebooks/keras-transfer-learning-tutorial.ipynb
```


