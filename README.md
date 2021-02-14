# Deep Learning Image Classification Insights with Transfer Learning in Keras

## Training

<img align="left" src="media/dl-cnn-keras-vis.png" width=300> 

ImageNet, an image recognition benchmark dataset, helped trigger the modern AI explosion.  In 2012, the AlexNet architecture (a deep convolutional-neural-network) rocked the ImageNet benchmark competition, handily beating the next best entrant.  By 2014, all the leading competitors were deep learning based.  Since then, accuracy scores continued to improve, eventually surpassing human performance.

In this hands-on tutorial we will build on this pioneering work to create our own neural-network architectures for image classification. 

Participants will use the elegant [Keras](https://keras.io/) deep learning programming interface to build and train [TensorFlow](https://www.tensorflow.org/) models for image classification tasks on the CIFAR-10 / MNIST datasets. We will demonstrate the use of transfer learning (to give our networks a head-start by building on top of existing, ImageNet pre-trained, network layers), and explore how to improve model performance for standard deep learning pipelines.

The tutorial, and complimentary hands-on exercise, aim beyond surface-level deep learning model creation and training mechanics. We explore the entire workflow, from data processing to learned features, using a variety of tools (e.g., [matplotlib](https://matplotlib.org/), [scikit-learn](https://scikit-learn.org/stable/)) and visualization techniques to aid with understanding the inner workings of neural networks.

We will use cloud-based interactive [Jupyter](https://jupyter.org/) notebooks to work through our explorations step-by-step.  Once participants have successfully trained their custom model we will introduce them to the wider data science community on Kaggle, and have them submit their model's predictions for scoring.

Join us to learn how to use deep learning and Keras to classify images.


## Venue, Date and Time 

  * **SciPy 2021** - **TBD**
  * Online - **TBD** 


## Details

This tutorial is designed as an introduction to the topic for a general, but technical audience. As a practical introduction, it will focus on tools and their application. Previous ML (Machine Learning) experience is not required; but, previous experience with scripting in Python will help.

In the tutorial, participants follow along with the worked examples. Each key step will be discussed or explained by the presenter. We will demonstrate a variety of image classification CNNs (convolutional neural networks) making use of transfer learning, skip connections, ensemble learning, sequential & functional APIs, and a variety of visualization techniques to create models and better understand them. The notebook is interactive throughout, with copious notes and references so support exploration and discovery by participants on their own at later dates.

The exercise follows the familiar layout of the tutorial, but provide participants an opportunity to recall and apply what they learned in the tutorial. Instructors are there to assist. Participants are also introduced to the larger data science community via Kaggle, where they partipate in an on-going competition, and have their efforts rewarded with placement on the competition leaderboard.


### Schedule

| Duration           | Topic                         |
| :----------------- | :---------------------------- |
| 2 hr 30 min        | Tutorial                      |
| 10 min             | _Break_                       |
| 1 hr 20 min        | Exercise & Kaggle Submission  |


### Tutorial Materials

* Notebooks:
  * [Tutorial Materials](https://github.com/kaust-vislab/keras-tutorials-lite/tree/scipy-2021)

* Datasets:
  * [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
  * [MNIST](http://yann.lecun.com/exdb/mnist/)


## Setup Instruction

Participants are expected to use their own laptops and sign-up for free online cloud services (e.g., Google Colab, Kaggle).  They may also need to download free, open-source software prior to arriving for the workshop.

Running the example notebooks requires compute resources with a GPU device to speed training.  These resource can be either in the cloud or on a local computer; however, for the purposes of training, we will use Google Colab exclusively.


### Google Colab

To run the notebooks in [Google Colab](https://colab.research.google.com) you will need a [Google Account](https://accounts.google.com/).  Sign-in to your Google account, if necessary, and then start the Tutorial / Exercise Notebook by clicking the buttons below:

* Tutorial: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaust-vislab/keras-tutorials-lite/blob/scipy-2021/notebooks/keras-transfer-learning-tutorial.ipynb)
* Exercise: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaust-vislab/keras-tutorials-lite/blob/scipy-2021/notebooks/keras-mnist-kaggle-exercise.ipynb)

Then follow the *Setup Colab* instructions in the notebook.


### Local

For technical users, we provide guidance for running the notebooks on their local machines; at a later date (we cannot provide technical support for this option during the training session).

Requires:

  * NVIDIA GPU – Performance required to follow tutorial in allotted time.
  * Conda [Miniconda Installers](https://docs.conda.io/en/latest/miniconda.html).

To work locally on your own laptop or workstation, use the [Conda](https://docs.conda.io/en/latest/miniconda.html) package managment system to create a work environment with the required software. After installing `miniconda` (above), follow these steps to setup the work environment and run the tutorial:

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


## Contributors

* Glendon Holst
  * Staff Scientist in the Visualization Core Lab at KAUST (King Abdullah University of Science and Technology) specializing in HPC workflow solutions for deep learning, image processing, and scientific visualization.
* David Pugh
  * Staff Scientist in the Visualization Core Lab at KAUST (King Abdullah University of Science and Technology) specializing in Data Science and Machine Learning.
  * Certified Software and Data Carpentry Instructor and Instructor Trainer. Lead instructor of the Introduction to Data Science Workshop series at KAUST.


## References

* https://qz.com/1034972/the-data-that-changed-the-direction-of-ai-research-and-possibly-the-world/
* https://www.cs.toronto.edu/~kriz/cifar.html
* http://yann.lecun.com/exdb/mnist/index.html
* 
  * https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751
  * https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e
  * https://machinelearningmastery.com/how-to-improve-performance-with-transfer-learning-for-deep-learning-neural-networks/
* https://towardsdatascience.com/deep-learning-at-scale-accurate-large-mini-batch-sgd-8207d54bfe02
* 
  * https://arxiv.org/abs/1409.1556
  * https://arxiv.org/abs/1610.02391
* https://www.kaggle.com/c/digit-recognizer
* https://jupyter-notebook.readthedocs.io/en/stable/
* https://github.com/kaust-vislab/handson-ml2
* 
  * https://keras.io/examples/cifar10_cnn/
  * https://keras.io/examples/cifar10_resnet/

