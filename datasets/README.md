# Datasets

## Provenance

* `cifar-10-python.tar.gz` and `cifar-100-python.tar.gz` are from: 
  * https://www.cs.toronto.edu/~kriz/cifar.html
  * `cifar-10-batches-py.tar.gz` comes from: 
    * https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    * from tensorflow.keras.datasets import cifar10
      * cifar10.load_data();
      * `load_data` also extracts archive into directory: `cifar-10-batches-py`
    * Idential to `cifar-10-python.tar.gz`
  * `cifar-100-python.tar.gz` comes from:
    * https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    * from tensorflow.keras.datasets import cifar100
      * cifar100.load_data();
      * `load_data` also extracts archive into directory: `cifar-100-python`

* `cifar-10.npz` comes from Kaggle: 
  * https://www.kaggle.com/guesejustin/cifar10-keras-files-cifar10load-data
  * `cifar10-keras-files-cifar10load-data.zip` -> `cifar-10.npz`

* `CIFAR-10-C.tar.*` come from:
  * https://zenodo.org/record/2535967
* `CIFAR-100-C.tar.*` come from:
  * https://zenodo.org/record/3555552

* `mnist.npz` comes from: 
  * https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
  * from tensorflow.keras.datasets import mnist
    * mnist.load_data()

* `mnist/kaggle/`{`train.csv`, `test.csv`, `sample_submission.csv`} come from Kaggle: 
  * https://www.kaggle.com/c/digit-recognizer
  * `digit-recognizer.zip` -> `train.csv`, `test.csv`, `sample_submission.csv`

