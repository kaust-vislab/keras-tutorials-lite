import pathlib
import tensorflow.keras.utils as Kutils

def cache_mnist_data():
  for n in ["mnist.npz", "kaggle/train.csv", "kaggle/test.csv"]:
    path = pathlib.Path("../datasets/mnist/%s" % n).absolute()
    if not path.is_file():
      print("Skipping: missing local dataset file: %s" % n)
    else:
      DATA_URL = "file:///" + str(path)
      try:
        data_file_path = Kutils.get_file(n.replace('/','-mnist-'), DATA_URL)
        print("Cached file: %s" % n)
      except (FileNotFoundError, ValueError, Exception) as e:
        print("Cache Failed: First fetch file: %s" % n)

def cache_cifar10_data():
  for n in ["cifar-10.npz", "cifar-10-batches-py.tar.gz"]:
      path = pathlib.Path("../datasets/cifar10/%s" % n).absolute()
      if not path.is_file():
        print("Skipping: missing local dataset file: %s" % n)
      else:
        DATA_URL = "file:///" + str(path)
        try:
          data_file_path = Kutils.get_file(n, DATA_URL)
          print("Cached file: %s" % n)
        except (FileNotFoundError, ValueError, Exception) as e:
          print("Cache Failed: First fetch file: %s" % n)

def cache_models():
  for n in ["vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"]:
    path = pathlib.Path("../models/%s" % n).absolute()
    if not path.is_file():
      print("Skipping: missing local dataset file: %s" % n)
    else:
      DATA_URL = "file:///" + str(path)
      try: 
        data_file_path = Kutils.get_file(n, DATA_URL, cache_subdir='models')
        print("Cached file: %s" % n)
      except (FileNotFoundError, ValueError, Exception) as e:
        print("Cache Failed: First fetch file: %s" % n)
