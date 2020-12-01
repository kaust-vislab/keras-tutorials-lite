# Verify runtime environment

try:
    # %tensorflow_version only exists in Colab.
    %tensorflow_version 2.x
    IS_COLAB = True
except Exception:
    IS_COLAB = False
print("is_colab:", IS_COLAB)

assert tf.__version__ >= "2.0", "TensorFlow version >= 2.0 required."
print("tensorflow_version:", tf.__version__)

assert sys.version_info >= (3, 5), "Python >= 3.5 required."
print("python_version:", "%s.%s.%s-%s" % (sys.version_info.major, 
                                          sys.version_info.minor,
                                          sys.version_info.micro,
                                          sys.version_info.releaselevel
                                         ))

print("executing_eagerly:", tf.executing_eagerly())

try:
    __physical_devices = tf.config.list_physical_devices('GPU')
except AttributeError:
    __physical_devices = tf.config.experimental.list_physical_devices('GPU')
    
if len(__physical_devices) == 0:
    print("No GPUs available. Expect training to be very slow.")
    if IS_COLAB:
        print("Go to `Runtime` > `Change runtime` and select a GPU hardware accelerator."
              "Then `Save` to restart session.")
else:
    print("is_built_with_cuda:", tf.test.is_built_with_cuda())
    print("gpus_available:", [d.name for d in __physical_devices])
