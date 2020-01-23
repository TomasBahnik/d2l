import tensorflow as tf
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())  # list of DeviceAttributes

#  tensorflow >= 1.4
print(" gpu available : {} ".format(tf.test.is_gpu_available()))  # True/False

# Or only check for gpu's with cuda support
print(" gpu cuda available : {} ".format(tf.test.is_gpu_available(cuda_only=True)))  # True/False

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))