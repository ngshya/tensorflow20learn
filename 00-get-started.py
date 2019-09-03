#%% [markdown]
# # Get started with TensorFlow 2.0
# In this example we train an image classifier. 


#%% [markdown]
# **Load packages**
#%%
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.python.client import device_lib
import datetime
#%%
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


#%% [markdown]
# **Print versions**
#%%
!cat /etc/*release
#%%
!nvidia-smi --query-gpu=driver_version --format=csv,noheader
#%%
!cat /usr/local/cuda*/version.txt
#%%
!cat /usr/include/x86_64-linux-gnu/cudnn_v*.h | grep CUDNN_MAJOR -A 2
#%%
!dpkg -l | grep TensorRT
#%%
print(tf.__version__)



#%% [markdown]
# **Check GPU**
#%%
device_lib.list_local_devices()
#%%
tf.test.is_gpu_available()
#%%
tf.test.gpu_device_name()


#%% [markdown]
# **Load and prepare the MNIST dataset. Convert the samples from integers to floating-point numbers**
#%%
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))
x_train, x_test = x_train / 255.0, x_test / 255.0
print("Training set shape: " + str(x_train.shape))
print("Test set shape: " + str(x_test.shape))


#%% [markdown]
# **Build the tf.keras.Sequential model by stacking layers. Choose an optimizer and loss function for training**
#%%
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), 
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'), 
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(10, activation='softmax')
])
#%%
model.summary()
#%%
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#%% [markdown]
# **Train and evaluate the model**
#%%
tms_start = datetime.datetime.now()
model.fit(x_train, y_train, epochs=30, batch_size=64)
tms_end = datetime.datetime.now()
tms_delta = tms_end - tms_start
print("Elapsed: " + str(tms_delta))
print("Elapsed in seconds: " + str(int(tms_delta.total_seconds())))
#%%
model.evaluate(x_test, y_test, verbose=0)


#%%
