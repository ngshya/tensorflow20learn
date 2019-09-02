#%% [markdown]
# # Get started with TensorFlow 2.0 for beginners
# In this example we train an image classifier. 


#%% [markdown]
# Load packages
#%%
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
print("Tensorflow version: " + tf.__version__)


#%% [markdown]
# Verify GPU 
#%%
device_lib.list_local_devices()
#%%
tf.test.is_gpu_available()
#%%
tf.test.gpu_device_name()


#%% [markdown]
# Load and prepare the MNIST dataset. Convert the samples from integers to floating-point numbers
#%%
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("Training set shape: " + str(x_train.shape))
print("Test set shape: " + str(x_test.shape))


#%% [markdown]
# Build the tf.keras.Sequential model by stacking layers. Choose an optimizer and loss function for training
#%%
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
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
# Train and evaluate the model
#%%
model.fit(x_train, y_train, epochs=5)
#%%
model.evaluate(x_test, y_test)


#%% [markdown]
# The image classifier is now trained to ~98% accuracy on this dataset.

