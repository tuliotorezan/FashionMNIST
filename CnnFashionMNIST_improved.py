# -*- code by: Tulio Torezan -*-

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__) #just checking (working on 1.14)

#importing dataset from tensorflow keras datasets
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#keeping track of the classes to translate later
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


###just exemples for ilustrating how to use in case i didnt already know the size and number of images ###############
#checking dataset specifications
train_images.shape

#confirming the amount of labels is the same as the amount of images
len(train_labels)
train_labels
test_images.shape
len(test_labels)


#### preprocessing immages to range from 0-1 instead of 0-255
train_images = train_images / 255.0
test_images = test_images / 255.0


train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)


model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.3))

#Note that I am using 32 filtersn in the first convolutional layer and then 64 on the second one
#I noticed that using 64 1st and 32 2nd gets worst results
#while using 64 1st and 128 2nd gets similar results but takes a bit longer 

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001),
                             activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001),
                             activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10,
                             activation=tf.nn.softmax))



#configuring model compiler
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', 'sparse_categorical_crossentropy'])


#feeding and training the model
model.fit(train_images,
          train_labels,
          batch_size=128,
          epochs=25,
          validation_data=(test_images, test_labels),
          verbose=1)

#evaluating model performance on the test set (in overfit cases it may get almost perfect for the train and then almost random selection for the test)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)


