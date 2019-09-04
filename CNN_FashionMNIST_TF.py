# -*- coding: utf-8 -*-

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


############### in case u want to view the first image to get an idea of the subject to be classified
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


#### preprocessing immages to range from 0-1 instead of 0-255
train_images = train_images / 255.0
test_images = test_images / 255.0


####plotting first 25 immages of the training set to make sure the images and labels match
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#flatten layer just transforms the 28x28 matrix onto a array of 784 positions
#the next layer is one 128 nodes(or neurosn) fully connected layer
#and then another fully connected layer with 10 softmax nodes (each one indicating the probability of each class) 
model = keras.Sequential([    keras.layers.Flatten(input_shape=(28, 28)),
                              keras.layers.Dense(128, activation=tf.nn.relu),
                              keras.layers.Dense(10, activation=tf.nn.softmax)])

#configuring model compiler
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#feeding and training the model
model.fit(train_images, train_labels, epochs=10)


#evaluating model performance on the test set (in overfit cases it may get almost perfect for the train and then almost random selection for the test)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)



#check predictions for each individual immage of the test set
predictions = model.predict(test_images)
predictions[0]
np.argmax(predictions[0])


############################# PREDICTIONS VISUALIZATION FUNCTIONS ####################################
#PLOTS IMAGE WITH THE PREDICTED AND THE CORRECT CLASS
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)
  
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

#PLOTS BAR GRAPHS OF PREDICTIONS (BLUE THE CORRECT ONE AND RED THE PREDICTED ONE, ONLY BLUE IF CORRECT PREDICTION WAS MADE)
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#####PLOTTING BOTH, ONE BESIDES THE OTHER 
i = 42
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()


#### PLOTTING THE FIRST N TEST IMAGES WITH PREDICTIONS
num_rows = 10
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()


##############PREDICTION FOR A SINGLE IMAGE
img = test_images[0]
# Add the image to a batch where it's the only member.
#image need to be in shape such as img.shape = (numberOfImages, imgX, imgY)
#because tf.keras are made to predict batches or collections of exemples at once
img = (np.expand_dims(img,0))
prediction_single = model.predict(img)

plot_value_array(0, prediction_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()







































