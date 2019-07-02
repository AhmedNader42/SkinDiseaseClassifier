import tensorflow as tf
import keras as k
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import sklearn as sk
import random
import pickle
import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split
# Model imports
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence
from keras import optimizers
from keras import backend as K
from keras.applications import VGG16
from keras.utils import to_categorical
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
# Confirm tensorflow sees the GPU
from tensorflow.python.client import device_lib
assert 'GPU' in str(device_lib.list_local_devices())
print("TensorFlow using GPU")

# Confirm Keras sees the GPU
from keras import backend
assert len(backend.tensorflow_backend._get_available_gpus()) > 0
print("Keras using GPU")
from sklearn.utils import shuffle


# pre_trained_VGG = VGG16(weights='imagenet', include_top=False, input_shape=(200,200,3))

def LoadBatch(paths, inputWidth, inputHeight):
	images = []
	for img in paths:
		image = cv2.imread(img)
		if image is not None:
			image = cv2.resize(image, (inputWidth, inputHeight))
			images.append(image)
		else:
			img = img.replace(".jpeg", ".png")
			print("Couldn't load : " + img)
			im = cv2.imread(img)
			if im is not None:
				im = cv2.resize(im, (inputWidth, inputHeight))
				images.append(im)
	images = np.array(images, dtype="float16") / 255.0
	return images

def data_generator(image_paths, labels, batch_size, inputWidth, inputHeight):
    L = len(image_paths)
    labels = to_categorical(labels)


    # print(labels)
    # Needs to stay infinitely going for keras    
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)

            print("\n\t\t\tStarting from image : " + str(batch_start))
            print("\n\t\t\tEnding at image : " + str(limit)) 
            
            image_batch = LoadBatch(image_paths[batch_start:limit], inputWidth, inputHeight)
            label_batch = labels[batch_start:limit]

            # features_batch = pre_trained_VGG.predict(image_batch)
            # features_batch = np.reshape(features_batch, (limit, 7*7*512))
            
            print("\n\t\t\tfeatures Batch : " + str(len(image_batch)))
            print("\n\t\t\tLabel Batch : " + str(len(label_batch)))
            yield (image_batch, label_batch) #a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size

    
		

list_of_images = [] 
# absolutePathImages = "C:\\My Files\\GP\\Graduation Project\\Dataset\\ISIC-Archive-Downloader\\Data\\Images"
inputWidth = 200
inputHeight = 200
dataSize = 23906 # -8707 - 542 = 9249



absolutePathLabels = "labels.csv"
dataCSV = pd.read_csv(absolutePathLabels)

data = dataCSV['classification']

data_benign     = dataCSV[dataCSV['classification'] < 1]
data_malignant  = dataCSV[dataCSV['classification'] > 0]


data_benign = data_benign.sample(n=len(data_malignant))
print("Benign")
print(len(data_benign))
print("Malig")
print(len(data_malignant))

labels_frames = [data_benign['classification'], data_malignant['classification']] 
image_frames  = [data_benign['name'], data_malignant['name']]

labels      = pd.concat(labels_frames)
image_names = pd.concat(image_frames)

labels, image_names = shuffle(labels, image_names, random_state=0)

print(image_names.head())
print(labels.head())

print("labels : " + str(len(labels)))
print("image_names : " + str(len(image_names)))

# Add the names of the images
for image_name in image_names:
    list_of_images.append(str(image_name) + ".jpeg")



# Split the data into training and validation.
(trainX, validationX, trainY, validationY) = train_test_split(list_of_images, labels, test_size=0.2, random_state=42)





def BuildModel(width, height, depth, classes):
    model = Sequential()
    dataFormat = K.image_data_format()
    print(dataFormat)
    if dataFormat == "channels_first":
        # Theano
        inputShape = (depth, height, width)
        chanDim = 1
    else:
        # Tensorflow
        inputShape = (height, width, depth)
        chanDim = -1
        
    # Layer 1
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format=dataFormat))
    model.add(Dropout(0.25))
        
    # Layer 2
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format=dataFormat))
    model.add(Dropout(0.25))
        
    # Layer 3
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format=dataFormat))
    model.add(Dropout(0.25))


    # Layer 4
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
 
    # softmax classifier
    print(str(classes) + " Classes")
    model.add(Dense(classes))
    model.add(Activation("softmax"))
 
    # return the constructed network architecture
    return model

model = BuildModel(width=inputWidth, height=inputHeight, depth=3, classes=2)




learning_rate = 0.01
epochs = 25
batch_size = 32
train_steps = 32 #np.ceil(len(trainX) / batch_size)  
test_steps  = np.ceil(len(validationY) / batch_size)

adam = optimizers.Adam(lr= learning_rate)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
# model.compile(loss=[focal_loss], optimizer=adam, metrics=['accuracy'])



weights = {
	0: 1.,
	1: 2.23
}
print("Fitting network for : "  + str(train_steps) + " Steps")
H = model.fit_generator( data_generator(trainX, trainY, batch_size, inputWidth, inputHeight),
	                 steps_per_epoch=train_steps,
	                 epochs=epochs)
	                 # class_weight=weights) 

print("fit model")

model.save("output/trainedModel.model")

print(H.history)
# plot the training loss and accuracy
N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("output/plt.png")



print("Testing Network for : " + str(test_steps) + " Steps" ) 
predictions = model.predict_generator( data_generator(validationX, validationY, batch_size, inputWidth, inputHeight),
	                                   steps=test_steps)

preds = []
for each in predictions:
	# print(each)
	if each[0] > each[1]:
		preds.append(0)
	else :
		preds.append(1)
# print(len(predictions))
# print(len(validationY))
# print(len(predictions.shape))
# print(len(validationY.shape))
# print(predictions)
# print(preds)
# print(validationY)
print(classification_report(validationY, preds, target_names=["Benign", "Malignant"]))





