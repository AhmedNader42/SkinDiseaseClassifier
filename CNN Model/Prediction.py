# Import the used libs
from keras.models import load_model
from keras.engine.sequential import Sequential
import pickle 
import cv2
import tensorflow as tf


class ModelBuilder():

	def __init__(self, modelUrl:str):
		super(ModelBuilder, self).__init__()
		# Loading the model
		self.model = load_model(modelUrl)
		global graph
		graph = tf.get_default_graph()
		

	def predict(self, imageUrl:str, width:int=200, height:int=200):
		# Load image at given url
		with graph.as_default():
				image = cv2.imread(imageUrl)
				output = image.copy()

				# Resize the image to fit model
				image = cv2.resize(image, (width, height))

				# Normalize the colors of the image
				image = image.astype("float") / 255.0

				# reshape the image
				image = image.reshape((1, width, height, 3))


		
				# Predict the given image
				preds = self.model.predict(image)
				#print(str(imageUrl + " IS "))
				#print(preds)
				# Load the prediction with the highest probability 
				i = preds.argmax(axis=1)[0]
		
				pred = ''
				precision = preds[0][i]
				if i:
					pred = 'Malignant'
				else:
					pred = 'Benign'
		return pred, precision*100

