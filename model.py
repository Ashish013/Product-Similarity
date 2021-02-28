import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.layers import Dense,Input,Flatten
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50

img_size = 256

class Euclidean_Distance(tf.keras.layers.Layer):
	def __init__(self):
		super(Euclidean_Distance, self).__init__()

	def call(self, outputs):
		emb_A,emb_B = outputs
		return tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(emb_A,emb_B),2),axis = 1))

def instantiate_model(weights_path):

	# Initializes the embedding model and pre trained weights
	ptm = ResNet50(input_shape = (img_size,img_size,3), include_top=False)
	x = Flatten()(ptm.output)
	x = Dense(128)(x)
	emb_model = Model(ptm.input,x)

	img_A = Input(shape = (img_size,img_size,3))
	img_B = Input(shape = (img_size,img_size,3))
	emb_A = emb_model(img_A)
	emb_B = emb_model(img_B)
	distance = Euclidean_Distance()([emb_A,emb_B])
	model = Model(inputs = [img_A,img_B], outputs = distance)
	model.load_weights(weights_path)

	return model


def calculate_similarity(img1,img2,t):

	img1 = cv2.resize(np.asarray(img1,np.uint8),(256,256))
	img2 = cv2.resize(np.asarray(img2,np.uint8),(256,256))
	model = instantiate_model("./pretrained/weights")
	#model = tf.keras.models.load_model('./trainedModel')


	distance = model.predict([np.expand_dims(img1,axis = 0),np.expand_dims(img2,axis = 0)])
	t = 16
	if distance > t:
		return "similar",np.squeeze(distance)
	else:
		return "dis-similar",np.squeeze(distance)