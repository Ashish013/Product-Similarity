import tensorflow as tf
import cv2
import gdown,os
import numpy as np
from zipfile import ZipFile
import streamlit as st

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

class Difference_Squared(tf.keras.layers.Layer):
	def __init__(self):
		super(Difference_Squared, self).__init__()

	def call(self, outputs):
		emb_A,emb_B = outputs
		return tf.pow(tf.subtract(emb_A,emb_B),2)


def contrastive_model(weights_path):

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

def cross_entropy_model(weights_path):
	# Initializes the embedding model and pre trained weights
	ptm = ResNet50(input_shape = (img_size,img_size,3), include_top=False)
	x = Flatten()(ptm.output)
	x = Dense(128)(x)
	emb_model = Model(ptm.input,x)

	img_A = Input(shape = (img_size,img_size,3))
	img_B = Input(shape = (img_size,img_size,3))
	emb_A = emb_model(img_A)
	emb_B = emb_model(img_B)
	distance = Difference_Squared()([emb_A,emb_B])
	distance = Dense(1,activation="sigmoid")(distance)
	model = Model(inputs = [img_A,img_B], outputs = distance)
	model.load_weights(weights_path)

	return model

def weights_download(path,weights_number,text):
	if os.path.exists(path) == False:
		text.write("### Downloading pre-trained weights....")
		if weights_number == 1:
			gdown.download("https://drive.google.com/uc?export=download&confirm=1TcX&id=1TdlcIwmbW4604XMw550jfmo5fAU8XoRH")
		elif weights_number == 2:
			gdown.download("https://drive.google.com/uc?export=download&confirm=WLfU&id=1xGFEkb5TbLFzJ4-Q30RgSeomUtVdzdkM")

		with ZipFile(path, 'r') as zip:
		  zip.extractall(path = './')

		text.write("### Predicting the similarity score....")


def calculate_similarity(img1,img2,weights_number,text):

	img1 = cv2.resize(np.asarray(img1,np.uint8),(256,256))[:,:,:3]
	img2 = cv2.resize(np.asarray(img2,np.uint8),(256,256))[:,:,:3]

	if weights_number == 1:
		weights_download("./pretrained-1.zip",1,text)
		model = contrastive_model("./pretrained-1/weights")
	elif weights_number == 2:
		weights_download("./pretrained-2.zip",2,text)
		model = cross_entropy_model("./pretrained-2/weights")

	prediction = model.predict([np.expand_dims(img1,axis = 0),np.expand_dims(img2,axis = 0)])
	return np.squeeze(prediction)

