"""
__author__ : Kumar shubham 
__date__   :  07-11-2019
__desc__   : code for unsupervised mapping of the data
"""
## TODO : Define loss function and other details for the reconstruction loss 
## TODO : Define model structure 
## TODO : Define train and gradient tape   

import tf 

class Model(object):
	def __init__(self,vocabSize,embedShape):
		# definig the structure of the CNN model 
		"""
		param :
		 vocabSize   : size of the vocabulary in the dictionary
		 embedShapee : shape of the embed layer  
		"""

		self.vocabSz = vocabSize
		self.embedShape = embedShape

	def convStruct(self,outChannel,kerSize,stride,applyBatchNorm=True,applyDropOut=True):
		## function for downsampling the image 
		initializer = tf.random_normal_initializer(0.,0.02)
		output = tf.keras.Sequential()

		output.add(
		  tf.keras.layers.Conv1D(outChannel, kerSize, strides=stride, padding='same',
							 kernel_initializer=initializer))

		if applyBatchNorm:
			output.add(tf.keras.layers.BatchNormalization())

		if applyDropOut:
			output.add(tf.keras.layers.Dropout(0.5))

		output.add(tf.keras.layers.LeakyReLU())

		return output


	def deconvStruct(self,outChannel,kerSize,stride,applyBatchNorm=True,applyDropOut=True):
		# function to apply deconvolution over image
		initializer = tf.random_normal_initializer(0.,0.02)
		output = tf.keras.Sequential()

		output.add(
		  tf.keras.layers.Conv1DTranspose(outChannel, size, strides=strides, padding='same',
							 kernel_initializer=initializer, use_bias=False))

		if applyBatchNorm:
			output.add(tf.keras.layers.BatchNormalization())
		if applyDropOut:
			output.add(tf.keras.layers.Dropout(0.5))

		output.add(tf.keras.layers.ReLU())
		return output

	def lstmEncoder(self,layerInfo=[],timestamp=10,inputShape = ()):
		# definig the lstm Structure for the model. 
		output = tf.keras.Sequential()
		for idx,hdLayr in ennumerate(layerInfo):
			if idx == 0:
				output.add(
					tf.keras.layers.LSTM(hdLayr, activation='relu', input_shape=inputShape, return_sequences=True)
					)
			elif idx!= len(layerInfo)-1:
				output.add(
					tf.keras.layers.LSTM(hdLayr, activation='relu',  return_sequences=True)
					)
			else:
				output.add(
					tf.keras.layers..LSTM(hdLayr, activation='relu',  return_sequences=False)
					)

		return output
	def manyToManylstmDecoder(self,layerInfo=[],timestamp=10,distributedDense,inputShape=()):
		## defining the lstm decoder for the same network
		output = tf.keras.Sequential()
		output.add(tf.keras.layers.RepeatVector(timestamp))
		for idx,hdLayer in layerInfo:
			output.add(tf.keras.layers.LSTM(hdLayr, activation='relu', return_sequences=True))
		output.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(distributedDense)))
		return output
