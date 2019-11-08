"""
__author__ : Kumar shubham 
__date__   :  07-11-2019
__desc__   : code for unsupervised mapping of the data
"""
import tf 

class modelCNN(object):

	def __init__(self,vocabSize,embedShape,):
		# definig the structure of the CNN model 
		"""
		param :
		 vocabSize   : size of the vocabulary in the dictionary
		 embedShapee : shape of the embed layer  
		"""

		self.vocabSz = vocabSize
		self.embedShape = embedShape

	def cnnStruct(self,filters,size,applyBatchnorm=True,applyDropOut=True):
		## function for downsampling the image 
		initializer = tf.random_normal_initializer(0.,0.02)
		output = tf.keras.Sequential()

		output.add(
		  tf.keras.layers.Conv1D(filters, size, strides=2, padding='same',
							 kernel_initializer=initializer))

		if applyBatchnorm:
			output.add(tf.keras.layers.BatchNormalization())

		if applyDropOut:
			output.add(tf.keras.layers.Dropout(0.5))

		output.add(tf.keras.layers.LeakyReLU())

		return output
