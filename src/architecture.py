"""
__author__ : Kumar shubham 
__date__   : 13-11-2019
__Desc__   : defining the architecture and related information for th code 
"""

import tf 
from model import Model


class archForTrain(Model):
	## defining the structure of the arhitecture for the model traijning 
	def __init__(self,wordLength,speechLenth):
		self.wordLen  = wordLength
		self.speechLen = speechLenth
		super(archForTrain, self).__init__()
		
	def textEncoder(self):
		## defining the model for the encoder 
		inputVec = tf.keras.layers.Input(shape=[None])
		archStructure = [

						[self.convStruct(100,2),self.convStruct(50,1)],
						[self.convStruct(100,3),self.convStruct(50,1)],
						[self.convStruct(100,4),self.convStruct(50,1)],
						[self.convStruct(100,4),self.convStruct(50,1)], 
					]
		
		outputList = []
		for idx, listPerConv in ennumerate(archStructure):
			x = inputVec
			for elm in  archStructure:
				x = elm(x)
			outputList.append(x)


		## concatenate the value in the list 
		finalOut = tf.keras.layers.Concatenate()(outputList)
		outputMean = tf.keras.layers.Dense(100)(finalOut)
		outputVarLog = tf.keras.layers.Dense(100)(finalOut)
		return tf.keras.Model(inputs=inputVec,outputs=[outputMean,outputVarLog])


	def speechEncoder(self):
		## defining the encoder for the speech 
		
		## defining the model for the encoder 
		inputVec = tf.keras.layers.Input(shape=[None])
		archStructure = [

						[self.convStruct(100,10),self.convStruct(50,1)],
						[self.convStruct(100,50),self.convStruct(50,1)],
						[self.convStruct(100,100),self.convStruct(50,1)],
						[self.convStruct(100,200),self.convStruct(50,1)],
						[self.convStruct(100,400),self.convStruct(50,1)] 
					]
		
		outputList = []
		for idx, listPerConv in ennumerate(archStructure):
			x = inputVec
			for elm in  archStructure:
				x = elm(x)
			outputList.append(x)


		## concatenate the value in the list 
		finalOut = tf.keras.layers.Concatenate()(outputList)
		outputMean = tf.keras.layers.Dense(100)(finalOut)
		outputVarLog = tf.keras.layers.Dense(100)(finalOut)
		return tf.keras.Model(inputs=inputVec,outputs=[outputMean,outputVarLog])

		

	def discLoss(self,disOutput,realOutput):
		## defining advesarial losss
		realLoss = self.loss(tf.ones_like(realOutput),realOutput)

		## wrong output loss 
		advLoss = self.loss(tf.zeros_like(disOutput),disOutput)
		
		## defining the total loss 
		totalLoss = realLoss+advLoss
		return totalLoss

	def genLoss(self,discOutput):
		## defining the generator Loss 
		genLoss = self.loss(tf.ones_like(discOutput),disOutput)
		return genLoss


	def discNetwork(self,):
		pass

	def textDecoder(self):
		## TODO : fill how to process the final genereated vector
		outputModel  = self.manyToManylstmDecoder(layerInfo=[200,100],timestamp=self.wordLen)
		pass

	def speechDecoder(self):
		## TODO : fill how to process the final genereated vector
		outputModel  = self.manyToManylstmDecoder(layerInfo=[200,100],timestamp=self.speechLen)
		pass
		

	def klLoss(self):
		pass

	def reconLoss(self):
		pass

	def train(self):
		pass