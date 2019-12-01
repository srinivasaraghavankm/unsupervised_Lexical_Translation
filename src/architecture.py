"""
__author__ : Kumar shubham 
__date__   : 13-11-2019
__Desc__   : defining the architecture and related information for th code 
"""

import tf 
from model import Model


class archForTrain(Model):
	## defining the structure of the arhitecture for the model traijning 
	def __init__(self,wordLength,speechLenth,latentSize,speechVecDim,wordVecDim):
		self.wordLen  = wordLength
		self.speechLen = speechLenth
		self.latenSize=latenSize
		self.speechVecDim = speechVecDim
		self.wordVecDim = wordVecDim

		super(archForTrain, self).__init__()
		
	def textEncoder(self):
		## defining the model for the encoder 
		inputVec = tf.keras.layers.Input(shape=[self.wordLen,self.wordVecDim])
		## zero paadding 
		archStructure = [

						[self.convStruct(100,2),self.convStruct(50,1)],
						[self.convStruct(100,3),self.convStruct(50,1)],
						[self.convStruct(100,4),self.convStruct(50,1)],
						[self.convStruct(100,5),self.convStruct(50,1)], 
					]
		
		outputList = []

		for idx, listPerConv in ennumerate(archStructure):
			x = inputVec
			for elm in  listPerConv:
				x = elm(x)
			outputList.append(x)


		## concatenate the value in the list 
		finalOut = tf.keras.layers.Concatenate()(outputList)
		outputMean = tf.keras.layers.Dense(self.latenSize)(finalOut)
		outputVarLog = tf.keras.layers.Dense(self.latenSize)(finalOut)
		return tf.keras.Model(inputs=inputVec,outputs=[outputMean,outputVarLog])


	def speechEncoder(self):
		## defining the encoder for the speech 
		
		## defining the model for the encoder 
		inputVec = tf.keras.layers.Input(shape=[self.speechLen,self.speechVecDim])
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
			for elm in  listPerConv:
				x = elm(x)
			outputList.append(x)


		## concatenate the value in the list 
		finalOut = tf.keras.layers.Concatenate()(outputList)
		outputMean = tf.keras.layers.Dense(self.latenSize)(finalOut)
		outputVarLog = tf.keras.layers.Dense(self.latenSize)(finalOut)
		return tf.keras.Model(inputs=inputVec,outputs=[outputMean,outputVarLog])

		

	def discLoss(self,discwordInp,discSpeechInp):
		## defining advesarial losss
		realLoss = self.loss(tf.ones_like(discwordInp),discwordInp)

		## wrong output loss 
		advLoss = self.loss(tf.zeros_like(discSpeechInp),discSpeechInp)
		
		## defining the total loss 
		totalLoss = realLoss+advLoss
		return totalLoss

	def genLoss(self,discSpeechInp):
		## defining the generator Loss 
		genLoss = self.loss(tf.ones_like(discSpeechInp),discSpeechInp)
		return genLoss


	def discNetwork(self):
		## discriminator network
		inputVec = tf.keras.Input(shape = [self.latenSize])
		outDense =  tf.keras.layers.Dense(50)(inputVec) 
		finalOut = tf.keras.layers.Dense(100)(outDense)
		return tf.keras.Model(inputs=inputVec,outputs=finalOut)

	def textDecoder(self):
		## TODO : fill how to process the final genereated vector
		outputModel  = self.manyToManylstmDecoder(layerInfo=[200,100],timestamp=self.wordLen,distributedDense=self.wordVecDim)
		pass

	def speechDecoder(self):
		## TODO : fill how to process the final genereated vector
		outputModel  = self.manyToManylstmDecoder(layerInfo=[200,100],timestamp=self.speechLen,distributedDense=self.speechVecDim)
		pass
		

	def klLoss(self):
		pass

	def reconLoss(self,actIn,actOut):
		## function to calculate the reconstructon loss for the genere
		recLoss =  tf.reduce_sum(0.5*tf.pow(actIn-actOut,2))
		return recLoss


	def Run():
		speechEncoderModel = self.speechEncoder()
		discriminatorModel = self.discNetwork()
		wordEncoderModel  = self.textEncoder()

		speechDecoderModel = self.speechDecoder()
		wordDecoderModel = self.textDecoder()


	def train(self,inputSpeechVec,inputTextVec):

			with tf.gradientTape() as speechTape, tf.gradientTape() as wordTape:
				speechLatentMean,speechLatentVar =speechEncoderModel(inputSpeechVec)
				epsSpeech = tf.random_normal(self.latenSize)
				zVecSpeech = speechLatentMean+tf.multiply(tf.sqrt(tf.exp(speechLatentVar)),epsSpeech)

				wordLatentMean,wordLatentVar = wordEncoderModel(inputTextVec)
				epsWord = tf.random_normal(self.latenSize)
				zVecWord = wordLatentMean+tf.multiply(tf.sqrt(tf.exp(wordLatentVar)),epsWord)
	
				discrimOutSpeech = discriminatorModel(zVecSpeech)
				discrimOutWord   = discriminatorModel(zVecWord)

				decoderOutSpeech = speechDecoderModel(zVecSpeech)
				decoderOutWord   = wordDecoderModel(zVecWord)

				########################## Loss Functions ################################
				discLossOut = self.discLoss(discwordInp=discrimOutWord,discSpeechInp=discrimOutSpeech)
				genLossOut = self.genLoss(discSpeechInp=discrimOutSpeech)

				reconsLossSpeech =self.reconLoss(decoderOutSpeech,inputSpeechVec) 
				reconsLossWord = self.reconLoss(decoderOutWord,inputTextVec)

				###########################################################################

				lossSpeech = genLossOut+reconsLossSpeech
				lossWord = reconsLossWord
				lossDiscModel = discLossOut  

				decoderSpeechGrad = 

