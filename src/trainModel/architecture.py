"""
__author__ : Kumar shubham, Srinivasa Raghavan K M
__date__   : 13-11-2019
__Desc__   : defining the architecture and related information for th code 
"""

import tensorflow as tf 
from model import Model
import sys
sys.path.insert(0,"../embPreparation")
from charVecToNp import timitWordEMbed,timitSpeechEmbed

class archForTrain(Model):
    ## defining the structure of the arhitecture for the model traijning 
    def __init__(self,wordLength=32,wordTimitData="../../data/TIMIT_words.txt",charEmbData="../../data/char-embeddings.txt",speechEmbData="../../data/output_auxiliary_z_vq_encoding.npz",speechLength=180,latentSize=64,speechVecDim=100,wordVecDim=300,batchSize = 32,EPOCH =100,LAMBDA=0.8,wbn=0,sbn=0):
        self.wordLen  = wordLength
        self.speechLen = speechLength
        self.latenSize=latentSize
        self.speechVecDim = speechVecDim
        self.wordVecDim = wordVecDim
        self.LAMBDA = LAMBDA
        self.EPOCH = EPOCH
        self.wbn = wbn
        self.sbn = sbn
        self.wordTimitData = wordTimitData
        self.charEmbData=charEmbData
        self.speechEmbData = speechEmbData

        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.speechOpt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.wordOpt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discOpt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        super(archForTrain, self).__init__()
        


    def textEncoder(self):
        ## defining the model for the encoder 
        inputVec = tf.keras.layers.Input(shape=[self.wordLen,self.wordVecDim])
        ## zero paadding 
        archStructure = [

                        [self.convStruct(100,2,1),self.convStruct(50,1,1)],
                        [self.convStruct(100,3,1),self.convStruct(50,1,1)],
                        [self.convStruct(100,4,1),self.convStruct(50,1,1)],
                        [self.convStruct(100,5,1),self.convStruct(50,1,1)], 
                    ]
        
        outputList = []

        for idx, listPerConv in enumerate(archStructure):
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

                        [self.convStruct(100,5,1),self.convStruct(50,1,1)],
                        [self.convStruct(100,25,1),self.convStruct(50,1,1)],
                        [self.convStruct(100,50,1),self.convStruct(50,1,1)],
                        [self.convStruct(100,75,1),self.convStruct(50,1,1)] 
                    ]
        
        outputList = []
        for idx, listPerConv in enumerate(archStructure):
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
        return(outputModel)
        #pass

    def speechDecoder(self):
        ## TODO : fill how to process the final genereated vector
        #inputVec = tf.keras.layers.Input(shape=[16,self.speechLen,self.speechVecDim])
        outputModel  = self.manyToManylstmDecoder(layerInfo=[200,100],timestamp=self.speechLen,distributedDense=self.speechVecDim)
        return(outputModel)
        #pass

    def klLoss(self):
        pass

    def reconLoss(self,actIn,actOut):
        ## function to calculate the reconstructon loss for the genere
        recLoss =  tf.reduce_sum(0.5*tf.pow(actIn-actOut,2))
        return recLoss


    def Run(self):
        print("Model Creation")
        speechEncoderModel = self.speechEncoder()
        discriminatorModel = self.discNetwork()
        wordEncoderModel  = self.textEncoder()

        speechDecoderModel = self.speechDecoder()
        wordDecoderModel = self.textDecoder()

        print("Data set loading")
        wordEmbedDataset,_ = timitWordEMbed(fileNameTimit=self.wordTimitData,fileNameEMbed=self.charEmbData)
        speechEmbedDataset,_ = timitSpeechEmbed(z_vqListFile=self.speechEmbData)
        
        print(wordEmbedDataset.shape)
        print(speechEmbedDataset.shape)
        print("Creating word pipeline")
        wordDataset =  tf.data.Dataset.from_tensor_slices(wordEmbedDataset)
        wordDataset =  wordDataset.cache().shuffle(256)
        #wordDataset = wordDataset.repeat().batch(self.wordBatch)
        wordDataset = wordDataset.repeat().batch(16)
        

        print("Creating speech pipeline")
        speechDataset =  tf.data.Dataset.from_tensor_slices(speechEmbedDataset)
        speechDataset =  speechDataset.cache().shuffle(256)
        #speechDataset = speechDataset.repeat().batch(self.speechBatch)
        speechDataset = speechDataset.repeat().batch(16)
        

        print("Creating train pipeline")
        def train(inputSpeechVec,inputTextVec):

            #with tf.GradientTape() as speechTape, tf.GradientTape() as wordTape:
            with tf.GradientTape(persistent=True) as speechTape, tf.GradientTape(persistent=True) as wordTape:
                speechLatentMean,speechLatentVar =speechEncoderModel(inputSpeechVec)
                epsSpeech = tf.random.normal(speechLatentMean.shape)
                zVecSpeech = speechLatentMean+tf.multiply(tf.sqrt(tf.exp(speechLatentVar)),epsSpeech)

                wordLatentMean,wordLatentVar = wordEncoderModel(inputTextVec)
                epsWord = tf.random.normal(wordLatentMean.shape)
                zVecWord = wordLatentMean+tf.multiply(tf.sqrt(tf.exp(wordLatentVar)),epsWord)
    
                discrimOutSpeech = discriminatorModel(zVecSpeech)
                discrimOutWord   = discriminatorModel(zVecWord)
        
                #print("No problem")
                print(tf.shape(zVecSpeech))
                decoderOutSpeech = speechDecoderModel(zVecSpeech)
                #print(decoderOutSpeech)
                decoderOutWord   = wordDecoderModel(zVecWord)

                ########################## Loss Functions ################################
                discLossOut = self.discLoss(discwordInp=discrimOutWord,discSpeechInp=discrimOutSpeech)
                genLossOut = self.genLoss(discSpeechInp=discrimOutSpeech)

                reconsLossSpeech =self.reconLoss(decoderOutSpeech,inputSpeechVec) 
                reconsLossWord = self.reconLoss(decoderOutWord,inputTextVec)

                ###########################################################################

                lossSpeech = genLossOut+self.LAMBDA*reconsLossSpeech
                lossWord = reconsLossWord
                lossDiscModel = discLossOut  

                decoderSpeechGrad = speechTape.gradient(lossSpeech,speechDecoderModel.trainable_variables)
                encoderSpeechGrad = speechTape.gradient(lossSpeech,speechEncoderModel.trainable_variables)
                discModelGrad     = speechTape.gradient(lossDiscModel,discriminatorModel.trainable_variables)

                decoderWordGrad = wordTape.gradient(lossWord,wordDecoderModel.trainable_variables)
                encoderWordGrad  = wordTape.gradient(lossWord,wordEncoderModel.trainable_variables)

                ###################### apply gradient ######################
                self.speechOpt.apply_gradients(zip(decoderSpeechGrad, speechDecoderModel.trainable_variables))
                self.speechOpt.apply_gradients(zip(encoderSpeechGrad, speechEncoderModel.trainable_variables))

                self.wordOpt.apply_gradients(zip(decoderWordGrad, wordDecoderModel.trainable_variables))
                self.wordOpt.apply_gradients(zip(encoderWordGrad, wordEncoderModel.trainable_variables))

                self.discOpt.apply_gradients(zip(discModelGrad,discriminatorModel.trainable_variables))
                return lossSpeech,lossWord,lossDiscModel

        #int wrd_btch_no = 0 
        #int spch.btch_no = 0
        for epNo in range(self.EPOCH):
            print("Epoch Number :",epNo)
            for wordBatch in wordDataset:
                print("inside wordBatch")
                self.wbn=self.wbn+1
                for speechBatch in speechDataset:
                    print("inside speechBatch")
                    self.sbn=self.sbn+1
                    lossSpeech,lossWord,lossDic = train(inputSpeechVec=speechBatch,inputTextVec=wordBatch)
                    print("EPISODE NO : {:4d}  Batch Speech NO : {:4d} batch Word No :{:2f}   LOSS  speech : {:2f} LOSS Word : {:2f} loss discriminator : {:2f}".format(epNo,self.sbn,self.wbn,lossSpeech,lossWord,lossDic))
                    #print("EPISODE NO : {:4d}  Batch Speech NO : {:4d} batch Word No :{:2f}   LOSS  speech : {:2f} LOSS Word : {:2f} loss discriminator : {:2f}".format())


if __name__=="__main__":
    obj = archForTrain()
    obj.Run()
