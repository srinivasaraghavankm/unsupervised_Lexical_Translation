"""
__author__ : Srinivasa Raghavan K M, Kumar shubham
__date__   : 13-11-2019
__Desc__   : defining the architecture and related information for th code 
"""

import tensorflow as tf 
from model import Model
import sys
import os
sys.path.insert(0,"../embPreparation")
from charVecToNp import timitWordEMbed,timitSpeechEmbed

class archForTrain(Model):
  ## defining the structure of the arhitecture for the model traijning 
  ## stepToCkpt was 100 
  def __init__(self,wordLength=32,wordTimitData="../../data/TIMIT_words.txt",charEmbData="../../data/char-embeddings.txt",speechEmbData="../../data/output_auxiliary_z_vq_encoding.npz",speechLength=180,latentSize=64,speechVecDim=100,wordVecDim=300,batchSize = 32,EPOCH =100,LAMBDA=0.8,wbn=0,sbn=0,sumDir="../../summary",stepToCkpt=10):

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

    self.speechEncoderModel = self.speechEncoder()
    self.discriminatorModel = self.discNetwork()
    self.wordEncoderModel  = self.textEncoder()

    self.speechDecoderModel = self.speechDecoder()
    self.wordDecoderModel = self.textDecoder()


    self.speechOpt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    self.wordOpt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    self.discOpt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    super(archForTrain, self).__init__()

    #Initialise summary writer for tensorboard
    self.sumDir = sumDir
    self.summary_writer = tf.compat.v2.summary.create_file_writer(os.path.join(self.sumDir,'logs/'), flush_millis=10000)
    # self.ckpt_writer = tf.compat.v2.summary.create_file_writer(os.path.join(self.sumDir,'ckpt/'), flush_millis=10000)
    self.summary_writer.set_as_default()

    #Initialise checkpoint manager to save models
    self.global_step =  tf.compat.v1.train.get_or_create_global_step()
    self.ckpt = tf.train.Checkpoint(speechEncModel=self.speechEncoderModel,
		speechDecModel=self.speechDecoderModel,
		textEncModel=self.wordEncoderModel,
		textDecModel=self.wordDecoderModel,
		discriminator=self.discriminatorModel,
		textOpt=self.wordOpt,
		speechOpt=self.speechOpt,
		discrimOpt=self.discOpt)

    self.pathCkpt = os.path.join(self.sumDir,'ckpt')
    self.ckptManager = tf.train.CheckpointManager(self.ckpt, self.pathCkpt, max_to_keep=3)
    print (self.pathCkpt)
    self.stepToCkpt = stepToCkpt
    if self.ckptManager.latest_checkpoint:
        print("Restored from {}".format(self.ckptManager.latest_checkpoint))
        self.ckpt.restore(self.ckptManager.latest_checkpoint)
    else:
        print("Initializing from scratch.")

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


  def logTensorBoard(self,step,discLoss,genLoss,reconsSpeechLoss,reconsWordLoss,speechTotalLoss):
    tf.summary.experimental.set_step(step)
    tf.compat.v2.summary.scalar('disc Loss', discLoss)
    tf.compat.v2.summary.scalar('gen Loss', genLoss)
    tf.compat.v2.summary.scalar('Speech recons Loss', reconsSpeechLoss)
    tf.compat.v2.summary.scalar('Word recons Loss', reconsWordLoss)
    tf.compat.v2.summary.scalar('Total Speech Loss', speechTotalLoss)

  def saveModel(self,epState):
    ## function to save the checkpoint
    save_path = self.ckptManager.save(checkpoint_number=epState+1)
    print("Saved checkpoint for step {}: {}".format(epState+1, save_path))

  def Run(self):
    print("Model Creation")
    #speechEncoderModel = self.speechEncoder()
    #discriminatorModel = self.discNetwork()
    #wordEncoderModel  = self.textEncoder()

    #speechDecoderModel = self.speechDecoder()
    #wordDecoderModel = self.textDecoder()

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
            speechLatentMean,speechLatentVar =self.speechEncoderModel(inputSpeechVec)
            epsSpeech = tf.random.normal(speechLatentMean.shape)
            zVecSpeech = speechLatentMean+tf.multiply(tf.sqrt(tf.exp(speechLatentVar)),epsSpeech)

            wordLatentMean,wordLatentVar = self.wordEncoderModel(inputTextVec)
            epsWord = tf.random.normal(wordLatentMean.shape)
            zVecWord = wordLatentMean+tf.multiply(tf.sqrt(tf.exp(wordLatentVar)),epsWord)
  
            discrimOutSpeech = self.discriminatorModel(zVecSpeech)
            discrimOutWord   = self.discriminatorModel(zVecWord)
    
            #print("No problem")
            print(tf.shape(zVecSpeech))
            decoderOutSpeech = self.speechDecoderModel(zVecSpeech)
            #print(decoderOutSpeech)
            decoderOutWord   = self.wordDecoderModel(zVecWord)

            ########################## Loss Functions ################################
            discLossOut = self.discLoss(discwordInp=discrimOutWord,discSpeechInp=discrimOutSpeech)
            genLossOut = self.genLoss(discSpeechInp=discrimOutSpeech)

            reconsLossSpeech =self.reconLoss(decoderOutSpeech,inputSpeechVec) 
            reconsLossWord = self.reconLoss(decoderOutWord,inputTextVec)

            ###########################################################################

            lossSpeech = genLossOut+self.LAMBDA*reconsLossSpeech
            lossWord = reconsLossWord
            lossDiscModel = discLossOut  

            decoderSpeechGrad = speechTape.gradient(lossSpeech,self.speechDecoderModel.trainable_variables)
            encoderSpeechGrad = speechTape.gradient(lossSpeech,self.speechEncoderModel.trainable_variables)
            discModelGrad     = speechTape.gradient(lossDiscModel,self.discriminatorModel.trainable_variables)

            decoderWordGrad = wordTape.gradient(lossWord,self.wordDecoderModel.trainable_variables)
            encoderWordGrad  = wordTape.gradient(lossWord,self.wordEncoderModel.trainable_variables)

            ###################### apply gradient ######################
            self.speechOpt.apply_gradients(zip(decoderSpeechGrad, self.speechDecoderModel.trainable_variables))
            self.speechOpt.apply_gradients(zip(encoderSpeechGrad, self.speechEncoderModel.trainable_variables))

            self.wordOpt.apply_gradients(zip(decoderWordGrad, self.wordDecoderModel.trainable_variables))
            self.wordOpt.apply_gradients(zip(encoderWordGrad, self.wordEncoderModel.trainable_variables))

            self.discOpt.apply_gradients(zip(discModelGrad, self.discriminatorModel.trainable_variables))
            #return lossSpeech,lossWord,lossDiscModel
            return lossSpeech,lossWord,lossDiscModel,genLossOut,reconsLossSpeech
    
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
                lossSpeech,lossWord,lossDisc,genLossOut,reconsLossSpeech = train(inputSpeechVec=speechBatch,inputTextVec=wordBatch)
                print("EPISODE NO : {:4d}  Batch Speech NO : {:4d} batch Word No :{:2f}   LOSS  speech : {:2f} LOSS Word : {:2f} loss discriminator : {:2f}".format(epNo,self.sbn,self.wbn,lossSpeech,lossWord,lossDisc))

                self.logTensorBoard(step=self.sbn, reconsWordLoss=lossWord, discLoss=lossDisc, genLoss=genLossOut, reconsSpeechLoss=reconsLossSpeech, speechTotalLoss=lossSpeech)
                if ((self.sbn+1)%self.stepToCkpt==0):
                    self.saveModel(self.sbn)

if __name__=="__main__":
    obj = archForTrain()
    obj.Run()
