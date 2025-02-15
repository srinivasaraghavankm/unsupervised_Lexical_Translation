"""
__author__ : Kumar shubham, Srinivasa Raghavan K M
__place__ : IIIT Bangalore
__date__ : 01-12-2019 
"""

import numpy as np 

def createDict(fileName):
    ## function to create dictionary of char
    charDict = {}
    with open(fileName,"rb") as fileNameReader:
        for line in fileNameReader:
            
            strpLine = line.rstrip()
            vecSplit = strpLine.split()
            
            charDict[vecSplit[0].decode()] = np.array([float(item.decode()) for item in vecSplit[1:]],dtype=np.float32)
        
        return charDict

def timitWordEMbed(fileNameTimit,fileNameEMbed,maxCharLen=32,embedingLen=300):
    ### function to crate word embedding based on charecter based embedding 
    charDict = createDict(fileName=fileNameEMbed)
    arrayOfWords = []
    dictOfWordMap = {}
    with open(fileNameTimit,"rb") as fileNameReader:
        for line in fileNameReader:
            word = line.rstrip().decode()
            wordEmbRep = []
            for char in word:
                wordEmbRep.append(charDict[char])
            for _ in range(maxCharLen-len(word)):
                wordEmbRep.append(np.zeros(embedingLen,dtype=np.float32))
            out = np.vstack(wordEmbRep)
            out = out[np.newaxis,:]
            arrayOfWords.append(out)
            dictOfWordMap[word] = arrayOfWords
    finalWordMatrix = np.vstack(arrayOfWords)
    return finalWordMatrix,dictOfWordMap

def timitSpeechEmbed(z_vqListFile,maxFrameLen=180,embedBits=100):
    arrayOfSpeechVec = []
    dictOfSpeechVecMap = {}
    z_vqListReader=np.load(z_vqListFile)
    for audioSegment in sorted(z_vqListReader.keys()):
        padZero=[]
        for _ in range(maxFrameLen- int(z_vqListReader[audioSegment].shape[0])):
            padZero.append(np.zeros(embedBits,dtype=np.float32))
        padZeroNp=np.asarray(padZero,dtype=np.float32)
        speechOut=np.vstack((z_vqListReader[audioSegment],padZeroNp))
        speechOut = speechOut[np.newaxis,:]
        arrayOfSpeechVec.append(speechOut)
        dictOfSpeechVecMap[audioSegment]=speechOut
    finalSpeechMatrix = np.vstack(arrayOfSpeechVec)
    return finalSpeechMatrix,dictOfSpeechVecMap
if __name__=="__main__":
    fileNameEMbed = "../../data/char-embeddings.txt"
    fileNameTimit = "../../data/TIMIT_words.txt"
    # dictOut = createDict(fileNameEMbed)
    finalWordMatrix,embVector = timitWordEMbed(fileNameTimit=fileNameTimit,fileNameEMbed=fileNameEMbed)
    print(finalWordMatrix.shape)
