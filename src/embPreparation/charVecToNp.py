"""
__author__ : Kumar shubham, srinivas
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

def timitWordEMbed(fileName,charDict,maxCharLen=32,embedingLen=300):
	### function to crate word embedding based on charecter based embedding 
	arrayOfWords = []
	dictOfWordMap = {}
	with open(fileName,"rb") as fileNameReader:
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
if __name__=="__main__":
	fileNameEMbed = "../../data/char-embeddings.txt"
	fileNameTimit = "../../data/TIMIT_words.txt"
	dictOut = createDict(fileNameEMbed)
	finalWordMatrix,embVector = timitWordEMbed(fileName=fileNameTimit,charDict=dictOut)
	print(finalWordMatrix.shape)