import tensorflow as tf
import pandas as pd 
import numpy as np 

import sys
sys.path.insert(0, 'models')

import vgg16
import biLstm
from utils import *

fileTrain = "data/train/All_QA_Pairs_train.txt"
fileValidation = "data/validation/All_QA_Pairs_val.txt"
trainX, trainY = readFile(fileTrain)
valX, valY = readFile(fileValidation)
X = np.array(trainX)

modelW2VTrainX = wordEmbeddings(trainX)
modelW2VTrainY = wordEmbeddings(trainY)
modelW2VValX = wordEmbeddings(valX)
modelW2VValY = wordEmbeddings(valY)


print(type(X))

print(X)
mean = getMean(X)
print(mean)
#modelW2VTrainX.build_vocab(trainX)
#modelW2VTrainX.train(trainX,total_examples=len(trainX),epochs=10)

#print(modelW2VTrainX.most_similar('good'))

#learningRate = np.power(10.0,-2.0)
#trainingSteps = 20000
#batchSize = 1000
#displayStep = 1000
#numInput = 119
#timeStep = 1
#numUnits = 119
#numClasses = 1
#nunLayers = 8

