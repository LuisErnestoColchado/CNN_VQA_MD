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
sentencesTrain =  readFile(fileTrain)
sentencesVal = readFile(fileValidation)

modelW2VTrain = wordEmbeddings(sentencesTrain)
modelW2VVal = wordEmbeddings(sentencesVal)

