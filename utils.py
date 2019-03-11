from gensim.models import word2vec
import numpy as np

def readFile(nameFile):
    sentencesX = []
    sentencesY = []
    f = open(nameFile, "r")
    for l in f:
        separators = l.split("|")
        sentencesX.append(separators[1].split())
        sentencesY.append(separators[2].split())
    return sentencesX, sentencesY

def wordEmbeddings(sentences):
    model = word2vec.Word2Vec(sentences, size=200)
    return model 

def getMean(xTrain):
    numWords = []
    for x in xTrain:
        count = len(x)
        numWords.append(count)
    mean = np.mean(numWords)
    return int(mean)

def featureVecMethod(words, model, num_features):

    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0

    index2word_set = set(model.wv.index2word)
    
    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])

    featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:

        if counter%1000 == 0:
            print("Review %d of %d"%(counter,len(reviews)))
            
        reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features)
        counter = counter+1
        
    return reviewFeatureVecs