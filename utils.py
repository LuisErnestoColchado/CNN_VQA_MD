from gensim.models import word2vec

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