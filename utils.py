from gensim.models import word2vec

def readFile(nameFile):
    lines = []
    f = open(nameFile, "r")
    for l in f:
        lines.append(l)
    return lines 

def wordEmbeddings(sentences):
    model = word2vec.Word2Vec(sentences, size=200)
    return model 