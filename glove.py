import sys, pickle, numpy as np
from data import Vocab


gloveFile = sys.argv[1]
path_to_vocab = sys.argv[2]
save_file = sys.argv[3]
emb_dim = int(sys.argv[4])

vocab = Vocab('../data/vocab', 50000)

#https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    with open(gloveFile,'r') as f:
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print("Done.",len(model)," words loaded!")
        return model

if __name__ == "__main__":

    glove300_encodings = loadGloveModel(gloveFile)
    matrix = np.zeros((len(vocab._word_to_id) , emb_dim), dtype=np.float32)

    print("Glove matrices loaded, producing embedding matrix")

    for element in vocab._word_to_id:
        if element in glove300_encodings:
            matrix[vocab._word_to_id[element]] = glove300_encodings[element] + matrix[vocab._word_to_id[element]]

    print("Embedding matrices successfully created")

    with open(save_file, "wb") as file:
        pickle.dump(matrix, file)

    print("Embedding matrices successfully saved using pickle")
