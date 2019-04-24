import pickle
import argparse
import os
import sys
from data import Vocab
import numpy as np


def createEmb(vocab_dir, emb_dir, size):

	vocab = Vocab(vocab_dir, size)

	word_to_id = vocab._word_to_id
	id_to_word = vocab._id_to_word

	with open(emb_dir + "/skipgram_matrice.p", 'rb') as pickle_file:
	    emb = pickle.load(pickle_file)
	    
	with open(emb_dir + "/skipgram_labels.p", 'rb') as pickle_file:
	    lab = pickle.load(pickle_file)

	matrix = np.zeros((len(word_to_id) , emb.shape[1]), dtype=np.float32)
	lables = list(lab.values())
	for element in word_to_id:
	    if element in lables:
	        matrix[word_to_id[element]] = emb[lables.index(element)] + matrix[word_to_id[element]]
	    else:
	        matrix[word_to_id[element]] = emb[lables.index("UNK")] + matrix[word_to_id[element]]

	with open("skipgram_matrix.p", "wb") as file:
	    pickle.dump(matrix, file)
	return
def main():

	current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--vocab_file',
		type=str,
		default=os.path.join(current_path, 'data/vocab'),
		help='The directory for vocab file.')

	parser.add_argument(
		'--emb_dir',
		type=str,
		default=os.path.join(current_path, 'data/'),
		help='The directory for label and matrix.')

	parser.add_argument(
		'--v_size',
		type=int,
		default=50000,
		help='size of vocab')

	flags, unused_flags = parser.parse_known_args()

	print(flags.vocab_file, flags.emb_dir)

	createEmb(flags.vocab_file, flags.emb_dir, flags.v_size)

if __name__ == '__main__':
	main()