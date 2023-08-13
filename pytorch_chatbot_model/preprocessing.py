import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

def tokenizer(sentence):
	return word_tokenize(sentence)

def stem(word):
	return stemmer.stem(word.lower())

def bag_of_words(tokenized_sent, all_words):
	tokenized_sent = [stem(w) for w in tokenized_sent]

	bag = np.zeros(len(all_words), dtype = np.float32)

	for idx, w in enumerate(all_words):
		if w in tokenized_sent:
			bag[idx] = 1.0

	return bag