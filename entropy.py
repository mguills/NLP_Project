import preprocessing as pre

import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn import model_selection

def entropy(y):
	if(len(y) == 0):
		return 0
	P = 1.0*np.count_nonzero(y)/len(y)

	if P==0 or P==1:
		return 0

	H = -(P*np.log2(P)) - ((1-P)*np.log2(1-P))
	return H

def conditional_entropy(X,y):
	metProb = 1.0*np.count_nonzero(X)/len(X)

	#Examples with negative feature
	E1 = y[X==0]
	H1 = entropy(E1)

	#examples with positive feature
	E2 = y[X==1]
	H2 = entropy(E2)

	H = (1-metProb)*H1 + metProb*H2

	return H

def get_X_label(word, r, text_array):
	X = np.zeros((len(text_array),))
	for i, example in enumerate(text_array):
		count = len(np.nonzero(np.array(example)==word)[0])
		if count >= r:
			X[i] = 1
	return X

def getEntropyDict():
	# text_array = get_all_text_from_xml() # run once to get text from xml files
	# write_text_to_files(text_array) # run once to save text from xml files to disk
	text_array = pre.get_all_text()
	word_counts, length = pre.get_text_dictionary(text_array)
	y = pre.create_y_data()

	sortedWords = sorted(word_counts.items(), key = lambda word: word[1])

	uniqueWords = sortedWords[-100:]
	print uniqueWords
	for i in range(len(uniqueWords)):
		uniqueWords[i] = uniqueWords[i][0]

	rVals = [15]

	allEntropies = []
	for label in range(len(y[0])):
		entropyDict = {}
		for word in uniqueWords:
			r_entropies = np.zeros((5,))
			for i,r in enumerate(rVals):
				X = get_X_label(word, r, text_array)
				e = conditional_entropy(X,y[:,label])
				r_entropies[i] = e
			entropyDict[word] = r_entropies
		allEntropies.append(entropyDict)


	return allEntropies