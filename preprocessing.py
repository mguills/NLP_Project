from xml.dom.minidom import parse, Node
import os
import string
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# need to download nltk for this to work
import nltk
from nltk.tokenize import word_tokenize
# nltk.download("stopwords")
from nltk.corpus import stopwords


def get_text_from_xml(filename, stripPunc = True, removeStopwords = True):
	"""
	Takes in a xml file and returns an array of strings representing tokens in 
	xml file

	Parameters
	--------------------
	        filename			-- string, filename
	        stripPunc			-- boolean flag, string punc from output string
	        removeStopwords		-- boolean flag, remove stopwords from output string
    
    Returns
    --------------------
        dom_text				-- array of strings, represents tokens in xml file

	"""
	dom = parse(filename)
	dom_text = dom.getElementsByTagName("TEXT")[0].toxml().lower() # get raw text from xml
	dom_text = dom_text.encode('ascii','ignore') # convert text to string

	if stripPunc: 
		dom_text = dom_text.translate(None, string.punctuation) # strip punctuation

	dom_tokens = word_tokenize(dom_text) # tokenize text from xml

	if removeStopwords:
		dom_tokens = [word for word in dom_tokens if word not in set(stopwords.words('english'))] # remove stopwords

	return dom_tokens

def get_all_text_from_xml():
	"""
	Loops over files in directory, and gets text from all 202 input xml files.
	Returns array of length 202 with arrays of strings in each entry

	Returns
    	--------------------
        text_array				-- array of string arrays, represents tokens in all xml files
	"""
	text_array = []

	for filename in os.listdir("train"):
		text_array.append(get_text_from_xml(os.path.join("train", filename)))

	return text_array

def get_all_text():
	"""
	Gets all text from train_text directory
	"""
	text_array = []

	for filename in os.listdir("train_text"):
		file = open(os.path.join("train_text", filename), "r")
		words = file.read()
		text_array.append(words.split())

	return text_array

def write_text_to_files(text_array):
	"""
	Loops over files in text_array and writes each file to a train_text 
	directory
	"""
	text_directory = os.path.join(os.getcwd(), r'train_text')
	if not os.path.exists(text_directory):
   		os.makedirs(text_directory)

	for i in range(len(text_array)):
		filename = os.path.join('train_text', "filenum" + str(i) + '.txt')
		np.savetxt(filename, text_array[i], fmt='%s', newline=' ')


def get_labels_from_xml(filename):
	"""
	Takes in an xml file and returns the label vector for the file

	Parameters
	--------------------
	        filename			-- string, filename

	Returns
    	--------------------
        label_vector			-- np array of 1s and 0s. A 1 represents the tag was met, 0 is not met
	"""
	dom = parse(filename)

	tag_names = ["ABDOMINAL", "ADVANCED-CAD", "ALCOHOL-ABUSE", "ASP-FOR-MI",
				"CREATININE", "DRUG-ABUSE", "ENGLISH", "HBA1C", "KETO-1YR",
				"MAJOR-DIABETES", "MAKES-DECISIONS", "MI-6MOS"]

	tag_values = [dom.getElementsByTagName(tag_names[i])[0].getAttribute("met") for i in range(12)]

	label_vector = [1 if tag_values[i] == "met" else 0 for i in range(12)]

	return np.array(label_vector)


def create_y_data():
	"""
	Loops over files in directory, and gets label vectors from all 202 input xml files.
	Returns array of length 202 with lebel vectors arrays in each entry

	Returns
    --------------------
        label_matrix				-- np matrix of shape (202, 12) of label vectors for each xml file
	"""
	label_matrix = np.zeros((202, 12))

	ind = 0
	for filename in os.listdir("train"):
		label_matrix[ind] = get_labels_from_xml(os.path.join("train", filename))
		ind+=1

	return label_matrix

def get_text_dictionary(text_array):
	"""
	Takes in a array of string arrays and returns dictionary of word counts for the input
	"""
	word_counts = defaultdict(int)

	for text in text_array:
		for word in text:
			word_counts[word]+=1

	return word_counts

def plot_common_words(word_counts, k):
	"""
	Takes in a word counts dictionary and outputs a barchart of the k most used words
	"""
	ind = np.arange(k)
	width = 2.0/k

	sorted_counts = sorted(word_counts.iteritems(),key=lambda (k,v): v,reverse=True)
	
	labels = [sorted_counts[i][0] for i in range(k)]
	counts = [sorted_counts[i][1] for i in range(k)]


	fig, ax = plt.subplots()
	rects1 = ax.bar(ind - width/2, counts, width, color='SkyBlue')

	ax.set_ylabel('Counts')
	ax.set_title('Counts of Most Common Words')
	ax.set_xticks(ind)
	ax.set_xticklabels(labels)
	ax.legend()

	plt.show()


def create_X_data(text_array, word_counts, d, j):
	"""
	Takes in a text_array, word_counts, d, and j.
	Creates X vector by using d most used words in dictionary as features.
	Entry i,k in X has a 1 if the text for xml file i uses the word k at least j times.

	Parameters
	--------------------
	        text_array			-- array of string arrays representing words from xml
	        word_counts			-- dictionary of word counts in xml files
	        d 					-- number of words to use as features
	        j 					-- number of words in xml file to get a 1 as a feature

	Returns
	--------------------
			feature_matrix		-- np array of shape (202, d) representing features 
	"""
	n = len(text_array)
	sorted_counts = sorted(word_counts.iteritems(),key=lambda (k,v): v,reverse=True)
	words = [sorted_counts[i][0] for i in range(d)]
	feature_matrix = np.zeros((n, d))

	for i in range(n):
		text = text_array[i]
		curr_counts = get_text_dictionary([text])
		for k in range(d):
			word = words[k]
			for curr_word, curr_count in curr_counts.iteritems():
				if word == curr_word and curr_counts >= j:
					feature_matrix[i][k] = 1

	return feature_matrix


def main() :
	# text_array = get_all_text_from_xml() # run once to get text from xml files
	# write_text_to_files(text_array) # run once to save text from xml files to disk
	text_array = get_all_text()
	word_counts = get_text_dictionary(text_array)
	# plot_common_words(word_counts, 10) # plot common words
	X = create_X_data(text_array, word_counts, 12, 15)
	print X
	y = create_y_data()
	print y
 	print np.sum(y, axis=0)
	



if __name__ == "__main__" :
    main()