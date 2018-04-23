from xml.dom.minidom import parse, Node
import os
import string
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# need to download nltk for this to work
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download("stopwords")
from nltk.corpus import stopwords


######################################################################
# extracting tokens from xml
######################################################################

def get_text_from_xml(filename, stripPunc = True, removeStopwords = True, stem = True):
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

	if stem:
		tagged_tokens = nltk.pos_tag(dom_tokens)
		dom_tokens = [lemmatizer.lemmatize(word[0], get_wordnet_pos(word[1])) for word in tagged_tokens]

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

	for filename in os.listdir("train"): # loops over files in xml directory
		text_array.append(get_text_from_xml(os.path.join("train", filename))) # appends text from each file to text_array

	return text_array

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
	dom = parse(filename) # parse the xml file

	tag_names = ["ABDOMINAL", "ADVANCED-CAD", "ALCOHOL-ABUSE", "ASP-FOR-MI",
				"CREATININE", "DRUG-ABUSE", "ENGLISH", "HBA1C", "KETO-1YR",
				"MAJOR-DIABETES", "MAKES-DECISIONS", "MI-6MOS"]

	tag_values = [dom.getElementsByTagName(tag_names[i])[0].getAttribute("met") for i in range(12)] # get "met" or "not met" for each tag

	label_vector = [1 if tag_values[i] == "met" else 0 for i in range(12)] # make label vector for file 

	return np.array(label_vector)

def get_all_labels_from_xml():
	"""
	Gets all labels from all files
	"""
	labels_array = []

	for filename in os.listdir("train"):
		labels_array.append(get_labels_from_xml(os.path.join("train", filename)))

	return labels_array


######################################################################
# text preprocessing
######################################################################

def get_all_text():
	"""
	Gets all text from train_text directory
	"""
	text_array = []

	for filename in os.listdir("train_text"): # loop through files in text directory
		if filename.startswith('filenum'):
			file = open(os.path.join("train_text", filename), "r")
			words = file.read()
			text_array.append(word_tokenize(words)) # append words from file to text_array

	return text_array

def get_text_for_tfidf():
	"""
	Gets all text in the form file: text for tfidf form.
	"""
	text_dict = {}
	for filename in os.listdir("train_text"): # loop through files in text directory
		if filename.startswith('filenum'):
			file = open(os.path.join("train_text", filename), "r")
			words = file.read()
			text_dict[file] = words
	return text_dict

def write_text_to_files(text_array):
	"""
	Loops over text for each of the input xml files in text_array 
	and writes the text to a train_text directory
	"""
	text_directory = os.path.join(os.getcwd(), r'train_text') 

	if not os.path.exists(text_directory):
		os.makedirs(text_directory) # create new train_text directory

	for i in range(len(text_array)):
		filename = os.path.join('train_text', "filenum" + str(i) + '.txt')
		np.savetxt(filename, text_array[i], fmt='%s', newline=' ') # write text to file

def get_all_labels():
	"""
	Gets all labels from train_labels directory
	"""
	labels_array = []

	for filename in os.listdir("train_labels"):
		file = open(os.path.join("train_labels", filename), "r")
		words = file.read()
		labels_array.append(words.split('\n')[:-1])

	return labels_array

def write_labels_to_files(labels_array):
	"""
	Loops over labels for each of the input xml files in labels_array
	and writes the text to a train_labels directory
	"""

	labels_directory = os.path.join(os.getcwd(), r'train_labels')

	if not os.path.exists(labels_directory):
		os.makedirs(labels_directory)

	for i in range(len(labels_array)):
		filename = os.path.join('train_labels', 'filenum' + str(i) + '.txt')
		np.savetxt(filename, labels_array[i], fmt='%d')

def get_text_dictionary(text_array):
	"""
	Takes in an array of string arrays and returns dictionary of word counts for the input
	and the number of distinct words
	"""
	word_counts = defaultdict(int)

	for text in text_array:
		for word in text: # count number of occurences for each word
			word_counts[word]+=1

	return word_counts, len(word_counts)

def get_text_dictionary_split(text_array, labels_array, i, avg=False):
	"""
	Tkes in an array of string arrays, an array of the corresponding labels, an
	integer i representing a split on the feature i and a bool flag avg. 
	"""
	word_counts_met = defaultdict(int) # initalize dictionaries for met and not met
	word_counts_not = defaultdict(int)

	met_count = 0
	not_count = 0

	for j in range(len(text_array)):
		met = labels_array[j][i] == '1' # if the class was met in the given file
		if met:
			met_count+=1
		else:
			not_count+=1
		for word in text_array[j]:
			if met:
				word_counts_met[word]+=1 # count occurences
			else:
				word_counts_not[word]+=1 # count occurences

	if avg:
		word_counts_met = defaultdict(int, {k: v/met_count for k,v in word_counts_met.iteritems()}) # get averages for each dict
		word_counts_not = defaultdict(int, {k: v/not_count for k,v in word_counts_not.iteritems()})
		print met_count
		print not_count

	return word_counts_met, word_counts_not

def create_clamp_data_word_helper(labels_array, i):
	"""
	"""

	semantic_list = get_semantic_list()

	met_count = 0
	not_count = 0

	semantic_counts_met = {"drug": defaultdict(int), "treatment": defaultdict(int), "test": defaultdict(int), 
							"problem": defaultdict(int), "temporal": defaultdict(int),  "BDL": defaultdict(int), 
							"SEV": defaultdict(int), "labvalue": defaultdict(int), "COU": defaultdict(int) } 

	semantic_counts_not = {"drug": defaultdict(int), "treatment": defaultdict(int), "test": defaultdict(int), 
							"problem": defaultdict(int), "temporal": defaultdict(int),  "BDL": defaultdict(int), 
							"SEV": defaultdict(int), "labvalue": defaultdict(int), "COU": defaultdict(int) } 

	
	for filename in semantic_list.keys():
		filenum = int(filename[7:-4])
		met = labels_array[filenum][i] == '1' # if the class was met in the given file
		if met:
			met_count+=1
		else:
			not_count+=1
		for semantic in semantic_list[filename].keys():
			for value in semantic_list[filename][semantic]:
				if value[0] == 'present' or value[0] == 'N/A':
					if met:
						semantic_counts_met[semantic][value[1]] += 1
					else:
						semantic_counts_not[semantic][value[1]] += 1


	for semantic in semantic_counts_met.keys():
		for word, count in semantic_counts_met[semantic].iteritems():
			semantic_counts_met[semantic][word] = count*1.0/met_count

	for semantic in semantic_counts_not.keys():
		for word, count in semantic_counts_not[semantic].iteritems():
			semantic_counts_not[semantic][word] = count*1.0/not_count

	return semantic_counts_met, semantic_counts_not

def create_clamp_data_word(labels_array):
	"""
	"""
	tag_names = ["ABDOMINAL", "ADVANCED-CAD", "ALCOHOL-ABUSE", "ASP-FOR-MI",
				"CREATININE", "DRUG-ABUSE", "ENGLISH", "HBA1C", "KETO-1YR",
				"MAJOR-DIABETES", "MAKES-DECISIONS", "MI-6MOS"]

	for i in range(12):
		semantic_counts_met, semantic_counts_not = create_clamp_data_word_helper(labels_array, i)
		for semantic in semantic_counts_met.keys():
			met_differences, not_differences = get_word_differences(semantic_counts_met[semantic], 
												semantic_counts_not[semantic])
			print "Largest disparities for met in " + tag_names[i] + " " + semantic + ": "
			print sorted(met_differences.iteritems(),key=lambda (k,v): v, reverse=True)[:5]
			print "Largest disparities for not met in " + tag_names[i] + " " + semantic + ": "
			print sorted(not_differences.iteritems(), key=lambda (k,v): v, reverse=True)[:5]

def get_word_differences(word_counts_met, word_counts_not):
	"""
	Gets differences in word counts
	"""
	met_differences = {k: v-word_counts_not[k] for k,v in word_counts_met.iteritems()}
	not_differences = {k: v-word_counts_met[k] for k,v in word_counts_not.iteritems()}

	return met_differences, not_differences

def get_wordnet_pos(treebank_tag):
	"""
	Part of speech mapping used for lemmatization.
	"""
	if treebank_tag.startswith('J'):
		return wordnet.ADJ
	elif treebank_tag.startswith('V'):
		return wordnet.VERB
	elif treebank_tag.startswith('N'):
		return wordnet.NOUN
	elif treebank_tag.startswith('R'):
		return wordnet.ADV
	else:
		return wordnet.NOUN


######################################################################
# data visualization
######################################################################

def plot_common_words(word_counts, k):
	"""
	Takes in a word counts dictionary and outputs a barchart of the k most used words
	"""
	ind = np.arange(k) # indicies of each word
	width = 2.2/k # width of each bar

	sorted_counts = sorted(word_counts.iteritems(),key=lambda (k,v): v,reverse=True) # sort words by num appearances
	
	labels = [sorted_counts[i][0] for i in range(k)] # labels of each bar
	counts = [sorted_counts[i][1] for i in range(k)] # count for each bar


	fig, ax = plt.subplots()
	rects1 = ax.bar(ind - width/2, counts, width, color='SkyBlue')

	ax.set_ylabel('Counts')
	ax.set_title('Counts of Most Common Words')
	ax.set_xticks(ind)
	ax.set_xticklabels(labels)
	ax.legend()

	plt.show()

def plot_stacked_words(text_array, labels_array, word_counts, k, avg=False):
	"""
	Plots 12 stacked bar charts (one for each label class). Each stacked bar chart displays
	the k most common words. The stacks are the counts for a given word in the files where
	the given feature was "met" and "not met".
	"""

	ind = np.arange(k)
	width = 4.0/k if avg else 2.0/k

	tag_names = ["ABDOMINAL", "ADVANCED-CAD", "ALCOHOL-ABUSE", "ASP-FOR-MI",
				"CREATININE", "DRUG-ABUSE", "ENGLISH", "HBA1C", "KETO-1YR",
				"MAJOR-DIABETES", "MAKES-DECISIONS", "MI-6MOS"]

	for i in range(12):
		sorted_counts = sorted(word_counts.iteritems(),key=lambda (k,v): v,reverse=True) # sort words by num appearances
	
		labels = [sorted_counts[j][0] for j in range(k)] # labels of each bar

		word_counts_met, word_counts_not = get_text_dictionary_split(text_array, labels_array, i, avg) # get word counts for met and not met

		met_counts = [word_counts_met[labels[j]] for j in range(k)] # word counts for met
		not_counts = [word_counts_not[labels[j]] for j in range(k)] # word counts for not met
		print met_counts
		print not_counts

		p1 = plt.bar(ind, met_counts, width, color='r') # create splits for bar chart
		p2 =  plt.bar(ind + width, not_counts, width, color='y') if avg \
		else plt.bar(ind, not_counts, bottom=met_counts)

		plt.ylabel('Counts')
		title = 'Counts of Most Common Words Split on ' + tag_names[i] + ' Avg' if avg \
				else 'Counts of Most Common Words Split on ' + tag_names[i] + ' Met/Not Met'
		plt.title(title)
		plt.xticks(ind + (avg*width/2), labels)
		legend_label = ('Met average occurences per file', 'Not met average occurences per file') if avg \
						else ('Met most common words', 'Not met most common words')
		plt.legend((p1[0], p2[0]), legend_label)

		plt.show()

def plot_word_differences(text_array, labels_array, k):
	"""
	Plots a grouped bar chart of the k words with the biggest differences in 
	number of appearances in met/not met files. The first k bars give the average counts for 
	words that appear more often in met files, and the next k bars give the average counts for the
	k words that appear more often in not met files.
	"""
	ind = np.arange(2*k)
	width = 2.0/k

	tag_names = ["ABDOMINAL", "ADVANCED-CAD", "ALCOHOL-ABUSE", "ASP-FOR-MI",
				"CREATININE", "DRUG-ABUSE", "ENGLISH", "HBA1C", "KETO-1YR",
				"MAJOR-DIABETES", "MAKES-DECISIONS", "MI-6MOS"]

	for i in range(12):
		word_counts_met, word_counts_not = get_text_dictionary_split(text_array, labels_array, i, True) # get word counts for met and not met

		met_differences, not_differences = get_word_differences(word_counts_met, word_counts_not)

		sorted_met = sorted(met_differences.iteritems(), key=lambda (k,v): v, reverse=True)
		sorted_not = sorted(not_differences.iteritems(), key=lambda (k,v): v, reverse=True)

		met_labels = [sorted_met[j][0] for j in range(k)]
		not_labels = [sorted_not[j][0] for j in range(k)]
		labels = met_labels + not_labels

		met_counts = [word_counts_met[labels[j]] for j in range(2*k)]
		not_counts = [word_counts_not[labels[j]] for j in range(2*k)]

		p1 = plt.bar(ind, met_counts, width, color='r')
		p2 = plt.bar(ind + width, not_counts, width, color='y')

		plt.ylabel('Counts')
		plt.title('Counts of biggest difference between ' + tag_names[i] + ' Met/Not Met')
		plt.xticks(ind + width/2, labels)
		plt.legend((p1[0], p2[0]), ('Met', 'Not met'))

		plt.show()


######################################################################
# create X and y data
######################################################################

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
	sorted_counts = sorted(word_counts.iteritems(),key=lambda (k,v): v,reverse=True) # sort words by appearances

	words = [sorted_counts[i][0] for i in range(d)] # get d most used words
	feature_matrix = np.zeros((n, d)) # create feature matrix

	for i in range(n):
		text = text_array[i] # text from file i
		curr_counts = get_text_dictionary([text])[0]

		for k in range(d):
			word = words[k] # current word
			for curr_word, curr_count in curr_counts.iteritems():
				if word == curr_word and curr_counts >= j: # see if current word appears at least j times
					feature_matrix[i][k] = 1

	return feature_matrix

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
		label_matrix[ind] = get_labels_from_xml(os.path.join("train", filename)) # create labels based on file
		ind+=1

	return label_matrix

def create_tfidf_data():
	"""
	Creates a 202x15 matrix. Each entry (i,j) gives the tfidf of word j in 
	file number i. 
	"""
	text_dict = get_text_for_tfidf()
	tfidf = TfidfVectorizer(max_features=15)
	tfs = tfidf.fit_transform(text_dict.values())

	return tfs

def generate_clamp_files():
	"""
	Generates the CLAMP files from the text medical records. Make sure to adjust the input and output 
	variables within run_attribute_pipeline.sh and run_ner_pipeline.sh
	"""
	path = os.getcwd()
	path += '/ClampCmd_1.4.0'
	os.chdir(path)
	os.system('pwd')
	os.system('./run_ner_pipeline.sh')
	os.system('./run_attribute_pipeline.sh')

def get_semantic_list():
	semantic_dict = {}
	for file in os.listdir("ClampCmd_1.4.0/attribute_output"):
		semantics = {"drug":[] , "treatment": [], "test": [], "problem": [], "temporal":[],  
					"BDL": [], "SEV": [] , "labvalue" : [], "COU" : []}
		if '.txt' in file: #only dealing with the txt files not xmi files 
			text = open("ClampCmd_1.4.0/attribute_output/" + file).readlines()
			for line in text:
				for key in semantics.keys():
					tag = 'semantic=' + key
					if tag in line and 'NamedEntity' in line:
						finalLine = []
						line = line.split('\t')
						assertion = ''
						diagnosis = ''
						for item in line:
							if "assertion=" in item:
								assertion = item[10:].strip()
							if "ne=" in item:
								diagnosis = item[3:].strip()
							if diagnosis != '':
								if assertion == '':
									finalLine.append(("N/A", diagnosis))
								else:
									finalLine.append ((assertion,diagnosis))
						semantics[key] += finalLine
			semantic_dict[str(file)] = semantics
	return semantic_dict


def main() :
	# text_array = get_all_text_from_xml() # run once to get text from xml files
	# labels_array = get_all_labels_from_xml() # run once to get labels from xml files
	# write_text_to_files(text_array) # run once to save text from xml files to disk
	# write_labels_to_files(labels_array) # run once to save labels from xml files to disk
	# text_array = get_all_text()
	labels_array = get_all_labels()
	create_clamp_data_word(labels_array)
	# word_counts, distinct_words = get_text_dictionary(text_array)
	# plot_common_words(word_counts, 10) # plot common words
	# plot_stacked_words(text_array, labels_array, word_counts, 12, avg=True)
	# plot_word_differences(text_array, labels_array, 6)
	# X = create_X_data(text_array, word_counts, 12, 15)
	# y = create_y_data()
	# tfs = create_tfidf_data()
	#print tfs
	#print distinct_words
	# generate_clamp_files() run once to get CLAMP files from txt files



if __name__ == "__main__" :
	main()
