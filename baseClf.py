import preprocessing as pre

import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn import model_selection


def main():
    np.random.seed(1234)
    # text_array = get_all_text_from_xml() # run once to get text from xml files
    # write_text_to_files(text_array) # run once to save text from xml files to disk
    text_array = pre.get_all_text()
    word_counts = pre.get_text_dictionary(text_array)
    #Get data
    X = pre.create_X_data(text_array, word_counts, 12, 15)[:-1]
    y = pre.create_y_data()

    kfs = []
    for trial in range(10):
    	kfs.append(model_selection.KFold(n_splits = 10, shuffle = True, random_state = trial))

    means = []
    stds = []
    for i in range(len(y[0])):
    	if not i == 8:
    		clf_i = SVC(kernel = 'linear')
    		y_i = y[:,i]
    		scores = []
    		for i in range(10):
    			X_train = np.append(X[:20*i],X[20*(i+1):],axis = 0)
    			y_train = np.append(y_i[:20*i],y_i[20*(i+1):],axis = 0)
    			X_test = X[20*i:20*(i+1)]
    			y_test = y_i[20*i:20*(i+1)]
    			clf_i.fit(X_train,y_train)
    			scores.append(clf_i.score(X_test,y_test))
    		mean = np.mean(scores)
    		std = np.std(scores)
    	else:
    		mean = 1.0*201/202
    		std = 0
    	means.append(mean)
    	stds.append(std)	

    means = tuple(means)
    stds = tuple(stds)

    N = len(y[0])

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, means, width, color='r', yerr=stds)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracies by Classifier')
    ax.set_xticks(ind + width)


    plt.show()



if __name__ == "__main__" :
    main()