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

    test_means = []
    test_stds = []
    train_means = []
    train_stds = []
    for i in range(len(y[0])):
    	if not i == 8:
    		clf_i = SVC(kernel = 'linear')
    		y_i = y[:,i]
    		test_scores = []
    		train_scores = []
    		for i in range(10):
    			X_train = np.append(X[:20*i],X[20*(i+1):],axis = 0)
    			y_train = np.append(y_i[:20*i],y_i[20*(i+1):],axis = 0)
    			X_test = X[20*i:20*(i+1)]
    			y_test = y_i[20*i:20*(i+1)]
    			clf_i.fit(X_train,y_train)
    			test_scores.append(clf_i.score(X_test,y_test))
    			train_scores.append(clf_i.score(X_train, y_train))
    		test_mean = np.mean(test_scores)
    		test_std = np.std(test_scores)
    		train_mean = np.mean(train_scores)
    		train_std = np.std(train_scores)
    	else:
    		mean = 1.0*201/202
    		std = 0
    	test_means.append(test_mean)
    	test_stds.append(test_std)	
    	train_means.append(train_mean)
    	train_stds.append(train_std)

    test_means = tuple(test_means)
    train_stds = tuple(test_stds)

    N = len(y[0])

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, test_means, width, color='r', yerr=test_stds)

    train_means = tuple(train_means)
    train_stds = tuple(train_stds)
    rects2 = ax.bar(ind + width, train_means, width, color='y', yerr=train_stds)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracies by Classifier, Training and Test')
    ax.set_xticks((ind + width) * 0.5)
    ax.set_xticklabels(('Test Accuraccy', 'Training Accuraccy'))

    ax.legend((rects1[0], rects2[0]), ('Test', 'Train'))


    plt.show()



if __name__ == "__main__" :
    main()