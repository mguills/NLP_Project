import preprocessing as pre

import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn import model_selection


def main():
    np.random.seed(1234)
    # text_array = get_all_text_from_xml() # run once to get text from xml files
    # write_text_to_files(text_array) # run once to save text from xml files to disk
    text_array = pre.get_all_text()
    word_counts, length = pre.get_text_dictionary(text_array)
    #Get data
    X = pre.create_X_data(text_array, word_counts, 12, 15)[:-1]
    y = pre.create_y_data()

    kfs = []
    for trial in range(10):
        kfs.append(model_selection.KFold(n_splits = 10, shuffle = True, random_state = trial))

    test_means = []
    for i in range(len(y[0])):
        if not i == 8:
            clf_i = DummyClassifier(strategy = 'most_frequent')
            y_i = y[:,i]
            test_scores = []
            for i in range(10):
                X_train = np.append(X[:20*i],X[20*(i+1):],axis = 0)
                y_train = np.append(y_i[:20*i],y_i[20*(i+1):],axis = 0)
                X_test = X[20*i:20*(i+1)]
                y_test = y_i[20*i:20*(i+1)]
                clf_i.fit(X_train,y_train)
                test_scores.append(clf_i.score(X_test,y_test))
            test_mean = np.mean(test_scores)
        else:
            mean = 1.0*201/202
            std = 0
        test_means.append(test_mean)
    print test_means   


if __name__ == "__main__" :
    main()