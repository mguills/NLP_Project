import preprocessing as pre

import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn import model_selection
from sklearn import metrics

def cv_performance(clf, train_data_X, train_data_y, kfs) :
    """
    Determine classifier performance across multiple trials using cross-validation
    
    Parameters
    --------------------
        clf        -- classifier
        train_data -- Data, training data
        kfs        -- array of size n_trials
                      each element is one fold from model_selection.KFold
    
    Returns
    --------------------
        scores     -- numpy array of shape (n_trials, n_fold)
                      each element is the (accuracy) score of one fold in one trial
    """
    
    n_trials = len(kfs)
    n_folds = kfs[0].n_splits
    scores = np.zeros((n_trials, n_folds))
    
    ### ========== TODO : START ========== ###
    # part b: run multiple trials of CV
    for trial in range(n_trials):
    	scores[trial] = cv_performance_one_trial(clf, train_data_X, train_data_y, kfs[trial])
    ### ========== TODO : END ========== ###
    
    return scores


def cv_performance_one_trial(clf, train_data_X, train_data_y, kf) :
    """
    Compute classifier performance across multiple folds using cross-validation
    
    Parameters
    --------------------
        clf        -- classifier
        train_data -- Data, training data
        kf         -- model_selection.KFold
    
    Returns
    --------------------
        scores     -- numpy array of shape (n_fold, )
    				 each element is the (accuracy) score of one fold
    """
    
    scores = np.zeros(kf.n_splits)
    
    ### ========== TODO : START ========== ###
    # part b: run one trial of CV
    for train_index, test_index in kf.split(train_data_X):
    	i = 0
    	X_train, X_test = train_data_X[train_index], train_data_X[test_index]
    	y_train, y_test = train_data_y[train_index], train_data_y[test_index]
    	clf.fit(X_train, y_train)
    	scores[i] = clf.score(X_test, y_test)
    	++i
    ### ========== TODO : END ========== ###
    
    return scores


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
    		score = cv_performance(clf_i, X, y_i, kfs).flatten()
    		mean = np.mean(score)
    		std = np.std(score)
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