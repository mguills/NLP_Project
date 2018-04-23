import preprocessing as pre

import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn import model_selection
from sklearn import metrics

from entropy import getEntropyDict


def get_AUROC(X,y):
    clf_i = SVC(kernel = 'linear', class_weight = 'balanced')
    test_scores = []
    for i in range(10):
        X_train = np.append(X[:20*i],X[20*(i+1):],axis = 0)
        y_train = np.append(y[:20*i],y[20*(i+1):],axis = 0)
        X_test = X[20*i:20*(i+1)]
        y_test = y[20*i:20*(i+1)]
        clf_i.fit(X_train,y_train)
        y_score = clf_i.decision_function(X_test)
        if len(np.unique(y_test)) != 1:
            AUROC = metrics.roc_auc_score(y_test, y_score)
            test_scores.append(clf_i.score(X_test,y_test))
            
    test_mean = np.mean(test_scores)
    test_std = np.std(test_scores)
    return test_mean, test_std

def get_AUROC_means(X,y):
    test_means = []
    test_stds = []
    for i in range(len(y[0])):
        y_i = y[:,i]
        test_mean, test_std = get_AUROC(X,y_i)
        test_means.append(test_mean)
        test_stds.append(test_std)  
    return test_means, test_stds


'''
input: X and y data FOR ONE CLASSIFIER ONLY, k-fold validation sets
output: training and test mean and std_deviation for the classifier from cross validation
details: Use either from main or as helper in get_means
'''
def get_mean(X,y):
    #8th classifier has no positive examples?

    clf_i = SVC(kernel = 'linear', class_weight = 'balanced')
    test_scores = []
    train_scores = []
    for i in range(10):
        X_train = np.append(X[:20*i],X[20*(i+1):],axis = 0)
        y_train = np.append(y[:20*i],y[20*(i+1):],axis = 0)
        X_test = X[20*i:20*(i+1)]
        y_test = y[20*i:20*(i+1)]
        
        clf_i.fit(X_train,y_train)
        test_scores.append(clf_i.score(X_test,y_test))
        train_scores.append(clf_i.score(X_train, y_train))
            
    test_mean = np.mean(test_scores)
    test_std = np.std(test_scores)
    train_mean = np.mean(train_scores)
    train_std = np.std(train_scores)
    return test_mean, train_mean, test_std, train_std

    '''
    input: X and y data
    output: tuples of training and test means and std_deviation for a linear classifer on each task
    details: use to get means from main(), plot results with plot(). Currently uses a linear classifier with balanced
        class weights
    '''
def get_means(X,y):
    test_means = []
    test_stds = []
    train_means = []
    train_stds = []
    for i in range(len(y[0])):
        y_i = y[:,i]
        test_mean, train_mean, test_std, train_std = get_mean(X,y_i)
        test_means.append(test_mean)
        test_stds.append(test_std)  
        train_means.append(train_mean)
        train_stds.append(train_std)
    return tuple(test_means),tuple(train_means),tuple(test_stds),tuple(train_stds)

    '''
    input: test and training means and std.deviations for a set of classifiers (as tuples)
    output: plots these, returns test and train means for future use
    use: Use in main(), get means from get_means of get_mean
    '''
def plot(test_means,train_means,test_stds,train_stds):
    N = len(test_means)

    ind = np.arange(2*N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, test_means, width, color='r', yerr=test_stds)

    rects2 = ax.bar(ind + width, train_means, width, color='y', yerr=train_stds)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('AUROC')
    ax.set_title('AUROC by Classifier, Frequence and Entropy')
    ax.set_xticks((ind + width) * 0.5)

    ax.legend((rects1[0], rects2[0]), ('Frequency', 'Entropy'))

    plt.show()

    return test_means,train_means

def plot2(test_means,test_stds, y_label):
    N = len(test_means)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, test_means, width, color='r', yerr=test_stds)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('AUROC')
    ax.set_title('AUROC by Dimension')
    ax.set_xticks((ind + width))

    plt.title("AUROC vs. Dimension for classifier" + str(y_label))

    plt.show()

    return test_means

def main():
    np.random.seed(1234)
    # text_array = get_all_text_from_xml() # run once to get text from xml files
    # write_text_to_files(text_array) # run once to save text from xml files to disk
    text_array = pre.get_all_text()
    word_counts, length = pre.get_text_dictionary(text_array)
    
    y = pre.create_y_data()

    best_test_means = [0 for i in range(len(y[0]))]
    best_test_stds = [0 for i in range(len(y[0]))]
    for d in np.arange(1,31,5):
        X = pre.create_X_data(text_array, word_counts, 15, d)
        test_means, test_stds = get_AUROC_means(X,y)
        for i in range(len(y[0])):
            if test_means[i] > best_test_means[i]:
                best_test_means[i] = test_means[i]
                best_test_stds[i] = test_stds[i]
    print "Did testing by Commonality"
    
    rVals = [1,5,10,15,20]

    entropies, bestRndx = getEntropyDict(rVals)
    print "Have entropies"

    entropy_means = []
    entropy_stds = []
    for i in range(len(y[0])):
        print "doing classifier " + str(i)
        wordEntropies = entropies[i]
        #Adjust wordEntropies
        for key in wordEntropies.keys():
            wordEntropies[key] = wordEntropies[key][bestRndx[i]]
        test_means = []
        test_stds = []
        for d in np.arange(1,100,10):
            print "d = " + str(d)
            X = pre.create_X_data(text_array, wordEntropies, d, rVals[bestRndx[i]], True)
            test_mean, test_std = get_AUROC(X,y[:,i])
            test_means.append(test_mean)
            test_stds.append(test_std)
        entropy_means.append(max(test_means))
        entropy_stds.append(max(test_stds))

    plot(tuple(best_test_means), tuple(entropy_means), tuple(best_test_stds), tuple(entropy_stds))
    

if __name__ == "__main__" :
    main()