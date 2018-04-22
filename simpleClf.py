import preprocessing as pre

import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn import model_selection


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
        if not i==8:
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

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, test_means, width, color='r', yerr=test_stds)

    rects2 = ax.bar(ind + width, train_means, width, color='y', yerr=train_stds)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracies by Classifier, Training and Test')
    ax.set_xticks((ind + width) * 0.5)
    ax.set_xticklabels(('Test Accuraccy', 'Training Accuraccy'))

    ax.legend((rects1[0], rects2[0]), ('Test', 'Train'))

    plt.show()

    return test_means,train_means

def main():
    np.random.seed(1234)
    # text_array = get_all_text_from_xml() # run once to get text from xml files
    # write_text_to_files(text_array) # run once to save text from xml files to disk
    text_array = pre.get_all_text()
    word_counts, length = pre.get_text_dictionary(text_array)
    
    X = pre.create_X_data(text_array, word_counts, 12, 15)
    y = pre.create_y_data()
    print y

    test_means, train_means, test_stds, train_stds = get_means(X,y)
    print test_means
    '''wordNums = np.arange(0,100)
    bestmeans = [0,0,0,0,0,0,0,0,0,0,0,0]
    bestNum = [0,0,0,0,0,0,0,0,0,0,0,0]
    for wordNum in wordNums:
        X = pre.create_X_data(text_array, word_counts, wordNum, 15)[:-1]
        y = pre.create_y_data()

        test_means, train_means, test_stds, train_stds = get_means(X,y)


        #plot(test_means, train_means, test_stds, train_stds)
        for i in range(len(bestmeans)):
            if bestmeans[i]<=test_means[i]:
                bestmeans[i] = test_means[i]
                bestNum[i]=wordNum
        #print test_means
        #print train_means

        print best_means 
        print bestNum'''


if __name__ == "__main__" :
    main()