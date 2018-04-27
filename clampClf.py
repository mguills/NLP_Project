import matplotlib.pyplot as plt
import preprocessing as pre
import entropy
import numpy as np
import simpleClf




def create_X_data_CLAMP(text_array, words_array): #generate words array outside this function
    X = np.zeros( (len(text_array), len(words_array)) )
    for example in range(len(text_array)):
        for word in range(len(words_array)):
            if words_array[word] in text_array[example]:
                X[example][word] = 1
    return X

def generate_words_array(tupleList):
    word_array = []
    for wordTuple in tupleList:
        word_array.append(wordTuple[0])
    return word_array

def main():
    kvalDict = entropy.getClampEntropy()
    text_array = pre.get_all_text(True)
    y = pre.create_y_data()
    # Loop over number of words 
    #for num_words in range(1, 100, 10): # Loop over number of words hyper parameter
    #    minEntropies = entropy.getMinEntropies(kvalDict, num_words) # generate new entropy array based on the num words
    #    for semantic_val in range(len(pre.SEMANTICS)): # Loop over number of semantics hyper parameter 
    #        curEntropies = minEntropies[semantic_val]
    #        for classifier in range(len(curEntropies)): # Loop over the 11 classifiers 
    #            test_means = []
    #            test_stds = []
    #            words_array = generate_words_array(curEntropies[classifier])
    #            X = create_X_data_CLAMP(text_array, words_array)
    #            test_mean, test_std = simpleClf.get_AUROC(X,y[:,classifier])
    #            test_means.append(test_mean)
    #            test_stds.append(test_std)
    #        
    class_means = [[]] * len(y[0])
    print class_means
    class_stds = [[]* len(y[0])] 
    for classifier in range(len(y[0])):
        print "Classifier Num: ", classifier
        test_means = [[]] * 20
        test_stds = [[]] * 20
        counter = 0
        for num_words in range(1, 100, 5):
            print "num words: ", num_words
            minEntropies = entropy.getMinEntropies(kvalDict, num_words) # generate new entropy array based on the num words
            mean_array = []
            stds_array = []
            for semantic_val in range(len(pre.SEMANTICS) - 1): # Loop over number of semantics hyper parameter 
                print "semantic_val: ", semantic_val
                curEntropies = minEntropies[semantic_val]
                words_array = generate_words_array(curEntropies[classifier])
                X = create_X_data_CLAMP(text_array, words_array)
                test_mean, test_std=simpleClf.get_AUROC(X, y[:, classifier])
                mean_array.append(test_mean)
                stds_array.append(test_std)
            test_means[counter] = mean_array
            test_stds[counter] = stds_array
            counter += 1
            
        
        # Plot heat map here with test_means 
        #test_means_new = np.empty(20, len(pre.SEMANTICS) - 1)
        #for i in range(len(test_means)):
        #    print "I: ", i
        #    for j in range(len(test_means[i])):
        #        print "J: ", j
        #        test_means[i][j] = test_means_new[j][i]
        #print test_means_new

        #test_means = map(test_means, zip( * l))


        test_means = np.array(test_means).T.tolist()
        
        plt.title(pre.TAG_NAMES[classifier])
        plt.imshow(test_means, cmap='hot', interpolation='nearest')
        plt.xticks(np.arange(0, 20), np.arange(0,100,step=5))
        plt.xlabel("Number of Extracted Phrases")
        plt.yticks(np.arange(7), pre.SEMANTICS[: - 1])
        plt.ylabel("Different Semantics Used -- Inclusive")
        plt.colorbar()
        plt.show()

        #class_means[classifier] = test_means
        #class_stds[classifier] = test_stds
        
    print class_means
    print class_stds

    return 0

main()
                






    # Loop over sematics hyperparamteres        
    # Loop over all the classifiers 
    # Loop over number of words 
