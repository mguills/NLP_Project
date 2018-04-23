import simpleClf as clf 
import preprocessing as pre
from entropy import getEntropyDict


def get_scores():
	y = pre.create_y_data()
	scores = []
	rVals = [10]
	full_dict = getEntropyDict(rVals)[0]
	for i in range(len(y[0])):
		test_means = []
		test_stds = []
		entropyDict = full_dict[i]
		sorted_dict = sorted(entropyDict.iteritems(), key=lambda (k,v): v[0])
		for d in range(10, 50, 5):
			vocab = {sorted_dict[i][0]: i for i in range(d)}
			X = pre.create_tfidf_data(vocab)
			test_mean, test_std = clf.get_AUROC(X,y[:,i])
			test_means.append(test_mean)
			test_stds.append(test_std)

		scores.append((test_means, test_stds))

	return scores
	