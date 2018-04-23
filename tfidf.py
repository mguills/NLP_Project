import simpleClf as clf 
import preprocessing as pre

def get_scores():
	y = pre.create_y_data()
	scores = []

	for i in range(len(y[0])):
		test_means = []
		test_stds = []

		for d in range(10, 100, 5):
			X = pre.create_tfidf_data(d)
			test_mean, test_std = clf.get_AUROC(X,y[:,i])
			test_means.append(test_mean)
			test_stds.append(test_std)

		scores.append((test_means, test_stds))

	return scores
	