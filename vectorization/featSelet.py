# -*- coding: utf-8 -*-



def tree_select(trainSet, testSet): 	# input as numpy array
	from sklearn.ensemble import ExtraTreesClassifier
	#import matplotlib.pyplot as plt
	import numpy
	
	X, y = trainSet[:,1:], trainSet[:,0]
	#print [X.shape, y.shape]
	clf = ExtraTreesClassifier(max_depth=10, n_jobs=-1, bootstrap=True, n_estimators=25)
	clf.fit(X, y)	
	importances = clf.feature_importances_
	#std = numpy.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
	indices = numpy.argsort(importances)[::-1]
	print("Feature ranking:")
	
	for f in range(importances.shape[0]):
	    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

	# Plot the feature importances of the forest
	'''
	plt.figure()
	plt.title("Feature importances")
	plt.bar(range(10), importances[indices],
	       color="r", yerr=std[indices], align="center")
	plt.xticks(range(10), indices)
	plt.xlim([-1, 10])
	plt.show()
	
	#clf.feature_importances_  	
	#print X_new.shape
	'''
	
	testSet = clf.transform(testSet)
	X_new = clf.transform(X)
	#raw_input('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
	#
	return numpy.hstack((numpy.reshape(y,(y.shape[0],1)), X_new)), testSet




