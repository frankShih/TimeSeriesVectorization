# -*- coding: utf-8 -*-

import numpy
import csv

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation

def logiReg_tune(driver, other, fold=5):
	from sklearn.linear_model import LogisticRegression	
	C_range = [pow(2,i) for i in range(-10,10)]
	result = numpy.zeros((len(C_range),3))

	for i in range(10):	# Random permutation
		driver_temp = driver[numpy.random.permutation(driver.shape[0]), :]
		other_temp = other[numpy.random.permutation(other.shape[0]), :]
		
		train_data = numpy.vstack(
			(numpy.hstack((numpy.ones((driver_temp.shape[0],1)), driver_temp)), 
			 numpy.hstack((numpy.zeros((other_temp.shape[0],1)), other_temp))))
		
		
		temp = numpy.zeros((len(C_range),3))
		avg_temp = numpy.zeros((1,3))
		best_valid_AUC = 0
		for c in C_range:
			avg_temp = numpy.zeros((1,3))			
			# Do cross-validation
			skf = cross_validation.StratifiedKFold(train_data[:, 0], n_folds=fold)        
			clf = LogisticRegression(C=c)
			for trainV_index, testV_index in skf:
				#print [trainV_index, testV_index]
				#raw_input("Press Enter to continue...")
				clf.fit(train_data[trainV_index, 1:], train_data[trainV_index, 0])
				#return estimates for all classes (ordered by the label of classes)
				train_pred = clf.predict_proba(train_data[trainV_index, 1:])
				valid_pred = clf.predict_proba(train_data[testV_index, 1:])	# clf.score(x,y)
				# the contest using ROC/AUC for evaluation 

				train_auc = roc_auc_score(train_data[trainV_index, 0], train_pred[:, 1])
				valid_auc = roc_auc_score(train_data[testV_index, 0], valid_pred[:, 1])
				avg_temp = avg_temp + numpy.array([train_auc, valid_auc, c])
			# If mean accuracy greater than best accuracy, then record it			
			#print avg_temp/float(fold)
			temp[numpy.where(numpy.asarray(C_range)==c),:] = avg_temp/float(fold)
		#print [C,c,numpy.where(numpy.asarray(C)==c)]
		result = result+temp	
	result =result/10
	#print '----------------result--------------------'
	#print result
	bestC = C_range[numpy.argmax(result[:,1])]
	print 'bestC = '+ str(bestC)
	clf = LogisticRegression(C=bestC)
	clf.fit(train_data[:, 1:], train_data[:, 0])
	prediction = clf.predict_proba(driver)
	
	#raw_input("Press Enter to continue...")
	return prediction[:,1]
	


def svm_tune(driver, other, fold=5, mode=1):
	from sklearn.svm import LinearSVC
	from sklearn.svm import LinearSVR
	C_range = [pow(2,i) for i in range(-8,8)]
	result = numpy.zeros((len(C_range),3))

	for i in range(10):	# Random permutation
		driver_temp = driver[numpy.random.permutation(driver.shape[0]), :]
		other_temp = other[numpy.random.permutation(other.shape[0]), :]
		
		train_data = numpy.vstack(
			(numpy.hstack((numpy.ones((driver_temp.shape[0],1)), driver_temp)), 
			 numpy.hstack((numpy.zeros((other_temp.shape[0],1)), other_temp))))
		
		
		temp = numpy.zeros((len(C_range),3))
		avg_temp = numpy.zeros((1,3))
		best_valid_AUC = 0
		for c in C_range:
			avg_temp = numpy.zeros((1,3))			
			# Do cross-validation
			skf = cross_validation.StratifiedKFold(train_data[:, 0], n_folds=fold)
			if mode:        
				clf = LinearSVC(C=c)
			else:
				clf = LinearSVR(C=c, epsilon=.1)
			for trainV_index, testV_index in skf:
				
				clf.fit(train_data[trainV_index, 1:], train_data[trainV_index, 0])
				#return estimates for all classes (ordered by the label of classes)
				train_pred = clf.predict(train_data[trainV_index, 1:])
				valid_pred = clf.predict(train_data[testV_index, 1:])	# clf.score(x,y)
				# the contest using ROC/AUC for evaluation 
				train_auc = roc_auc_score(train_data[trainV_index, 0], train_pred)
				valid_auc = roc_auc_score(train_data[testV_index, 0], valid_pred)
				avg_temp = avg_temp + numpy.array([train_auc, valid_auc, c])
					
			print avg_temp/float(fold)
			temp[numpy.where(numpy.asarray(C_range)==c),:] = avg_temp/float(fold)
		#print [C,c,numpy.where(numpy.asarray(C)==c)]
		result = result+temp	
	result =result/10
	print '----------------result--------------------'
	print result
	bestC = C_range[numpy.argmax(result[:,1])]
	print 'bestC = '+ str(bestC)
	if mode:        
		clf = LinearSVC(C=bestC)
	else:
		clf = LinearSVR(C=bestC, epsilon=.1)
	clf.fit(train_data[:, 1:], train_data[:, 0])
	prediction = clf.predict(driver)
	
	raw_input("Press Enter to continue...")
	return prediction



'''
if __name__ == '__main__':
	target = numpy.genfromtxt('/home/shih/GitHubs/TimeSeriesVectorization/vectorization/result.csv', 
							delimiter=',', skip_header=False)
	others = numpy.genfromtxt('/home/shih/GitHubs/TimeSeriesVectorization/vectorization/test_result.csv', 
							delimiter=',', skip_header=False)
	print([type(target)])
	print logiReg_tune(target,others)
'''	

