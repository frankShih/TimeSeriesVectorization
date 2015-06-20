# -*- coding: utf-8 -*-

import numpy
import csv

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation

def logiReg_tune(train_data, test_data, fold=5):
	from sklearn.linear_model import LogisticRegression	
	C_range = [pow(2,i) for i in range(-5,5)]
	result = numpy.zeros((len(C_range),3))

	#for i in range(10):	# Random permutation
	# -------------------- tuning ----------------------
	temp = numpy.zeros((len(C_range),3))
	avg_temp = numpy.zeros((1,3))
	best_valid_AUC = 0
	for c in C_range:
		avg_temp = numpy.zeros((1,3))			
		# Do cross-validation
		skf = cross_validation.StratifiedKFold(train_data[:, 0], n_folds=fold)        
		clf = LogisticRegression(C=c, penalty='l1')
		for trainV_index, testV_index in skf:			
			
			clf.fit(train_data[trainV_index, 1:], train_data[trainV_index, 0])
			#return estimates for all classes (ordered by the label of classes)
			train_pred = clf.predict_proba(train_data[trainV_index, 1:])
			valid_pred = clf.predict_proba(train_data[testV_index, 1:])	# clf.score(x,y)
			# the contest using ROC/AUC for evaluation 

			train_auc = roc_auc_score(train_data[trainV_index, 0], train_pred[:,1])
			valid_auc = roc_auc_score(train_data[testV_index, 0], valid_pred[:,1])
			avg_temp = avg_temp + numpy.array([train_auc, valid_auc, c])
		
		#print avg_temp/float(fold)
		temp[numpy.where(numpy.asarray(C_range)==c),:] = avg_temp/float(fold)
	
	result = result+temp	
	#result =result/10
	#print '----------------result--------------------'
	#print result
	bestC = C_range[numpy.argmax(result[:,1])]
	print 'bestC = '+ str(bestC) + ' best_result = '+ str(max(result[:,1]))
	#raw_input('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
	
	# ----------------- prediction ---------------
	clf = LogisticRegression(C=bestC)
	prediction = numpy.zeros((test_data.shape[0],1))
	# print prediction.shape	
	for trainV_index, testV_index in skf:		
		clf.fit(train_data[trainV_index, 1:], train_data[trainV_index, 0])		
		prediction = prediction + numpy.reshape((clf.predict_proba(test_data))[:,1],(200,1))
		
	#raw_input("Press Enter to continue...")
	return prediction/fold
	


def svm_tune(train_data, test_data, fold=5, mode=1):
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



def forest_tune(train_data, test_data, fold=5):	
	from sklearn.ensemble import RandomForestClassifier	
	import numpy
	
	clf=RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=15, \
		max_features='sqrt', n_jobs=-1, min_samples_leaf=5)      
	clf.fit(train_data[:, 1:], train_data[:, 0])
	prediction = clf.predict_proba(test_data)
	#print prediction.shape
	#raw_input("Press Enter to continue...")
	return prediction[:,1]


def gradientBoost_tune(train_data, test_data, fold=5):
	from sklearn.ensemble import GradientBoostingClassifier	
	import numpy
	
	clf=GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=3, \
		max_features='sqrt',  min_samples_leaf=5)      
	clf.fit(train_data[:, 1:], train_data[:, 0])
	prediction = clf.predict_proba(test_data)
	#print prediction.shape
	#raw_input("Press Enter to continue...")
	return prediction[:,1]



def local_eval(predict_result, driverOrder):
	from sklearn.metrics import mean_absolute_error
	import csv, numpy
	path = '/home/shih/GitHubs/TimeSeriesVectorization/probability_normalize.csv'
	f = open(path, 'r')
	score = []  
	for row in csv.DictReader(f):  
		score.append(float(row['prob']))
	f.close()  
	score = numpy.asarray(score)
	print [(score[(driverOrder-1)*200:driverOrder*200]).shape, predict_result.shape]
	Err = mean_absolute_error(score[(driverOrder-1)*200:driverOrder*200], predict_result)
	#print [predict_result.shape]	

	return Err


def overSample():
	pass


'''
if __name__ == '__main__':
	target = numpy.genfromtxt('/home/shih/GitHubs/TimeSeriesVectorization/vectorization/result.csv', 
							delimiter=',', skip_header=False)
	others = numpy.genfromtxt('/home/shih/GitHubs/TimeSeriesVectorization/vectorization/test_result.csv', 
							delimiter=',', skip_header=False)
	print([type(target)])
	print logiReg_tune(target,others)
'''	

