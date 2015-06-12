# -*- coding: utf-8 -*-
import numpy 
import csv, os, math, random
import bow, dataTransformation, bow_inter
import spams


def sample_test(driverNum, sampleNum, w_len = 40, k = 500):
	path = '/media/shih/新增磁碟區/ZiWen_packup/drivers/drivers'
	bow_sp = bow.BoW_sp()

	folder = os.listdir(path)
	sample = [int(i) for i in folder]
	#print sample
	sample.remove(driverNum)
	
	#print [sample, type(sample)]
	rand_driver = [random.choice(sample) for i in range(0,sampleNum)]
	#print [sample, type(sample)]
	#rand_driver = [folder[i] for i in sample ]
	#print rand_driver
	    
	a = 1.0/math.sqrt(w_len)
	trajectory = list()
	    #Iteratively process drivers data
	for i,driver in enumerate(rand_driver):		
		# Load 200 trajectory for random drivers
		
		j = random.randint(1,200)
		#print os.path.join(path, str(driver), str(j)+'.csv')
		temp = numpy.genfromtxt(os.path.join(path, str(driver), str(j)+'.csv'), 
		                        delimiter=',', skip_header=True)
		
		#print j
		
		temp = dataTransformation.trip_diff(temp)
		trajectory.append(numpy.asarray(temp))
	#print [len(trajectory)]
	trajectory = bow_sp.segment(trajectory,w_len=40)	
	print [len(trajectory), numpy.shape(numpy.hstack(trajectory))]
	D = spams.trainDL(numpy.asfortranarray(numpy.hstack(trajectory)), 
                          K=k, lambda1=a, posAlpha=True, iter=-3)
	#print [len(trajectory), len(trajectory[0])]

	sparseCode = bow_sp.coding_series(trajectory, D, a=a, iter=-3)
	#numpy.savetxt('test_result.csv', sparseCode, delimiter=',')
	#sparseCode = sparseCode/ numpy.linalg.norm(sparseCode)	
	# no need to normalize, trajectory always different length

	#raw_input("Press Enter to continue...")    
	return sparseCode

