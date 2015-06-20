# -*- coding: utf-8 -*-
import numpy 
import csv, os, math, random
import bow, dataTransformation, training, bow_inter
import spams
import testSet, junkDetect


path = '/home/shih/dataSets/no_rotate_flip'
#path = '/home/shih/dataSets/no_rotate_flip'
file = "sparseTesting.csv"
f = open(file, "w")
f.write("driver_trip,prob\n")
f.close()
bow_sp = bow.BoW_sp()

folder = os.listdir(path)
w_len = 40
k = 500
a = 1.0/math.sqrt(w_len)

import itertools
for m,n in itertools.product(range(20,51,5),range(100,1001,100)):
	counter = 0
	result = 0
    # Iteratively process drivers' data
	for driver in sorted(folder):
		print '---------------|'+str(driver)+'|-------------------'		
		# Load 200 trajectory for each drivers
		trajectory = list()
		for j in range(1, 201):
			temp = numpy.genfromtxt(os.path.join(path, str(driver), str(j)+'.csv'), 
			                        delimiter=',', skip_header=True)						
			trajectory.append(temp)
		trajectory = bow_inter.diff(trajectory)	
		trajectory_seg = bow_inter.slice(trajectory,w_len=m)
		
		D = spams.trainDL(numpy.asfortranarray(trajectory_seg), 
	                          K=n, lambda1=a, posAlpha=True, iter=-3)
		#print [len(trajectory), len(trajectory[0])]

		driverCode = bow_inter.bow(trajectory, D, a=a, w_len=m)
		#numpy.savetxt('result.csv', driverCode, delimiter=',')
		#driverCode = driverCode/ numpy.linalg.norm(driverCode)	
		# no need to normalize, trajectory always different length


		# ======================== sample negative data ==========================
		sample = [int(i) for i in folder]		
		sample.remove(int(driver))
		# Load 200 trajectory for random drivers
		rand_driver = [random.choice(sample) for i in range(0,200)]
		    		
		trajectory = list()
		    #Iteratively process drivers data
		for i,driver in enumerate(rand_driver):		
					
			j = random.randint(1,200)
			#print os.path.join(path, str(driver), str(j)+'.csv')
			temp = numpy.genfromtxt(os.path.join(path, str(driver), str(j)+'.csv'), 
			                        delimiter=',', skip_header=True)				
			#temp = dataTransformation.trip_diff(temp)
			trajectory.append(temp)
		
		trajectory = bow_inter.diff(trajectory)	
		trajectory_seg = bow_inter.slice(trajectory,w_len=m)	
		
		D = spams.trainDL(numpy.asfortranarray(trajectory_seg), 
	                          K=n, lambda1=a, posAlpha=True, iter=-3)
		#print [len(trajectory), len(trajectory[0])]
		otherCode = bow_inter.bow(trajectory, D, a=a, w_len=m)		
		
		# ---------------- training phase ----------------
		counter = counter+1
		trainData = junkDetect.findJunk(int(driver), driverCode, otherCode, mode=1)
		prediction = training.logiReg_tune(trainData, otherCode)
		result = result + training.local_eval(prediction,counter)

		if counter==10:
			print '============local evaluation result============='
			print result/counter
			break


	f = open(file, "a")
	#	print prediction.shape
	#	for j in range(prediction.shape[0]):
			#print [j, prediction[j]]
	f.write(str(m)+'_'+str(n)+','+str(result/counter)+'\n')
	f.close()
	



