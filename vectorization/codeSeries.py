# -*- coding: utf-8 -*-
import numpy 
import csv, os, math
import bow, dataTransformation, training
import spams
import testSet


path = '/media/shih/新增磁碟區/ZiWen_packup/drivers/drivers'
file = "submission_logi_40-60.csv"
f = open(file, "w")
f.write("driver_trip,prob\n")
f.close()
bow_sp = bow.BoW_sp()

folder = os.listdir(path)
w_len = 40
k = 500
a = 1.0/math.sqrt(w_len)

    # Iteratively process drivers' data
for i,driver in enumerate(folder):
	print '---------------|'+str(driver)+'|-------------------'
		
	#testSample = testSet.sample_test(int(driver),200)
	# Load 200 trajectory for each drivers
	trajectory = list()
	for j in range(1, 201):
		temp = numpy.genfromtxt(os.path.join(path, str(driver), str(j)+'.csv'), 
		                        delimiter=',', skip_header=True)
		#print j
		temp = dataTransformation.trip_diff(temp)
		#print numpy.shape(numpy.asarray(temp))
		trajectory.append(numpy.asarray(temp))
	trajectory = bow_sp.segment(trajectory,w_len=40)	
	#print [len(trajectory), numpy.shape(numpy.hstack(trajectory))]
	D = spams.trainDL(numpy.asfortranarray(numpy.hstack(trajectory)), 
                          K=k, lambda1=a, posAlpha=True, iter=-3)
	#print [len(trajectory), len(trajectory[0])]

	driverCode = bow_sp.coding_series(trajectory, D, a=a, iter=-3)
	#numpy.savetxt('result.csv', driverCode, delimiter=',')
	#driverCode = driverCode/ numpy.linalg.norm(driverCode)	
	# no need to normalize, trajectory always different length

	#raw_input("Press Enter to continue...")    

	# ---------------- training phase ----------------
	prediction = training.logiReg_tune(driverCode, testSample)
	f = open(file, "a")
	print prediction.shape
	for j in range(prediction.shape[0]):
		#print [j, prediction[j]]
		f.write(driver+'_'+str(j+1)+','+str(prediction[j])+'\n')
	f.close()




