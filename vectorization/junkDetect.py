# -*- coding: utf-8 -*-
import dataTransformation
import os, numpy, time
'''
path = '/media/shih/新增磁碟區/ZiWen_packup/drivers/drivers'
file = "useless_trip.csv"
f = open(file, "w")
f.write("driver_Num,trip_Num\n")
drivers = os.listdir(path)  
f.close()

# human walk speeed: 5km/hr ~~ 1.4 m/sec
for driver in drivers:
	for i in range(200):
		#time.sleep(0.1)
		trip = numpy.genfromtxt(os.path.join(path, str(driver), str(i+1)+ ".csv"), 
	                        delimiter=',', skip_header=True)  
		
		if (max(abs(trip[:,0]))< 15) and (max(abs(trip[:,1]))< 15):	# too short distance

			print [driver, i+1, 111]
			f = open(file, "a")
			f.write(driver+','+str(i+1)+'\n')
			f.close()
			continue 

		temp = dataTransformation.trip_diff(trip)
		if max(temp) < 3.0:		#too low speed
			print [driver, i, 222]
			f = open(file, "a")
			f.write(driver+','+str(i+1)+'\n')
			f.close()
			continue

		if temp.shape[0] < 150:	# too short time
			print [driver, i, 333]
			f = open(file, "a")
			f.write(driver+','+str(i+1)+'\n')
			f.close()
			continue	
'''
	

def findJunk(driver_Num, driver, other,  mode=1):
	#remove useless trips & return clean dataset
	path = '/home/shih/GitHubs/TimeSeriesVectorization/vectorization/useless_trip.csv'
	f = open(path, 'r')	
	junk = numpy.genfromtxt(path, delimiter=',', skip_header=True)      
	
	if mode == 0:	# doing nothing	(for testSet)	
		
		train_data = numpy.vstack(
		(numpy.hstack((numpy.ones((driver.shape[0],1)), driver)), 
		 numpy.hstack((numpy.zeros((other.shape[0],1)), other))))		

	elif mode ==1:	#conservative, training without those trip
		
		target = numpy.int_(junk[numpy.nonzero(junk[:,0]==driver_Num),1])
		#target = [i-1 for i in target ]
		print target-1
		driver = numpy.delete(driver,target-1,0)
		
		train_data = numpy.vstack(
		(numpy.hstack((numpy.ones((driver.shape[0],1)), driver)), 
		 numpy.hstack((numpy.zeros((other.shape[0],1)), other))))

	else:	# aggresive, set useless trip to 0, then train it
		
		target = numpy.int_(junk[numpy.nonzero(junk[:,0]==driver_Num),1])
		#print target
		driver = numpy.hstack((numpy.ones((driver.shape[0],1)), driver))		
		for i in [target]:
			#print i
			driver[i-1,0] = 0
		train_data = numpy.vstack(
					(driver, numpy.hstack((numpy.zeros((other.shape[0],1)), other))))		

	train_data = train_data[numpy.random.permutation(train_data.shape[0]), :]
	#print train_data.shape
	return train_data