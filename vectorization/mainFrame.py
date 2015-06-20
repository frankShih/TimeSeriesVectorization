# -*- coding: utf-8 -*-

import numpy, os, random
import matplotlib.pyplot as plt
import training, dataTransformation, featSelet
from junkDetect import findJunk
def speedDistribution(trip): #input as numpy aray
  import numpy
  # transform to velocity
  #print [trip[:,0], trip[:,1] ]
  velocity = numpy.round(numpy.sqrt(numpy.diff(trip[:,0])**2 + numpy.diff(trip[:,1])**2),2)
  # create 20 quantile values of each trip
  #print velocity
  #print numpy.percentile(velocity, range(5,101,5))
  #raw_input("Press Enter to continue...")
  return numpy.percentile(velocity, range(5,101,5))


path = '/home/shih/dataSets/no_rotate_flip'
file = "submission_RF_20quan_uselessRemoval.csv"
f = open(file, "w")
f.write("driver_trip,prob\n")
f.close()

temp=0
drivers = os.listdir(path)  
counter = 0
for driver in sorted(drivers):
  counter = counter+1
  refData = []  
  currentData = []
  
  #------- train every driver -------
  target = 1
  print '==================driver - '+str(driver)+'===================='
  for i in range(200):
    #print 'trip - '+str(i+1)
    trip = numpy.genfromtxt(os.path.join(path, str(driver), str(i+1)+ ".csv"), 
                            delimiter=',', skip_header=True)          
    
    #currentData.append(dataTransformation.featEx(trip,5))
    currentData.append(dataTransformation.quanFeat(trip))
    #print dataTransformation.featEx(trip,11)
    
  currentData = numpy.asarray(currentData)
  
  #-------- random select 5 other drivers --------
  '''
  target = 0

  sample = [int(i) for i in drivers]
  sample.remove(int(driver))  
  randomDrivers = [random.choice(sample) for i in range(0,1)]  
  for rd in randomDrivers:  #create a negative set with label=0
    for i in range(200):
      trip = numpy.genfromtxt(os.path.join(path, str(rd), str(i+1)+ ".csv"), 
                            delimiter=',', skip_header=True)      
      #features = numpy.insert(speedDistribution(trip), 0, target) # add label
      refData.append(speedDistribution(trip))
    
  refData = numpy.asarray(refData)
  '''

  sample = [int(i) for i in drivers]  
  sample.remove(int(driver))
  
  rand_driver = [random.choice(sample) for i in range(0,190)]       
      
  for rd in rand_driver:   
    # Load 200 trajectory for random drivers    
    j = random.randint(1,190)
    #print os.path.join(path, str(rd), str(j)+'.csv')
    #print 'random driver-  '+str(rd)+' ; '+'trip - '+str(j)
    trip = numpy.genfromtxt(os.path.join(path, str(rd), str(j)+'.csv'), 
                            delimiter=',', skip_header=True)
    #refData.append(dataTransformation.featEx(trip,5))
    refData.append(dataTransformation.quanFeat(trip))
    #print  len(refData)
    #raw_input('---------------------------------------------------------')
  refData = numpy.asarray(refData)


  # ------------tuning & prediction ------------
  trainData = findJunk(int(driver), currentData, refData, mode=1)
  print [currentData.shape, refData.shape]
  #trainSet, testSet = featSelet.tree_select(trainData, currentData)
  #print [trainSet.shape, testSet.shape]

  
  #prediction = training.logiReg_tune(trainSet, testSet)
  prediction = training.forest_tune(trainData, currentData, fold=5)
  '''
  temp = temp + training.local_eval(prediction,counter)

  if counter==10:
    print '============local evaluation result============='
    print temp/counter
    break
'''
  f = open(file, "a")
  #print prediction.shape
  for j in range(prediction.shape[0]):
    #print [j, prediction[j]]
    f.write(driver+'_'+str(j+1)+','+str(prediction[j])+'\n')
  f.close()




