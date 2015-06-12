# -*- coding: utf-8 -*-
def find_repeat(label,pretiction):	#a numpy array with probabilities
	import csv, os, re, numpy
	path1 = u'/media/shih/新增磁碟區/dataSets/drivers'
	print path1
	#path1 = unicode('/media/shih/新增磁碟區/dataSets/drivers', 'utf8')
	path2 = u'/media/shih/新增磁碟區/ZiWen_packup/drivers/drivers'
	#path2 = unicode('/media/shih/新增磁碟區/ZiWen_packup/drivers/drivers', 'utf8')
	target1 = list(set(os.listdir(path1)).intersection(set(os.listdir(path2))))
	target2 = list(set(os.listdir(path2)).difference(set(os.listdir(path1))))
	file = '/home/shih/GitHubs/TimeSeriesVectorization/vectorization/logi_40-60_repeat.csv'
	f = open(file, "w")
	f.write("driver_trip,prob\n")
	f.close()
	file_list = os.listdir(path2)
	for sub_dir in target1:
		f = open(os.path.join(path1,sub_dir)+'/tripo.csv', 'r')
		print sub_dir
		for row in csv.DictReader(f):
			if row['value']=='[]':
				continue
			else:
				print [row['key'], row['value']]
				temp =  numpy.asarray(re.findall('\d+',row['value']))
				temp = numpy.hstack((temp, row['key'].split('.')[0]))
				#print temp
				number = []
				for i in range(len(temp)):

					number = int(temp[i])
					index = file_list.index(sub_dir)*200+number
					pretiction[index]=0
	

	for sub_dir in target2:
		f = open(os.path.join(path2,sub_dir)+'/tripo.csv', 'r')
		print sub_dir
		for row in csv.DictReader(f):
			if row['value']=='[]':
				continue
			else:
				print [row['key'], row['value']]
				temp =  numpy.asarray(re.findall('\d+',row['value']))
				temp = numpy.hstack((temp, row['key'].split('.')[0]))
				#print temp
				number = []
				for i in range(len(temp)):

					number = int(temp[i])
					index = file_list.index(sub_dir)*200+number
					pretiction[index]=0
	




	f = open(file, "a")
	for i in range(len(label)):
		f.write(label[i]+','+str(pretiction[i])+'\n')
	f.close()



'''
#print temp.shape
file = '/home/shih/GitHubs/TimeSeriesVectorization/vectorization/logi_40-60_01.csv'
f = open(file, "w")
f.write("driver_trip,prob\n")
f.close()

print [numpy.min(temp), numpy.max(temp), numpy.std(temp), numpy.mean(temp),numpy.median(temp)]

print [numpy.mean(temp)-numpy.std(temp)*2, numpy.mean(temp)+numpy.std(temp)*2]

print [numpy.mean(temp)-numpy.std(temp), numpy.mean(temp)+numpy.std(temp)]

threshold = numpy.mean(temp)+numpy.std(temp)*1

f = open(file, "a")
for i in range(temp.shape[0]):
	if temp[i] > threshold:
		temp[i] = 1
	else:
		temp[i] = 0

	f.write(index[i]+','+str(temp[i])+'\n')
	
f.close()
#numpy.savetxt('/home/shih/GitHubs/TimeSeriesVectorization/vectorization/logi_40-60_01.csv',(numpy.asarray(temp)),delimiter=',',fmt='%i')
'''





import csv, numpy

f = open('/home/shih/GitHubs/TimeSeriesVectorization/vectorization/submission_logi_40-60.csv', 'r')
temp = [];
index = [];
for row in csv.DictReader(f):
	index.append(row['driver_trip'])
	temp.append(row['prob'])
f.close();
temp = numpy.array(temp,dtype=float)


find_repeat(index, temp)



