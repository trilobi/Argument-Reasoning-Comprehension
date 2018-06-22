import os.path
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier

if __name__ == '__main__':
	filepath = './result/'
	pathDir =  os.listdir(filepath)
	pred_labels = []
	dev_file = open('../../data/row/dev-full.txt')
	instance_id = []
	labels = []
	dev_file.readline()
	for line in dev_file.readlines():
		items = line.strip().split('\t')
		instance_id.append(items[0])
		labels.append(int(items[3]))
	for allDir in pathDir:
		vec = []
		child_filepath = filepath + allDir
		print child_filepath 
		file_read = open(child_filepath)
		for line in file_read.readlines():
			items = line.strip().split(" ")
			vec.append(int(items[1]))
		vec = np.array(vec)
		pred_labels.append(vec)
	pred_labels = np.array(pred_labels)
	pred_labels = pred_labels.transpose((1,0))
	p_labels = []
	for vec in pred_labels:
		if np.sum(vec)>=3:
			p_labels.append(1)
		else:
			p_labels.append(0)
	acc = accuracy_score(p_labels, labels)
	print acc
			
