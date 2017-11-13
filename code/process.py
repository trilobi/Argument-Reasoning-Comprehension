#-*-coding:utf-8-*-
import pandas as pd 
import numpy as np
import pickle
import re
from keras.preprocessing.text import Tokenizer

def process(sen):
	sen = re.sub("'re", " are", sen)
	sen = re.sub("n's", " is not", sen)
	sen = re.sub("'s", "", sen)
	sen = re.sub("can't", "can not", sen)
	sen = re.sub("n't", " not", sen)
	sen = re.sub("'t", "", sen)
	sen = re.sub("n'd", " not", sen)
	sen = re.sub("'d", "", sen)
	sen = re.sub("[^A-Za-z0-9 ]", "", sen)
	return sen

EMBEDDING_DIM = 300
#--------------------------------Source data----------------------------------------------
print('load source sentences............')
train_df = pd.read_csv('../../snli/input/snli_train.csv')
dev_df = pd.read_csv('../../snli/input/snli_dev.csv')

print('Indexing word vectors.....')
word_dic = {}
for line in open('glove.840B.300d.txt','r'):
    temp = line.strip().split(' ')
    word_dic[temp[0]] = np.array(map(float,temp[1:]))

t_train_sen1 = [i for i in train_df['sentence1']]
t_train_sen2 = [i for i in train_df['sentence2']]
t_dev_sen1 = [i for i in dev_df['sentence1']]
t_dev_sen2 = [i for i in dev_df['sentence2']]

t_s_train_labels = [i for i in train_df['labels']]
t_s_dev_labels = [i for i in dev_df['labels']]
t_s_train_labels = np.array(map(int, t_s_train_labels))
t_s_dev_labels = np.array(map(int, t_s_dev_labels))

train_sen1 = []
train_sen2 = []
s_train_labels = []
for i in range(len(t_s_train_labels)):
	if t_s_train_labels[i] != 1:
		train_sen1.append(t_train_sen1[i])
		train_sen2.append(t_train_sen2[i])
		s_train_labels.append(t_s_train_labels[i])

dev_sen1 = []
dev_sen2 = []
s_dev_labels = []
for i in range(len(t_s_dev_labels)):
	if t_s_dev_labels[i] != 1:
		dev_sen1.append(t_dev_sen1[i])
		dev_sen2.append(t_dev_sen2[i])
		s_dev_labels.append(t_s_dev_labels[i])

for i in range(len(s_train_labels)):
	if s_train_labels[i] == 2:
		s_train_labels[i] = 1

for i in range(len(s_dev_labels)):
	if s_dev_labels[i] == 2:
		s_dev_labels[i] = 1

print len(train_sen1)
print len(train_sen2)
print len(s_train_labels)
print len(dev_sen1)
print len(dev_sen2)
print len(s_dev_labels)
print train_sen1[:10]
print train_sen2[:10]
print s_train_labels[:10]
print dev_sen1[:10]
print dev_sen2[:10]
print s_dev_labels[:10]

#----------------------------------Target data--------------------------------------------------
print('load target sentences.............')
train_read = open("../data/row/train-full.txt")
dev_read = open("../data/row/dev-full.txt")
train_read.readline()
dev_read.readline()
train_warrant0 = []
train_warrant1 = []
train_labels = []
train_reason = []
train_claim = []
train_title = []
train_info = []
dev_warrant0 = []
dev_warrant1 = []
dev_labels = []
dev_reason = []
dev_claim = []
dev_title = []
dev_info = []
for line in train_read.readlines():
	items = line.strip().split("\t")
	if len(items) < 8:
		continue
	train_warrant0.append(process(items[1]))
	train_warrant1.append(process(items[2]))
	train_labels.append(int(items[3]))
	train_reason.append(process(items[4]))
	train_claim.append(process(items[5]))
	train_title.append(process(items[6]))
	train_info.append(process(items[7]))
for line in dev_read.readlines():
	items = line.strip().split("\t")
	if len(items) < 8:
		continue
	dev_warrant0.append(process(items[1]))
	dev_warrant1.append(process(items[2]))
	dev_labels.append(int(items[3]))
	dev_reason.append(process(items[4]))
	dev_claim.append(process(items[5]))
	dev_title.append(process(items[6]))
	dev_info.append(process(items[7]))
#--------------------------Coding---------------------------------------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_sen1 + train_sen2 + dev_sen1 + dev_sen2 +train_warrant0 + train_warrant1 + train_reason + train_claim + train_title + train_info + dev_warrant0 + dev_warrant1 + dev_reason + dev_claim + dev_title + dev_info)
word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))


nb_words = len(word_index) + 1
embedding_matrix = np.random.rand(nb_words, EMBEDDING_DIM)
for word, i in word_index.items():
    if word_dic.has_key(word):
        embedding_matrix[i] = word_dic[word]
pickle.dump(embedding_matrix, open('vector.pkl', 'wb'))

train_sen1 = tokenizer.texts_to_sequences(train_sen1)
train_sen2 = tokenizer.texts_to_sequences(train_sen2)
dev_sen1 = tokenizer.texts_to_sequences(dev_sen1)
dev_sen2 = tokenizer.texts_to_sequences(dev_sen2)
train_warrant0 = tokenizer.texts_to_sequences(train_warrant0)
train_warrant1 = tokenizer.texts_to_sequences(train_warrant1)
train_reason = tokenizer.texts_to_sequences(train_reason)
train_claim = tokenizer.texts_to_sequences(train_claim)
train_title = tokenizer.texts_to_sequences(train_title)
train_info = tokenizer.texts_to_sequences(train_info)
dev_warrant0 = tokenizer.texts_to_sequences(dev_warrant0)
dev_warrant1 = tokenizer.texts_to_sequences(dev_warrant1)
dev_reason = tokenizer.texts_to_sequences(dev_reason)
dev_claim = tokenizer.texts_to_sequences(dev_claim)
dev_title = tokenizer.texts_to_sequences(dev_title)
dev_info = tokenizer.texts_to_sequences(dev_info)

pickle.dump([s_train_labels, train_sen1, train_sen2], open('../data/source/train.pkl', 'wb'))
pickle.dump([s_dev_labels, dev_sen1, dev_sen2], open('../data/source/dev.pkl', 'wb'))
pickle.dump([train_labels, train_warrant0, train_warrant1, train_reason, train_claim, train_title, train_info], open('../data/coding/train.pkl', 'wb'))
pickle.dump([dev_labels, dev_warrant0, dev_warrant1, dev_reason, dev_claim, dev_title, dev_info], open('../data/coding/dev.pkl', 'wb'))

