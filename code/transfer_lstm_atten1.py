# -*- coding: utf-8 -*-
import tensorflow as tf
from keras.preprocessing import sequence
from keras.utils import np_utils
import numpy as np
import data_helper
import pickle
import random as rd
from sklearn.metrics import accuracy_score

#--------------------------------------Model Begin--------------------------------------------------------
class Model(object):
#获取最后一个时刻的输出
	def get_last(self, outputs, seq_len):
		batch_size = tf.shape(outputs)[0]
		max_seq_len = outputs.get_shape().as_list()[1]
		dim = outputs.get_shape().as_list()[2]
		index_list = tf.range(batch_size) * max_seq_len + (seq_len - 1)
		reshape_outputs = tf.reshape(outputs, (-1, dim))
		last_outputs = tf.gather(reshape_outputs , index_list)
		return last_outputs
#向量乘法
	def batch_matmul(self, x, y):
		batch_size = tf.shape(x)[0]
		dim = tf.shape(x)[2]
		return tf.reshape(tf.matmul(tf.reshape(x, (-1,dim)), y), (batch_size, -1, dim))
		
	def __init__(self, max_len, word_vector, lstm_num_units, n_class):
		#LSTM输入输出的dropout
		self.input_keep_prob = tf.placeholder(tf.float32, name='rnn_input_keep_prob')
		self.output_keep_prob = tf.placeholder(tf.float32, name='rnn_output_keep_prob')
		#----------------Source data------------------
		#源数据的数据存放器
		self.sen1_ids = tf.placeholder(tf.int32, [None, max_len], name='sen2_ids')
		self.sen2_ids = tf.placeholder(tf.int32, [None, max_len], name='sen1_ids')
		self.sen1_len = tf.placeholder(tf.int32, [None], name='sen1_len')
		self.sen2_len = tf.placeholder(tf.int32, [None], name='sen2_len')
		self.s_labels = tf.placeholder(tf.int32, [None], name='s_labels')
		#----------------Target data------------------
		#目标任务的数据存放器
		self.warrant0 = tf.placeholder(tf.int32, [None, max_len], name='warrant0')
		self.warrant1 = tf.placeholder(tf.int32, [None, max_len], name='warrant1')
		self.reason = tf.placeholder(tf.int32, [None, max_len], name='reason')
		self.claim = tf.placeholder(tf.int32, [None, max_len], name='claim')
		self.labels = tf.placeholder(tf.float32, [None, n_class], name='labels')
		self.warrant0_len = tf.placeholder(tf.int32, [None], name='warrant0_len')
		self.warrant1_len = tf.placeholder(tf.int32, [None], name='warrant1_len')
		self.reason_len = tf.placeholder(tf.int32, [None], name='reason_len')
		self.claim_len = tf.placeholder(tf.int32, [None], name='claim')

		#------------------Embedding layer---------------
		#获取词向量
		W = tf.Variable(word_vector, dtype=tf.float32, trainable=True, name='W')
		sen1_embedded = tf.nn.embedding_lookup(W, self.sen1_ids, name='sen1_embedded')
		sen2_embedded = tf.nn.embedding_lookup(W, self.sen2_ids, name='sen2_embedded')
		warrant0_embedded = tf.nn.embedding_lookup(W, self.warrant0, name='warrant0_embedded')
		warrant1_embedded = tf.nn.embedding_lookup(W, self.warrant1, name='warrant1_embedded')
		reason_embedded = tf.nn.embedding_lookup(W, self.reason, name='reason_embedded')
		claim_embedded = tf.nn.embedding_lookup(W, self.claim, name='claim_embedded')

		#-----------------------LSTM Coding-----------------------
		def lstm_layer(sen_embedding, sen_len ):
			cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(lstm_num_units, state_is_tuple=True),
																 input_keep_prob=self.input_keep_prob,
																 output_keep_prob=self.output_keep_prob)	
			outputs, state = tf.nn.dynamic_rnn(cell, sen_embedding, sequence_length=sen_len, dtype=tf.float32)
			return	outputs
		
		#-----------------------------Attention---------------------------
		#文本蕴含的静态注意力机制
		def attention(outputs, R_ave, max_len, lstm_num_unit):
			Wh = tf.Variable(tf.truncated_normal([lstm_num_units, lstm_num_units], stddev=0.1), name='Wh')
			Wr = tf.Variable(tf.truncated_normal([lstm_num_units, lstm_num_units], stddev=0.1), name='Wr')
			M = tf.nn.tanh(tf.add(self.batch_matmul(outputs, Wh),
				tf.tile(tf.expand_dims(tf.matmul(R_ave, Wr), 1), (1, max_len, 1))),name='M')
			w = tf.Variable(tf.truncated_normal([lstm_num_units], stddev=0.1), name='w')
			alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(M, w), axis=2), name='alpha')
			R_att = tf.squeeze(tf.matmul(tf.expand_dims(alpha, 1), outputs), squeeze_dims=[1], name='R_att')
			Wp = tf.Variable(tf.truncated_normal([lstm_num_units, lstm_num_units], stddev=0.1), name='Wp')
			Wx = tf.Variable(tf.truncated_normal([lstm_num_units, lstm_num_units], stddev=0.1), name='Wx')
			atten_lstm_output = tf.nn.tanh(tf.matmul(R_att, Wp) + tf.matmul(R_ave, Wx), name='atten_lstm_output')
			return R_att

		#--------------------------Building--------------------------------
		#源数据使用LSTM编码
		with tf.variable_scope('lstm1_layer'):
			sen1_outputs = lstm_layer(sen1_embedded, self.sen1_len)
			#sen1_last_outputs = self.get_last(sen1_outputs, self.sen1_len )
		with tf.variable_scope('lstm2_layer'):
			sen2_outputs = lstm_layer(sen2_embedded, self.sen2_len)
			#sen2_last_outputs = self.get_last(sen2_outputs, self.sen2_len)
	    #目标数据使用LSTM编码
		with tf.variable_scope('lstm1_layer', reuse=True):
			reason_outputs = lstm_layer(reason_embedded, self.reason_len)
		with tf.variable_scope('lstm1_layer', reuse=True):
			claim_outputs = lstm_layer(claim_embedded, self.claim_len)
		with tf.variable_scope('lstm2_layer', reuse=True):
			warrant0_outputs = lstm_layer(warrant0_embedded, self.warrant0_len)
		with tf.variable_scope('lstm2_layer', reuse=True):
			warrant1_outputs = lstm_layer(warrant1_embedded, self.warrant1_len)
		#源数据两个句子做池化处理
		sen1 = tf.reduce_mean(sen1_outputs, 1)
		sen2 = tf.reduce_mean(sen2_outputs, 1)
		
		#Seq2Seq的注意力机制
		#claim_meanpooling = tf.reduce_mean(claim_outputs, 1)
		claim_maxpooling = tf.reduce_max(claim_outputs, 1)
		warrant0 = tf.reduce_max(warrant0_outputs, 1)
		warrant1 = tf.reduce_max(warrant1_outputs, 1)
		alpha = tf.nn.softmax(tf.transpose(tf.matmul(reason_outputs,
								tf.expand_dims(claim_maxpooling, -1)), [0,2,1]))
		context = tf.reduce_mean(tf.matmul(alpha, reason_outputs), 1)
		#claim_meanpooling = tf.reduce_mean(claim_outputs, 1)
		#claim_maxpooling = tf.reduce_max(claim_outputs, 1)
		#context = attention(reason_outputs, claim_meanpooling, max_len, lstm_num_units)
		
		#全连接层
		def add_layer(inputs, in_size, out_size, activation_function=None):
			W = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
			b = tf.Variable(tf.constant(0.1, shape = [out_size]))
			out = tf.nn.xw_plus_b(inputs, W, b)
			if activation_function is None:
				outputs = out
			else:
				outputs = activation_function(out)
			return outputs
		
		#双线性矩阵
		W_m = tf.Variable(tf.truncated_normal([lstm_num_units, lstm_num_units], stddev=0.1), name='W_m')
		#------------------------------Source--------------------------------
		源任务做双线性计算
		self.sim_snli = tf.diag_part(tf.matmul(tf.matmul(sen1, W_m), tf.transpose(sen2, [1,0])))
		self.s_prob = tf.nn.sigmoid(self.sim_snli)
		#MSE损失函数
		self.s_loss = tf.reduce_mean(tf.square(self.s_prob - tf.to_float(self.s_labels, name='ToFloat')))
		self.s_prob_labels = tf.to_int32(self.s_prob + 0.5, name='ToInt32')
		self.s_acc = tf.reduce_mean(tf.cast(tf.equal(self.s_prob_labels, self.s_labels), tf.float32))
		
		#------------------------------Target--------------------------------
		#目标任务做双线性计算
		self.sim_warrant0 = tf.diag_part(tf.matmul(tf.matmul(context, W_m), tf.transpose(warrant0, [1,0])))
		self.sim_warrant1 = tf.diag_part(tf.matmul(tf.matmul(context, W_m), tf.transpose(warrant1, [1,0])))
		self.prob = tf.nn.softmax(tf.stack([self.sim_warrant0, self.sim_warrant1], axis=1))
		self.prob_labels = tf.argmax(self.prob, 1)
		#MSE损失函数
		self.loss = tf.reduce_mean(tf.square(self.prob - self.labels))
		self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob, 1), tf.argmax(self.labels, 1)), tf.float32))
#---------------------------------------Model End---------------------------------------------------------

if __name__ == '__main__':
	
	max_len = 25
	lstm_num_units = 256
	input_keep_prob = 0.9
	output_keep_prob = 0.9
	num_checkpoints = 1	
	checkpoints_dir = './model/'
	
	#----------Source model--------------
	#模型参数设置
	n_class = 2
	s_batch_size = 1024
	s_num_epochs = 10
	s_evaluate_every = 512
	s_max_num_undesc = 10
	
	t_batch_size = 64
	t_num_epochs = 30
	t_evaluate_every = 32
	t_max_num_undesc = 10	
	
	#-----------------------Source data--------------------------
	#获取源任务编码数据
	[s_train_labels, sen1_train, sen2_train] = pickle.load(open("../../data/source/train_snli.pkl"))
	[s_dev_labels, sen1_dev, sen2_dev] = pickle.load(open("../../data/source/dev_snli.pkl"))
	sen1_train_len = np.array([min(max_len, len(s)) for s in sen1_train])
	sen2_train_len = np.array([min(max_len, len(s)) for s in sen2_train])
	sen1_dev_len = np.array([min(max_len, len(s)) for s in sen1_dev])
	sen2_dev_len = np.array([min(max_len, len(s)) for s in sen2_dev])
	
	#源任务数据做定长处理
	sen1_train = sequence.pad_sequences(sen1_train, maxlen=max_len, truncating='post', padding='post')
	sen2_train = sequence.pad_sequences(sen2_train, maxlen=max_len, truncating='post', padding='post')
	sen1_dev = sequence.pad_sequences(sen1_dev, maxlen=max_len, truncating='post', padding='post')
	sen2_dev =  sequence.pad_sequences(sen2_dev, maxlen=max_len, truncating='post', padding='post')
	
	#分类编码
	#s_train_labels = np_utils.to_categorical(s_train_labels, n_class)
	#s_dev_labels = np_utils.to_categorical(s_dev_labels, n_class)
		
	#-----------------------Target data--------------------------
	print 'load target data.......'
	gold = []
	for line in open('../../data/row/gold_data.txt','r'):
		items = line.strip().split('\t')
		gold.append(int(items[1]))
	gold = np.array(gold)
	[train_labels, train_id, train_warrant0, train_warrant1, train_reason, train_claim, train_title, train_info] = pickle.load(open("../../data/coding/train_snli.pkl"))
	[dev_labels, dev_id, dev_warrant0, dev_warrant1, dev_reason, dev_claim, dev_title, dev_info] = pickle.load(open("../../data/coding/dev_snli.pkl"))	
	[test_id, test_warrant0, test_warrant1, test_reason, test_claim, test_title, test_info] = pickle.load(open("../../data/coding/test_snli.pkl"))
	
	#计算句子真实长度
	train_warrant0_len = np.array([min(max_len, len(s)) for s in train_warrant0])
	train_warrant1_len = np.array([min(max_len, len(s)) for s in train_warrant1])
	train_reason_len = np.array([min(max_len, len(s)) for s in train_reason])
	train_claim_len = np.array([min(max_len, len(s)) for s in train_claim])
	dev_warrant0_len = np.array([min(max_len, len(s)) for s in dev_warrant0])
	dev_warrant1_len = np.array([min(max_len, len(s)) for s in dev_warrant1])
	dev_reason_len = np.array([min(max_len, len(s)) for s in dev_reason])
	dev_claim_len = np.array([min(max_len, len(s)) for s in dev_claim])
	test_warrant0_len = np.array([min(max_len, len(s)) for s in test_warrant0])
	test_warrant1_len = np.array([min(max_len, len(s)) for s in test_warrant1])
	test_reason_len = np.array([min(max_len, len(s)) for s in test_reason])
	test_claim_len = np.array([min(max_len, len(s)) for s in test_claim])
	
	#目标任务句子做定长处理
	train_warrant0 = sequence.pad_sequences(train_warrant0, maxlen=max_len, truncating='post', padding='post')	
	train_warrant1 = sequence.pad_sequences(train_warrant1, maxlen=max_len, truncating='post', padding='post')
	train_reason = sequence.pad_sequences(train_reason, maxlen=max_len, truncating='post', padding='post')
	train_claim = sequence.pad_sequences(train_claim, maxlen=max_len, truncating='post', padding='post')
	dev_warrant0 = sequence.pad_sequences(dev_warrant0, maxlen=max_len, truncating='post', padding='post')
	dev_warrant1 = sequence.pad_sequences(dev_warrant1, maxlen=max_len, truncating='post', padding='post')
	dev_reason = sequence.pad_sequences(dev_reason, maxlen=max_len, truncating='post', padding='post')
	dev_claim = sequence.pad_sequences(dev_claim, maxlen=max_len, truncating='post', padding='post')
	test_warrant0 = sequence.pad_sequences(test_warrant0, maxlen=max_len, truncating='post', padding='post')
	test_warrant1 = sequence.pad_sequences(test_warrant1, maxlen=max_len, truncating='post', padding='post')
	test_reason = sequence.pad_sequences(test_reason, maxlen=max_len, truncating='post', padding='post')
	test_claim = sequence.pad_sequences(test_claim, maxlen=max_len, truncating='post', padding='post')

	#分类编号
	train_labels = np_utils.to_categorical(train_labels, n_class)
	dev_labels = np_utils.to_categorical(dev_labels, n_class)
	print 'load vector........'
	vector = pickle.load(open("../../W2V/vector_snli.pkl"))

	print 'running............'
	model = Model(max_len, vector, lstm_num_units, n_class)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	'''
	#-------------------------------Source Training------------------------------------
	#源任务模型训练
	with tf.Session(config=config) as sess:
		global_step = tf.Variable(0, trainable = False, name='global_step')
		s_train_op = tf.train.AdamOptimizer(0.001).minimize(model.s_loss, global_step = global_step)
		saver = tf.train.Saver(max_to_keep=num_checkpoints)
		sess.run(tf.global_variables_initializer())
								
		def s_train_step(sen1_ids, sen2_ids, sen1_len, sen2_len, s_labels):
			feed_dict={
				model.sen1_ids:sen1_ids,
				model.sen2_ids:sen2_ids,
				model.sen1_len:sen1_len,
				model.sen2_len:sen2_len,
				model.s_labels:s_labels,
				model.input_keep_prob:input_keep_prob,
				model.output_keep_prob:output_keep_prob,
				}
			_, step, loss, acc = sess.run([s_train_op, global_step, model.s_loss, model.s_acc], feed_dict)
			return step, loss, acc

		def s_dev_step(sen1_ids, sen2_ids, sen1_len, sen2_len, s_labels):
			feed_dict={
				model.sen1_ids:sen1_ids,
				model.sen2_ids:sen2_ids,
				model.sen1_len:sen1_len,
				model.sen2_len:sen2_len,
				model.s_labels:s_labels,
				model.input_keep_prob:1.0,
				model.output_keep_prob:1.0,
				}
			step, loss, acc = sess.run([global_step, model.s_loss, model.s_acc], feed_dict)
			return loss, acc
	
		print 'Start Source training........'
		batches = data_helper.iter_batch(s_batch_size,
										 s_num_epochs,
										 sen1_train,
										 sen2_train,
										 sen1_train_len,
										 sen2_train_len,
										 s_train_labels)
		s_train_loss_list = []
		s_train_acc_list = []
		s_max_dev_acc = float('-inf')
		s_min_dev_loss = 0
		s_num_undesc = 0
		for current_epoch, batch in batches:
			if s_num_undesc > s_max_num_undesc:
				break
			sen1_ids, sen2_ids, sen1_len, sen2_len, s_labels = batch
			step, s_train_batch_loss, s_train_batch_acc = s_train_step(sen1_ids, sen2_ids, sen1_len, sen2_len, s_labels)
			current_step = tf.train.global_step(sess, global_step)
			s_train_loss_list.append(s_train_batch_loss)
			s_train_acc_list.append(s_train_batch_acc)
			if current_step % s_evaluate_every == 0:
				s_train_loss = np.mean(s_train_loss_list)
				s_train_acc = np.mean(s_train_acc_list)
				print 'train loss: %g, train acc: %g' %(s_train_loss, s_train_acc)
				s_train_loss_list = []
				s_train_acc_list = []
				s_dev_loss_list = []
				s_dev_acc_list = []
				for _, dev_batch in data_helper.iter_batch(s_batch_size, 1, sen1_dev, sen2_dev, sen1_dev_len, sen2_dev_len, s_dev_labels):
					sen1_ids, sen2_ids, sen1_len, sen2_len, s_labels = dev_batch
					s_dev_batch_loss, s_dev_batch_acc = s_dev_step(sen1_ids, sen2_ids, sen1_len, sen2_len, s_labels)
					s_dev_loss_list.append(s_dev_batch_loss)
					s_dev_acc_list.append(s_dev_batch_acc)
				s_dev_loss = np.mean(s_dev_loss_list)
				s_dev_acc = np.mean(s_dev_acc_list)
				#保存源任务模型
				saver.save(sess, checkpoints_dir+'trans_atten_10/Snli_model.ckpt')
				print 'Dev loss: %g, Dev acc: %g'%(s_dev_loss, s_dev_acc)
	'''	
	print 'load transfer model.............'
	with tf.Session() as sess:
		global_step = tf.Variable(0, trainable = False, name='global_step')
		train_op = tf.train.AdamOptimizer(0.001).minimize(model.loss, global_step = global_step)
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(max_to_keep=num_checkpoints)
		#加载目标任务模型
		saver.restore(sess, checkpoints_dir+'trans_atten_100/Snli_model.ckpt')

		def train_step(warrant0, warrant1, reason, claim, labels, warrant0_len, warrant1_len, reason_len, claim_len):
			feed_dict={
				model.warrant0:warrant0,
				model.warrant1:warrant1,
				model.reason:reason,
				model.claim:claim,
				model.labels:labels,
				model.warrant0_len:warrant0_len,
				model.warrant1_len:warrant1_len,
				model.reason_len:reason_len,
				model.claim_len:claim_len,
				model.input_keep_prob:input_keep_prob,
				model.output_keep_prob:output_keep_prob,
					}
			_, step, loss, acc = sess.run([train_op, global_step, model.loss, model.acc], feed_dict)
			return step, loss, acc
			
		def dev_step(warrant0, warrant1, reason, claim, labels, warrant0_len, warrant1_len, reason_len, claim_len):
			feed_dict={
				model.warrant0:warrant0,
				model.warrant1:warrant1,
				model.reason:reason,
				model.claim:claim,
				model.labels:labels,
				model.warrant0_len:warrant0_len,
				model.warrant1_len:warrant1_len,
				model.reason_len:reason_len,
				model.claim_len:claim_len,
				model.input_keep_prob:1.0,
				model.output_keep_prob:1.0,
					}
			loss, acc = sess.run([model.loss, model.acc], feed_dict)
			return loss, acc

		def test_step(warrant0, warrant1, reason, claim, warrant0_len, warrant1_len, reason_len, claim_len):
			feed_dict={
				model.warrant0:warrant0,
				model.warrant1:warrant1,
				model.reason:reason,
				model.claim:claim,
				model.warrant0_len:warrant0_len,
				model.warrant1_len:warrant1_len,
				model.reason_len:reason_len,
				model.claim_len:claim_len,
				model.input_keep_prob:1.0,
				model.output_keep_prob:1.0,
					}
			prob_labels = sess.run(model.prob_labels, feed_dict)
			return prob_labels

		print 'Start Target training........'
		batches = data_helper.iter_batch(t_batch_size,
										 t_num_epochs,
										 train_warrant0,
										 train_warrant1,
										 train_reason,
										 train_claim,
										 train_warrant0_len,
										 train_warrant1_len,
										 train_reason_len,
										 train_claim_len,
										 train_labels)
		train_loss_list = []
		train_acc_list = []
		max_dev_acc = float('-inf')
		min_dev_loss = 0
		t_num_undesc = 0
		for current_epoch, batch in batches:
			if t_num_undesc > t_max_num_undesc:
				break
			warrant0, warrant1, reason, claim, warrant0_len, warrant1_len, reason_len, claim_len, labels = batch
			step, train_batch_loss, train_batch_acc = train_step(warrant0, warrant1, reason, claim, labels, warrant0_len, warrant1_len, reason_len, claim_len)
			current_step = tf.train.global_step(sess, global_step)
			train_loss_list.append(train_batch_loss)
			train_acc_list.append(train_batch_acc)
			if current_step % t_evaluate_every == 0:
				train_loss = np.mean(train_loss_list)
				train_acc = np.mean(train_acc_list)
				print 'train loss: %g, train acc: %g' %(train_loss, train_acc)
				train_loss_list = []
				train_acc_list = []
				dev_loss, dev_acc = dev_step(dev_warrant0, dev_warrant1, dev_reason, dev_claim, dev_labels, dev_warrant0_len, dev_warrant1_len, dev_reason_len, dev_claim_len)
				print 'Dev loss: %g, Dev acc: %g'%(dev_loss, dev_acc)
				prob_labels = test_step(test_warrant0, test_warrant1, test_reason, test_claim, test_warrant0_len, test_warrant1_len, test_reason_len, test_claim_len)
				test_acc = accuracy_score(prob_labels, gold)
				print '-----Test ACC : %g-----' %(test_acc)
	#'''
	'''
				if dev_acc >= max_dev_acc:
					max_dev_acc = dev_acc
					min_dev_loss = dev_loss
					prob_labels = test_step(test_warrant0, test_warrant1, test_reason, test_claim, test_warrant0_len, test_warrant1_len, test_reason_len, test_claim_len)
					test_acc = accuracy_score(prob_labels, gold)
					print '-----Test ACC : %g-----' %(test_acc)
					t_num_undesc = 0
				else:
					t_num_undesc += 1
				print 'Dev loss: %g, Dev acc: %g'%(dev_loss, dev_acc)
		print 'MinDev loss: %g, MinDev acc: %g'%(min_dev_loss, max_dev_acc)
	'''
