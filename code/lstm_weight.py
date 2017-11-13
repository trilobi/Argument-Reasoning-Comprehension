import tensorflow as tf
from keras.preprocessing import sequence
from keras.utils import np_utils
import numpy as np
import data_helper
import pickle
from sklearn.metrics import accuracy_score

def concate(claim, title, info):
	context = []
	for i in range(len(claim)):
		vec = []
		vec.extend(claim[i])
		vec.extend(title[i])
		vec.extend(info[i])
		vec = np.array(vec)
		context.append(vec)
	context = np.array(context)
	return context

#--------------------------------------Model Begin--------------------------------------------------------
class Model(object):
	def get_last(self, outputs, seq_len):
		batch_size = tf.shape(outputs)[0]
		max_seq_len = outputs.get_shape().as_list()[1]
		dim = outputs.get_shape().as_list()[2]
		index_list = tf.range(batch_size) * max_seq_len + (seq_len - 1)
		reshape_outputs = tf.reshape(outputs, (-1, dim))
		last_outputs = tf.gather(reshape_outputs , index_list)
		return last_outputs
		
	def __init__(self, max_len, word_vector, gru_num_units, n_class):
		self.warrant0 = tf.placeholder(tf.int32, [None, max_len], name='warrant0')
		self.warrant1 = tf.placeholder(tf.int32, [None, max_len], name='warrant1')
		self.reason = tf.placeholder(tf.int32, [None, max_len], name='reason')
		self.claim = tf.placeholder(tf.int32, [None, max_len], name='claim')
		self.labels = tf.placeholder(tf.float32, [None, n_class], name='labels')
		self.warrant0_len = tf.placeholder(tf.int32, [None], name='warrant0_len')
		self.warrant1_len = tf.placeholder(tf.int32, [None], name='warrant1_len')
		self.reason_len = tf.placeholder(tf.int32, [None], name='reason_len')
		self.claim_len = tf.placeholder(tf.int32, [None], name='claim')
		self.input_keep_prob = tf.placeholder(tf.float32, name='rnn_input_keep_prob')
		self.output_keep_prob = tf.placeholder(tf.float32, name='rnn_output_keep_prob')
		#Embedding layer
		W = tf.Variable(word_vector, dtype=tf.float32, trainable=False, name='W')
		warrant0_embedded = tf.nn.embedding_lookup(W, self.warrant0, name='warrant0_embedded')
		warrant1_embedded = tf.nn.embedding_lookup(W, self.warrant1, name='warrant1_embedded')
		reason_embedded = tf.nn.embedding_lookup(W, self.reason, name='reason_embedded')
		claim_embedded = tf.nn.embedding_lookup(W, self.claim, name='claim_embedded')
		
		with tf.name_scope('reason_claim_attention'):
			with tf.variable_scope('reason'):
				#reason_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(gru_num_units), 
				#										input_keep_prob=self.input_keep_prob, 
				#										output_keep_prob=self.output_keep_prob)
				reason_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(gru_num_units, state_is_tuple=True),
																 input_keep_prob=self.input_keep_prob,
																 output_keep_prob=self.output_keep_prob)
				reason_outputs, reason_state = tf.nn.dynamic_rnn(reason_cell, reason_embedded, 
											sequence_length=self.reason_len, dtype=tf.float32)
			with tf.variable_scope('claim'):
				#claim_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(gru_num_units), 
				#										input_keep_prob=self.input_keep_prob, 
				#										output_keep_prob=self.output_keep_prob)
				claim_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(gru_num_units, state_is_tuple=True),
															 input_keep_prob=self.input_keep_prob,
															 output_keep_prob=self.output_keep_prob)
				claim_outputs, claim_state = tf.nn.dynamic_rnn(claim_cell, claim_embedded, 
											sequence_length=self.claim_len, dtype=tf.float32)
				claim_last_outputs = self.get_last(claim_outputs, self.claim_len)
			self.alpha = tf.nn.softmax(tf.transpose(tf.matmul(reason_outputs, tf.expand_dims(claim_last_outputs, -1)), [0,2,1]))
			self.context = tf.reduce_mean(tf.matmul(self.alpha, reason_outputs), 1)

		def GRU_Layer(warrant_embedded, warrant_len):
			#warrant_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(gru_num_units), 
			#												input_keep_prob=self.input_keep_prob, 
			#												output_keep_prob=self.output_keep_prob)
			warrant_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(gru_num_units, state_is_tuple=True), 
															input_keep_prob=self.input_keep_prob,
															output_keep_prob=self.output_keep_prob)
			warrant_output, reason_state = tf.nn.dynamic_rnn(warrant_cell, warrant_embedded,
												sequence_length=warrant_len, dtype=tf.float32)
			return self.get_last(warrant_output, warrant_len)
		with tf.variable_scope('warrant'):
			warrant0 = GRU_Layer(warrant0_embedded, self.warrant0_len)
		with tf.variable_scope('warrant', reuse=True):
			warrant1 = GRU_Layer(warrant1_embedded, self.warrant1_len)
		W_m = tf.Variable(tf.truncated_normal([gru_num_units, gru_num_units], stddev=0.1), name='W_m') 
		self.sim_warrant0 = tf.diag_part(tf.matmul(tf.matmul(self.context, W_m), tf.transpose(warrant0, [1,0])))
		self.sim_warrant1 = tf.diag_part(tf.matmul(tf.matmul(self.context, W_m), tf.transpose(warrant1, [1,0])))
		self.prob = tf.nn.softmax(tf.stack([self.sim_warrant0, self.sim_warrant1], axis=1))
		self.prob_labels = tf.argmax(self.prob, 1)
		self.loss = tf.reduce_mean(tf.square(self.prob - self.labels))
		#self.loss = tf.reduce_mean(-tf.reduce_sum(self.labels * tf.log(self.prob), reduction_indices=[1]))
		self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob, 1),tf.argmax(self.labels, 1)), tf.float32))
		
#---------------------------------------Model End---------------------------------------------------------
if __name__ == '__main__':
	#model
	info_max_len = 45
	max_len = 25
	gru_num_units = 300
	n_class = 2
	input_keep_prob = 0.6
	output_keep_prob = 0.6
	
	#train
	batch_size = 256
	num_epochs = 20
	evaluate_every = 8
	num_checkpoints = 3
	max_num_undsc = 10
	checkpoints_dir = 'model/'

	print 'load data.......'
	[train_warrant0, train_warrant1, train_labels, train_reason, train_claim, train_title, train_info] = pickle.load(open("../../data/coding/train.pkl"))
	[dev_warrant0, dev_warrant1, dev_labels, dev_reason, dev_claim, dev_title, dev_info] = pickle.load(open("../../data/coding/dev.pkl"))	
	#train_context = concate(train_claim, train_title, train_info)
	#dev_context = concate(dev_claim, dev_title, dev_info)

	train_warrant0_len = np.array([min(max_len, len(s)) for s in train_warrant0])
	train_warrant1_len = np.array([min(max_len, len(s)) for s in train_warrant1])
	train_reason_len = np.array([min(max_len, len(s)) for s in train_reason])
	train_claim_len = np.array([min(max_len, len(s)) for s in train_claim])
	#train_title_len = np.array([len(s) for s in train_title])
	#train_info_len = np.array([len(s) for s in train_info])	
	dev_warrant0_len = np.array([min(max_len, len(s)) for s in dev_warrant0])
	dev_warrant1_len = np.array([min(max_len, len(s)) for s in dev_warrant1])
	dev_reason_len = np.array([min(max_len, len(s)) for s in dev_reason])
	dev_claim_len = np.array([min(max_len, len(s)) for s in dev_claim])
	#dev_title_len = np.array([len(s) for s in dev_title])
	#dev_info_len = np.array([len(s) for s in dev_info])
	#train_title_len = np.array([train_title_len[i]+train_claim_len[i] for i in range(len(train_title_len))])
	#train_info_len = np.array([train_title_len[i]+train_info_len[i] for i in range(len(train_info_len))])
	#dev_title_len = np.array([dev_title_len[i]+dev_claim_len[i] for i in range(len(dev_title_len))])
	#dev_info_len = np.array([dev_title_len[i]+dev_info_len[i] for i in range(len(dev_info_len))])
	
	train_warrant0 = sequence.pad_sequences(train_warrant0, maxlen=max_len, truncating='post', padding='post')	
	train_warrant1 = sequence.pad_sequences(train_warrant1, maxlen=max_len, truncating='post', padding='post')
	train_reason = sequence.pad_sequences(train_reason, maxlen=max_len, truncating='post', padding='post')
	train_claim = sequence.pad_sequences(train_claim, maxlen=max_len, truncating='post', padding='post')
	#train_context = sequence.pad_sequences(train_context, maxlen=max_len, truncating='post', padding='post') 
	dev_warrant0 = sequence.pad_sequences(dev_warrant0, maxlen=max_len, truncating='post', padding='post')
	dev_warrant1 = sequence.pad_sequences(dev_warrant1, maxlen=max_len, truncating='post', padding='post')
	dev_reason = sequence.pad_sequences(dev_reason, maxlen=max_len, truncating='post', padding='post')
	dev_claim = sequence.pad_sequences(dev_claim, maxlen=max_len, truncating='post', padding='post')
	#dev_context = sequence.pad_sequences(dev_context, maxlen=max_len, truncating='post', padding='post') 

	print 'load vector........'
	[vector, dic] = pickle.load(open("../../W2V/word_dic.pkl"))
	train_labels = np_utils.to_categorical(train_labels, n_class)
	dev_labels = np_utils.to_categorical(dev_labels, n_class)
	
	print 'running............'
	model = Model(max_len, vector, gru_num_units, n_class)
	with tf.Session() as sess:
		global_step = tf.Variable(0, trainable = False, name='global_step')
		train_op = tf.train.AdamOptimizer(0.001).minimize(model.loss, global_step = global_step)
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)
		sess.run(tf.global_variables_initializer())

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
			_, step, acc = sess.run([train_op, global_step, model.acc], feed_dict)	
			return step, acc
					
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
				model.input_keep_prob:input_keep_prob,
				model.output_keep_prob:output_keep_prob,
				}
			step, acc = sess.run([global_step, model.acc], feed_dict)	
			return acc

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
				model.input_keep_prob:input_keep_prob,
				model.output_keep_prob:output_keep_prob,
				}
			prob = sess.run([model.prob_labels], feed_dict)	
			return prob
		
		print 'start train..........'
		batches =  data_helper.iter_batch(batch_size, num_epochs, train_warrant0, train_warrant1, train_reason, train_claim, train_labels, train_warrant0_len, train_warrant1_len, train_reason_len, train_claim_len)
		train_acc_list = []
		max_dev_acc = float('-inf')
		num_undesc = 0
		for current_epoch , batch in batches:
			if num_undesc > max_num_undsc:
				break
			warrant0, warrant1, reason, claim, labels, warrant0_len, warrant1_len, reason_len, claim_len = batch
			step, train_acc = train_step(warrant0, warrant1, reason, claim, labels, warrant0_len, warrant1_len, reason_len, claim_len)
			current_step = tf.train.global_step(sess, global_step)
			train_acc_list.append(train_acc)
			if current_step % evaluate_every == 0:
				train_acc = np.mean(train_acc_list)
				print "train acc : %g" %(train_acc)
				train_acc_list = []
				dev_acc_list = []
				dev_batches =  data_helper.iter_batch(batch_size, 1, dev_warrant0, dev_warrant1, dev_reason, dev_claim, dev_labels, dev_warrant0_len, dev_warrant1_len, dev_reason_len, dev_claim_len)
				for _, dev_batch in dev_batches:
					warrant0, warrant1, reason, claim, labels, warrant0_len, warrant1_len, reason_len, claim_len = dev_batch
					dev_acc = dev_step(warrant0, warrant1, reason, claim, labels, warrant0_len, warrant1_len, reason_len, claim_len)
					dev_acc_list.append(dev_acc)
				dev_acc = np.mean(dev_acc_list)
				if dev_acc > max_dev_acc:
					saver.save(sess, checkpoints_dir+'model', global_step=current_step)
					max_dev_acc = dev_acc
					num_undesc = 0
				else:
					num_undesc += 1
				print "DEV ACC : %g" %(dev_acc)
		print "MinDevAcc: %g" %(max_dev_acc)
		'''	
		ckpt = tf.train.get_checkpoint_state(checkpoints_dir)
		if len(ckpt.model_checkpoint_path):
			saver.restore(sess, ckpt.model_checkpoint_path)
	
			test_batches = data_helper.iter_batch(len(dev_warrant0), 1, dev_warrant0, dev_warrant1, dev_reason, dev_claim, dev_warrant0_len, dev_warrant1_len, dev_reason_len, dev_claim_len)
			_, test_batch = test_batches.next()
			warrant0, warrant1, reason, claim, warrant0_len, warrant1_len, reason_len, claim_len = test_batch
			pred = test_step(warrant0, warrant1, reason, claim, warrant0_len, warrant1_len, reason_len, claim_len)
			pickle.dump(pred, open('../error/rnn_pred.pkl','wb'))	
		else:
			print "the checkpoint direction is error, please check it "
		'''
