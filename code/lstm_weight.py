import tensorflow as tf
from keras.preprocessing import sequence
from keras.utils import np_utils
import numpy as np
import data_helper
import pickle
from sklearn.metrics import accuracy_score

#--------------------------------------Model Begin--------------------------------------------------------
class Model(object):
	
	def batch_matmul(self, x, y):
		batch_size = tf.shape(x)[0]
		dim = tf.shape(x)[2]
		return tf.reshape(tf.matmul(tf.reshape(x, (-1,dim)), y), (batch_size, -1, dim))

	def get_last(self, outputs, seq_len):
		batch_size = tf.shape(outputs)[0]
		max_seq_len = outputs.get_shape().as_list()[1]
		dim = outputs.get_shape().as_list()[2]
		index_list = tf.range(batch_size) * max_seq_len + (seq_len - 1)
		reshape_outputs = tf.reshape(outputs, (-1, dim))
		last_outputs = tf.gather(reshape_outputs , index_list)
		return last_outputs
		
	def __init__(self, max_len, word_vector, lstm_num_units, n_class):
		
		self.input_keep_prob = tf.placeholder(tf.float32, name='rnn_input_keep_prob')
		self.output_keep_prob = tf.placeholder(tf.float32, name='rnn_output_keep_prob')
		
		#----------------Target data------------------
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
		W = tf.Variable(word_vector, dtype=tf.float32, trainable=True, name='W')
		warrant0_embedded = tf.nn.embedding_lookup(W, self.warrant0, name='warrant0_embedded')
		warrant1_embedded = tf.nn.embedding_lookup(W, self.warrant1, name='warrant1_embedded')
		reason_embedded = tf.nn.embedding_lookup(W, self.reason, name='reason_embedded')
		claim_embedded = tf.nn.embedding_lookup(W, self.claim, name='claim_embedded')
		
		#-----------------------LSTM Coding-----------------------
		def lstm_layer(sen_embedding, sen_len ):
				cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(lstm_num_units, state_is_tuple=True),
																 input_keep_prob=self.input_keep_prob,
																 output_keep_prob=self.output_keep_prob)	
				#cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(lstm_num_units),
				#												 input_keep_prob=self.input_keep_prob,
				#												 output_keep_prob=self.output_keep_prob)
				outputs, state = tf.nn.dynamic_rnn(cell,
												   sen_embedding, 
												   sequence_length=sen_len,
												   dtype=tf.float32)
				return	outputs
		
		def bilstm_layer(sen_embedding, sen_len):
			cell_fw = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(lstm_num_units, state_is_tuple=True),
																		input_keep_prob=self.input_keep_prob,
																		output_keep_prob=self.output_keep_prob)
			cell_bw = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(lstm_num_units, state_is_tuple=True),
																		input_keep_prob=self.input_keep_prob,
																		output_keep_prob=self.output_keep_prob) 
			outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, sen_embedding, 
															sequence_length=sen_len, dtype=tf.float32)
			sen_outputs = tf.concat(outputs,2)
			return sen_outputs
		
		def attention(outputs, R_ave, max_len, lstm_num_unit):
			Wh = tf.Variable(tf.truncated_normal([lstm_num_units, lstm_num_units], stddev=0.1),
							name='Wh')
			Wr = tf.Variable(tf.truncated_normal([lstm_num_units, lstm_num_units], stddev=0.1),
							name='Wr')
			M = tf.nn.tanh(tf.add(self.batch_matmul(outputs, Wh),
							tf.tile(tf.expand_dims(tf.matmul(R_ave, Wr), 1), (1, max_len, 1))),
							name='M')
			w = tf.Variable(tf.truncated_normal([lstm_num_units], stddev=0.1),
							name='w')
			alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(M, w), axis=2),
							name='alpha')
			R_att = tf.squeeze(tf.matmul(tf.expand_dims(alpha, 1), outputs), squeeze_dims=[1],
							name='R_att')
			Wp = tf.Variable(tf.truncated_normal([lstm_num_units, lstm_num_units], stddev=0.1),
							name='Wp')
			Wx = tf.Variable(tf.truncated_normal([lstm_num_units, lstm_num_units], stddev=0.1),
							name='Wx')
			atten_lstm_output = tf.nn.tanh(tf.matmul(R_att, Wp) + tf.matmul(R_ave, Wx),
							name='atten_lstm_output')
			return atten_lstm_output
						
		with tf.variable_scope('lstm1_layer'):
			reason_outputs = lstm_layer(reason_embedded, self.reason_len)
		with tf.variable_scope('lstm1_layer', reuse=True):
			claim_outputs = lstm_layer(claim_embedded, self.claim_len)
			claim_last_outputs = self.get_last(claim_outputs, self.claim_len)
		
		claim_meanpooling = tf.reduce_mean(claim_outputs, 1)
		claim_maxpooling = tf.reduce_max(claim_outputs, 1)
		alpha = tf.nn.softmax(tf.transpose(tf.matmul(reason_outputs, 
										   tf.expand_dims(claim_maxpooling, -1)), [0,2,1]))
		#context = tf.reduce_mean(tf.matmul(alpha, reason_outputs), 1)
		
		context = attention(reason_outputs, claim_maxpooling, max_len, lstm_num_units)

		with tf.variable_scope('lstm2_layer'):
			warrant0_outputs = lstm_layer(warrant0_embedded, self.warrant0_len)
			warrant0 = tf.reduce_max(warrant0_outputs, 1)
			#warrant0 = self.get_last(warrant0_outputs, self.warrant0_len)
		with tf.variable_scope('lstm2_layer', reuse=True):
			warrant1_outputs = lstm_layer(warrant1_embedded, self.warrant1_len)
			warrant1 = tf.reduce_max(warrant1_outputs, 1)
			#warrant1 = self.get_last(warrant1_outputs, self.warrant1_len)
		
		W_m = tf.Variable(tf.truncated_normal([lstm_num_units, lstm_num_units], stddev=0.1), name='W_m') 
		self.sim_warrant0 = tf.diag_part(tf.matmul(tf.matmul(context, W_m), tf.transpose(warrant0, [1,0])))
		self.sim_warrant1 = tf.diag_part(tf.matmul(tf.matmul(context, W_m), tf.transpose(warrant1, [1,0])))
		self.prob = tf.nn.softmax(tf.stack([self.sim_warrant0, self.sim_warrant1], axis=1))
		self.prob_labels = tf.argmax(self.prob, 1)
		#sort loss
		#self.loss = tf.reduce_mean(tf.maximum(tf.cast(tf.tile([0], [tf.shape(self.labels)[0]]), dtype=tf.float32),
		#							1.0 - tf.reduce_sum(tf.multiply(self.labels, self.prob), 1) 
		#							+ tf.reduce_sum(tf.multiply(1.0-self.labels, self.prob), 1)))
		#Log Loss
		#self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.prob))
		#MSE
		self.loss = tf.reduce_mean(tf.square(self.prob - self.labels))
		self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob, 1), tf.argmax(self.labels, 1)), tf.float32))
		
#---------------------------------------Model End---------------------------------------------------------
if __name__ == '__main__':
	
	max_len = 25
	lstm_num_units = 256
	input_keep_prob = 1
	output_keep_prob = 1
	num_checkpoints = 1	
	checkpoints_dir = './model/weight/'	
	
	#---------Target model-----------
	n_class = 2
	batch_size = 64
	num_epochs = 20
	evaluate_every = 40
	max_num_undsc = 5
	
	#-----------------------Target data--------------------------
	print 'load target data.......'
	gold = []
	for line in open('../../data/row/gold_data.txt','r'):
		items = line.strip().split('\t')
		gold.append(int(items[1]))
	[train_labels, train_id, train_warrant0, train_warrant1, train_reason, train_claim, train_title, train_info] = pickle.load(open("../../data/coding/train.pkl"))
	[dev_labels, dev_id, dev_warrant0, dev_warrant1, dev_reason, dev_claim, dev_title, dev_info] = pickle.load(open("../../data/coding/dev.pkl"))	
	[test_id, test_warrant0, test_warrant1, test_reason, test_claim, test_title, test_info] = pickle.load(open("../../data/coding/test.pkl"))	
	
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
	
	train_labels = np_utils.to_categorical(train_labels, n_class)
	dev_labels = np_utils.to_categorical(dev_labels, n_class)

	print 'load vector........'
	vector = pickle.load(open("../../W2V/vector.pkl"))

	print 'running............'
	model = Model(max_len, vector, lstm_num_units, n_class)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	#saver = tf.train.Saver(max_to_keep=num_checkpoints)
	
	with tf.Session(config=config) as sess:
		global_step = tf.Variable(0, trainable = False, name='global_step')
		train_op = tf.train.AdamOptimizer(0.0002).minimize(model.loss, global_step = global_step)
		saver = tf.train.Saver(max_to_keep=num_checkpoints)
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
				model.input_keep_prob:1,
				model.output_keep_prob:1,
				}
			loss, acc, prob, p_labels = sess.run([model.loss, model.acc, model.prob, model.prob_labels], feed_dict)	
			return loss, acc, prob, p_labels

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
				model.input_keep_prob:1,
				model.output_keep_prob:1,
				}
			p_labels = sess.run(model.prob_labels, feed_dict)	
			return p_labels
			
		print 'start train..........'
		batches =  data_helper.iter_batch(batch_size, num_epochs, train_warrant0, train_warrant1, train_reason, train_claim, train_labels, train_warrant0_len, train_warrant1_len, train_reason_len, train_claim_len)
		train_acc_list = []
		train_loss_list = []
		max_dev_acc = float('-inf')
		min_dev_loss = float('inf')
		num_undesc = 0
		for current_epoch , batch in batches:
			if num_undesc > max_num_undsc:
				break
			warrant0, warrant1, reason, claim, labels, warrant0_len, warrant1_len, reason_len, claim_len = batch
			step, train_batch_loss, train_batch_acc = train_step(warrant0, warrant1, reason, claim, labels, warrant0_len, warrant1_len, reason_len, claim_len)
			current_step = tf.train.global_step(sess, global_step)
			train_acc_list.append(train_batch_acc)
			train_loss_list.append(train_batch_loss)
			if current_step % evaluate_every == 0:
				train_acc = np.mean(train_acc_list)
				train_loss = np.mean(train_loss_list)
				print "train loss : %g, train acc : %g" %(train_loss, train_acc)
				train_acc_list = []
				train_loss_list = []
				dev_loss, dev_acc, dev_prob, p_dev_labels = dev_step(dev_warrant0,
			    							   dev_warrant1,
											   dev_reason,
											   dev_claim,
										       dev_labels, 
											   dev_warrant0_len,
											   dev_warrant1_len,
											   dev_reason_len,
											   dev_claim_len)
				print "Dev Loss : %g, Dev Acc : %g" %(dev_loss, dev_acc)
				test_labels = test_step(test_warrant0,
			    		   			   test_warrant1,
									   test_reason,
									   test_claim, 
									   test_warrant0_len,
									   test_warrant1_len,
									   test_reason_len,
									   test_claim_len)
				print '**************save it save it ************* '
				acc = accuracy_score(test_labels, gold)
				print acc
		'''
				if dev_acc >= max_dev_acc:
					max_dev_acc = dev_acc
					min_dev_loss = dev_loss
					num_undesc = 0
					test_labels = test_step(test_warrant0,
				    		   			   test_warrant1,
										   test_reason,
										   test_claim, 
										   test_warrant0_len,
										   test_warrant1_len,
										   test_reason_len,
										   test_claim_len)
					print '**************save it save it ************* '
					acc = accuracy_score(test_labels, gold)
					print acc
				else:
					num_undesc += 1
				print '-------------------------'
		print "TestAcc: %g, MinDevAcc: %g" %(acc, max_dev_acc)
		'''
