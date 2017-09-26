from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange# pylint: disable=redefined-builtin
from six.moves import zip	 # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.ops import variable_scope

from tensorflow.python.client import timeline

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange# pylint: disable=redefined-builtin
import tensorflow as tf

# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = core_rnn_cell_impl._linear  # pylint: disable=protected-access
def get_concat_embedding(example_idxs, embedding_list):
		tmp_list = [None for i in xrange(len(embedding_list))]
		for i in xrange(len(embedding_list)):
			tmp_list[i] = tf.nn.embedding_lookup(embedding_list[i], example_idxs)
		return tf.concat(axis=1, values=tmp_list)


class MultiViewEmbedding_model(object):
	def __init__(self, data_set, vocab_size, review_size, user_size, product_size, 
				 vocab_distribute, review_distribute, product_distribute, window_size,
				 embed_size, max_gradient_norm, batch_size, learning_rate, L2_lambda, image_weight,
				 net_struct, similarity_func, forward_only=False, negative_sample = 5):
		"""Create the model.
	
		Args:
			vocab_size: the number of words in the corpus.
			dm_feature_len: the length of document model features (query based).
			review_size: the number of reviews in the corpus.
			user_size: the number of users in the corpus.
			product_size: the number of products in the corpus.
			embed_size: the size of each embedding
			window_size: the size of half context window
			vocab_distribute: the distribution for words, used for negative sampling
			review_distribute: the distribution for reviews, used for negative sampling
			product_distribute: the distribution for products, used for negative sampling
			max_gradient_norm: gradients will be clipped to maximally this norm.
			batch_size: the size of the batches used during training;
			the model construction is not independent of batch_size, so it cannot be
			changed after initialization.
			learning_rate: learning rate to start with.
			learning_rate_decay_factor: decay learning rate by this much when needed.
			forward_only: if set, we do not construct the backward pass in the model.
			negative_sample: the number of negative_samples for training
		"""
		self.data_set = data_set
		self.vocab_size = vocab_size
		self.review_size = review_size
		self.user_size = user_size
		self.product_size = product_size
		self.negative_sample = negative_sample
		self.embed_size = embed_size
		self.window_size = window_size
		self.vocab_distribute = vocab_distribute
		self.review_distribute = review_distribute
		self.product_distribute = product_distribute
		self.max_gradient_norm = max_gradient_norm
		self.batch_size = batch_size * (self.negative_sample + 1)
		self.init_learning_rate = learning_rate
		self.L2_lambda = L2_lambda
		self.image_weight = image_weight
		self.net_struct = net_struct
		self.similarity_func = similarity_func
		self.global_step = tf.Variable(0, trainable=False)

		# Feeds for inputs.
		self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
		self.review_idxs = tf.placeholder(tf.int64, shape=[None], name="review_idxs")
		self.user_idxs = tf.placeholder(tf.int64, shape=[None], name="user_idxs")
		self.product_idxs = tf.placeholder(tf.int64, shape=[None], name="product_idxs")
		self.word_idxs = tf.placeholder(tf.int64, shape=[None], name="word_idxs")
		
		self.img_feature_num = data_set.img_feature_num
		self.rate_factor_num = data_set.rate_factor_num
		#self.product_img_features = tf.placeholder(tf.float32, shape=[None, self.img_feature_num], name="img_features")
		#self.neg_product_idxs = tf.placeholder(tf.int64, shape=[self.negative_sample], name="neg_product_idxs")
		#self.neg_img_features = tf.placeholder(tf.float32, shape=[self.negative_sample, self.img_feature_num], name="neg_img_features")

		# check model configuration
		self.need_review = True
		if 'simplified' in self.net_struct:
			print('Simplified model')
			self.need_review = False
			
		self.need_context = False
		if 'hdc' in self.net_struct:
			print('Use context words')
			self.need_context = True
			self.context_word_idxs = []
			for i in xrange(2 * self.window_size):
				self.context_word_idxs.append(tf.placeholder(tf.int64, shape=[None], name="context_idx{0}".format(i)))

		self.need_bpr = False #need to directly maximize the product between user and purchased product
		if 'bpr' in self.net_struct:
			print('BPR training')
			self.need_bpr = True

		self.extendable = False
		if 'extend' in self.net_struct:
			self.extendable = True

		self.need_text = self.need_image = self.need_rate = False
		if 'text' in self.net_struct:
			self.need_text = True
		if 'image' in self.net_struct:
			self.need_image = True
		if 'rate' in self.net_struct:
			self.need_rate = True
		if not (self.need_text | self.need_image | self.need_rate):
			self.need_text = self.need_image = self.need_rate = True
		print('Need text: ' + str(self.need_text))
		print('Need image: ' + str(self.need_image))
		print('Need rate: ' + str(self.need_rate))

		# Training losses.
		self.loss = self.build_embedding_graph_and_loss()

		# Gradients and SGD update operation for training the model.
		params = tf.trainable_variables()
		if not forward_only:
			opt = tf.train.GradientDescentOptimizer(self.learning_rate)
			self.gradients = tf.gradients(self.loss, params)
			
			self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients,
																	 self.max_gradient_norm)
			self.updates = opt.apply_gradients(zip(self.clipped_gradients, params),
											 global_step=self.global_step)
			
			#self.updates = opt.apply_gradients(zip(self.gradients, params),
			#								 global_step=self.global_step)
		self.product_scores = self.get_product_scores(self.user_idxs)
	
		self.saver = tf.train.Saver(tf.global_variables())

	def get_all_user_product_embedding_list(self):
		user_emb_list = []
		product_emb_list = []
		product_bias_list = []
		if self.need_text:
			user_emb_list.append(self.user_emb)
			product_emb_list.append(self.product_emb)
			product_bias_list.append(self.product_bias)
		if self.need_image:
			user_emb_list.append(self.img_user_emb)
			product_emb_list.append(self.img_product_emb)
			product_bias_list.append(self.img_product_bias)
		if self.need_rate:
			user_emb_list.append(self.rate_user_emb)
			product_emb_list.append(self.rate_product_emb)
			product_bias_list.append(self.rate_product_bias)
		return user_emb_list, product_emb_list, product_bias_list

	def similarity_function(self, user_vec, product_vec, product_bias):
		print('Similarity Function : ' + self.similarity_func)
		if self.similarity_func == 'product':
			return tf.matmul(user_vec, product_vec, transpose_b=True)
		elif self.similarity_func == 'bias_product':
			return tf.matmul(user_vec, product_vec, transpose_b=True) + product_bias
		else:
			user_norm = tf.sqrt(tf.reduce_sum(tf.square(user_vec), 1, keep_dims=True))
			product_norm = tf.sqrt(tf.reduce_sum(tf.square(product_vec), 1, keep_dims=True))
			return tf.matmul(user_vec/user_norm, product_vec/product_norm, transpose_b=True)

	def get_product_scores(self, user_idxs, product_idxs = None, scope=None):
		with variable_scope.variable_scope(scope or "embedding_graph"):
			#get needed embeddings
			user_emb_list, product_emb_list, product_bias_list = self.get_all_user_product_embedding_list()

			if self.extendable:
				rank_scores = None
				for i in xrange(len(user_emb_list)):
					#compute score on each view
					user_vec = tf.nn.embedding_lookup(user_emb_list[i], user_idxs)
					if product_idxs != None:
						product_vec = tf.nn.embedding_lookup(product_emb_list[i], product_idxs)
						product_bias = tf.nn.embedding_lookup(product_bias_list[i], product_idxs)
					else:
						product_vec = product_emb_list[i]
						product_bias = product_bias_list[i]

					scores = self.similarity_function(user_vec, product_vec, product_bias)
					if self.similarity_func == 'product' or self.similarity_func == 'bias_product':
						scores = tf.sigmoid(scores)
					if rank_scores == None:
						rank_scores = scores
					else:
						rank_scores += scores
				return rank_scores
			else:
				#get user embedding [None, n*embed_size]
				user_vec = get_concat_embedding(user_idxs, user_emb_list)
				#get candidate product embedding [None, embed_size]
				product_vec = None
				product_bias = None
				if product_idxs != None:
					product_vec = get_concat_embedding(product_idxs, product_emb_list)
					product_bias = tf.nn.embedding_lookup(self.overall_product_bias, product_idxs)
				else:
					product_vec = tf.concat(axis=1, values=product_emb_list)
					product_bias = self.overall_product_bias
				return self.similarity_function(user_vec, product_vec, product_bias)
			

	def build_embedding_graph_and_loss(self, scope = None):
		with variable_scope.variable_scope(scope or "embedding_graph"):
			batch_size = array_ops.shape(self.user_idxs)[0] #get batch_size
			loss_list = []
			l2_regulization_list = []

			if self.need_text:
				# Word embeddings.
				init_width = 0.5 / self.embed_size
				self.word_emb = tf.Variable( tf.random_uniform(
									[self.vocab_size, self.embed_size], -init_width, init_width),
									name="word_emb")
				self.word_bias = tf.Variable(tf.zeros([self.vocab_size]), name="word_b")

				# user/product embeddings.
				self.user_emb =	tf.Variable( tf.zeros([self.user_size, self.embed_size]),
									name="user_emb")
				#self.user_bias =	tf.Variable( tf.zeros([self.user_size]), name="user_b")
				self.product_emb =	tf.Variable( tf.zeros([self.product_size, self.embed_size]),
									name="product_emb")
				self.product_bias =	tf.Variable( tf.zeros([self.product_size]), name="product_b")

				# Review embeddings.
				if self.need_review:
					self.review_emb = tf.Variable( tf.zeros([self.review_size, self.embed_size]),
										name="review_emb")
					self.review_bias = tf.Variable(tf.zeros([self.review_size]), name="review_b")

				# Context embeddings.
				if self.need_context:
					self.context_emb = tf.Variable( tf.zeros([self.vocab_size, self.embed_size]),
										name="context_emb")
					self.context_bias = tf.Variable(tf.zeros([self.vocab_size]), name="context_b")
				
				# Review loss
				review_loss = None
				if self.need_context:
					review_loss, regularize_emb_list = self.review_nce_loss(self.user_idxs, self.product_idxs, self.review_idxs,
												self.word_idxs, self.context_word_idxs)
				else:
					review_loss, regularize_emb_list = self.review_nce_loss(self.user_idxs, self.product_idxs, self.review_idxs,
												self.word_idxs)
				loss_list.append(review_loss)
				l2_regulization_list += regularize_emb_list

				# Extendable bpr loss
				if self.need_bpr and self.extendable:
					r_bpr_loss, regularize_emb_list = self.single_nce_loss(self.user_idxs, self.user_emb, self.product_idxs, self.product_emb, 
							self.product_bias, self.product_size, self.product_distribute)
					loss_list.append(r_bpr_loss)
					l2_regulization_list += regularize_emb_list
			
			if self.need_image:
				# Image embeddings
				self.img_user_emb =	tf.Variable( tf.zeros([self.user_size, self.embed_size]),
									name="img_user_emb")
				self.img_product_features =	tf.constant(self.data_set.img_features, shape=[self.product_size, self.img_feature_num],
									name="img_product_features")
				self.img_product_emb =	tf.Variable( tf.zeros([self.product_size, self.embed_size]),
									name="img_product_emb")
				self.img_product_bias =	tf.Variable( tf.zeros([self.product_size]), name="img_product_b")	

				# image loss
				image_loss, regularize_emb_list = self.img_nce_loss(self.user_idxs, self.product_idxs)
				loss_list.append(self.image_weight * image_loss)
				l2_regulization_list += regularize_emb_list

				# Extendable bpr loss
				if self.need_bpr and self.extendable:
					img_bpr_loss, regularize_emb_list = self.single_nce_loss(self.user_idxs, self.img_user_emb, self.product_idxs, self.img_product_emb, 
							self.img_product_bias, self.product_size, self.product_distribute)
					loss_list.append(img_bpr_loss)
					l2_regulization_list += regularize_emb_list

			if self.need_rate:
				# rate factors
				self.rate_user_emb =	tf.constant(self.data_set.user_factors, shape=[self.user_size, self.rate_factor_num],
									name="rate_user_emb", dtype=dtypes.float32)
				self.rate_product_emb =	tf.constant(self.data_set.product_factors, shape=[self.product_size, self.rate_factor_num],
									name="rate_product_emb", dtype=dtypes.float32)
				self.rate_product_bias =	tf.Variable( tf.zeros([self.product_size]), name="rate_product_b")	
				# Extendable bpr loss
				if self.need_bpr and self.extendable:
					rate_bpr_loss, regularize_emb_list = self.single_nce_loss(self.user_idxs, self.rate_user_emb, self.product_idxs, self.rate_product_emb, 
							self.rate_product_bias, self.product_size, self.product_distribute)
					loss_list.append(rate_bpr_loss)
					l2_regulization_list += regularize_emb_list
			
			# bpr loss
			if self.need_bpr and not self.extendable:
				self.overall_product_bias =	tf.Variable( tf.zeros([self.product_size]), name="overall_product_b")	
				bpr_loss, regularize_emb_list = self.multiview_bpr_loss(self.user_idxs, self.product_idxs) 
				loss_list.append(bpr_loss)
				l2_regulization_list += regularize_emb_list

			# L2 loss
			if self.L2_lambda > 0 and len(l2_regulization_list) > 0:
				l2_loss = tf.nn.l2_loss(l2_regulization_list[0])
				for i in xrange(1,len(l2_regulization_list)):
					l2_loss += tf.nn.l2_loss(l2_regulization_list[i])
				loss_list.append(self.L2_lambda * l2_loss)

			loss = loss_list[0]
			for i in xrange(1, len(loss_list)):
				loss += loss_list[i]
			
			return loss / math_ops.cast(batch_size, dtypes.float32)


	def review_nce_loss(self, user_idxs, product_idxs, review_idxs,
								word_idxs, context_word_idxs = None):
		loss_list = []
		l2_regulization_list = []
		if self.need_review:
			#review prediction loss
			ur_loss, ur_emb_list  = self.single_nce_loss(user_idxs, self.user_emb, review_idxs, self.review_emb, 
							self.review_bias, self.review_size, self.review_distribute)
			pr_loss, pr_emb_list = self.single_nce_loss(product_idxs, self.product_emb, review_idxs, self.review_emb, 
							self.review_bias, self.review_size, self.review_distribute)
			#word prediction loss
			wr_loss, wr_emb_list = self.single_nce_loss(review_idxs, self.review_emb, word_idxs, self.word_emb, 
							self.word_bias, self.vocab_size, self.vocab_distribute)
			loss_list.append(ur_loss)
			loss_list.append(pr_loss)
			loss_list.append(wr_loss)
			l2_regulization_list += ur_emb_list + pr_emb_list + wr_emb_list
		else:
			#word prediction loss
			uw_loss, uw_emb_list = self.single_nce_loss(user_idxs, self.user_emb, word_idxs, self.word_emb, 
							self.word_bias, self.vocab_size, self.vocab_distribute)
			pw_loss, pw_emb_list = self.single_nce_loss(product_idxs, self.product_emb, word_idxs, self.word_emb, 
							self.word_bias, self.vocab_size, self.vocab_distribute)
			loss_list.append(uw_loss)
			loss_list.append(pw_loss)
			l2_regulization_list += uw_emb_list + pw_emb_list
		#context prediction loss
		if self.need_context:
			for context_word_idx in context_word_idxs:
				cw_loss, cw_emb_list = self.single_nce_loss(word_idxs, self.word_emb, context_word_idx, self.context_emb, 
							self.context_bias, self.vocab_size, self.vocab_distribute)
				loss_list.append(cw_loss)
				l2_regulization_list += cw_emb_list

		#product prediction loss (do we still need this?)
		#if self.need_bpr:
		#	loss += self.single_nce_loss(user_idxs, self.user_emb, product_idxs, self.product_emb, 
		#					self.product_bias, self.product_size, self.product_distribute)
		loss = loss_list[0]
		for i in xrange(1, len(loss_list)):
			loss += loss_list[i]

		return loss, l2_regulization_list

	def single_nce_loss(self, example_idxs, example_emb, label_idxs, label_emb, 
						label_bias, label_size, label_distribution):
		batch_size = array_ops.shape(example_idxs)[0]#get batch_size
		# Nodes to compute the nce loss w/ candidate sampling.
		labels_matrix = tf.reshape(tf.cast(label_idxs,dtype=tf.int64),[batch_size, 1])

		# Negative sampling.
		sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
				true_classes=labels_matrix,
				num_true=1,
				num_sampled=self.negative_sample,
				unique=False,
				range_max=label_size,
				distortion=0.75,
				unigrams=label_distribution))

		#get example embeddings [batch_size, embed_size]
		example_vec = tf.nn.embedding_lookup(example_emb, example_idxs)

		#get label embeddings and bias [batch_size, embed_size], [batch_size, 1]
		true_w = tf.nn.embedding_lookup(label_emb, label_idxs)
		true_b = tf.nn.embedding_lookup(label_bias, label_idxs)

		#get sampled embeddings and bias [num_sampled, embed_size], [num_sampled, 1]
		sampled_w = tf.nn.embedding_lookup(label_emb, sampled_ids)
		sampled_b = tf.nn.embedding_lookup(label_bias, sampled_ids)

		# True logits: [batch_size, 1]
		true_logits = tf.reduce_sum(tf.multiply(example_vec, true_w), 1) + true_b

		# Sampled logits: [batch_size, num_sampled]
		# We replicate sampled noise lables for all examples in the batch
		# using the matmul.
		sampled_b_vec = tf.reshape(sampled_b, [self.negative_sample])
		sampled_logits = tf.matmul(example_vec, sampled_w, transpose_b=True) + sampled_b_vec

		return self.nce_loss(true_logits, sampled_logits), [example_vec, true_w, sampled_w]
		#return self.nce_loss(true_logits, true_logits)

	#get product embeddings
	def decode_img(self, input_data, reuse, scope=None):
		#reuse = None if index < 1 else True
		with variable_scope.variable_scope(scope or 'img_decode',
										 reuse=reuse):
			output_data = input_data
			output_sizes = [int((self.img_feature_num + self.embed_size)/2), self.img_feature_num]
			#output_sizes = [self.embed_size]
			current_size = self.embed_size
			for i in xrange(len(output_sizes)):
				expand_W = variable_scope.get_variable("expand_W_%d" % i, [current_size, output_sizes[i]])
				expand_b = variable_scope.get_variable("expand_b_%d" % i, [output_sizes[i]])
				output_data = tf.nn.bias_add(tf.matmul(output_data, expand_W), expand_b)
				output_data = tf.nn.elu(output_data)
				current_size = output_sizes[i]
				#print(expand_W.name)
			return output_data

	def img_nce_loss(self, user_idxs, product_idxs):  #, neg_product_idxs, product_img_features, neg_img_features):
		loss = None

		#user/product image embeddings
		user_vec = tf.nn.embedding_lookup(self.img_user_emb, user_idxs)
		product_vec = tf.nn.embedding_lookup(self.img_product_emb, product_idxs)

		#get decoding loss
		img_vec = tf.nn.embedding_lookup(self.img_product_features, product_idxs)
		user_img_vec = self.decode_img(user_vec, None)
		product_img_vec = self.decode_img(product_vec, True)

		loss = tf.nn.l2_loss(user_img_vec - img_vec)
		loss += tf.nn.l2_loss(product_img_vec - img_vec)
		
		return loss, [user_vec, product_vec]

	def multiview_bpr_loss(self, user_idxs, product_idxs): #, neg_product_idxs, product_img_emb, neg_img_emb):
		batch_size = array_ops.shape(user_idxs)[0]#get batch_size
		loss = None

		# Nodes to compute the nce loss w/ candidate sampling.
		label_idxs = product_idxs
		label_distribution = self.product_distribute
		label_size = self.product_size

		# Negative sampling.
		labels_matrix = tf.reshape(tf.cast(label_idxs,dtype=tf.int64),[batch_size, 1])
		neg_product_idxs, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
				true_classes=labels_matrix,
				num_true=1,
				num_sampled=self.negative_sample,
				unique=False,
				range_max=label_size,
				distortion=0.75,
				unigrams=label_distribution))


		user_emb_list, product_emb_list, _ = self.get_all_user_product_embedding_list()
		# concat user embeddings
		example_vec = get_concat_embedding(user_idxs, user_emb_list)

		# concat product embeddings
		#get label embeddings and bias [batch_size, embed_size], [batch_size, 1]
		true_w = get_concat_embedding(product_idxs, product_emb_list)
		true_b = tf.nn.embedding_lookup(self.overall_product_bias, product_idxs)

		#get sampled embeddings and bias [num_sampled, embed_size], [num_sampled, 1]
		sampled_w = get_concat_embedding(neg_product_idxs, product_emb_list)
		sampled_b = tf.nn.embedding_lookup(self.overall_product_bias, neg_product_idxs)
		
		# True logits: [batch_size, 1]
		true_logits = tf.reduce_sum(tf.multiply(example_vec, true_w), 1) + true_b

		# Sampled logits: [batch_size, num_sampled]
		# We replicate sampled noise lables for all examples in the batch
		# using the matmul.
		sampled_b_vec = tf.reshape(sampled_b, [self.negative_sample])
		sampled_logits = tf.matmul(example_vec, sampled_w, transpose_b=True) + sampled_b_vec
		
		loss = self.nce_loss(true_logits, sampled_logits) 
		print('Sigmoid loss')
		return loss, [example_vec, true_w, sampled_w]


	def nce_loss(self, true_logits, sampled_logits):
		"""Build the graph for the NCE loss."""

		# cross-entropy(logits, labels)
		true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
				logits=true_logits, labels=tf.ones_like(true_logits))
		sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
				logits=sampled_logits, labels=tf.zeros_like(sampled_logits))

		# NCE-loss is the sum of the true and noise (sampled words)
		# contributions, averaged over the batch.
		nce_loss_tensor = (tf.reduce_sum(true_xent) + tf.reduce_sum(sampled_xent)) 
		return nce_loss_tensor


	
	def step(self, session, learning_rate, user_idxs, product_idxs, review_idxs, 
					word_idxs, context_idxs, forward_only, test_mode = 'product_scores'):
		"""Run a step of the model feeding the given inputs.
	
		Args:
			session: tensorflow session to use.
			learning_rate: the learning rate of current step
			user_idxs: A numpy [1] float vector.
			product_idxs: A numpy [1] float vector.
			review_idxs: A numpy [1] float vector.
			word_idxs: A numpy [None] float vector.
			context_idxs: list of numpy [None] float vectors.
			product_img_features: image features for the product
			neg_product_idxs: negative sample indexes for image training 
			neg_img_features: negative samples' image feature
			forward_only: whether to do the update step or only forward.
	
		Returns:
			A triple consisting of gradient norm (or None if we did not do backward),
			average perplexity, and the outputs.
	
		Raises:
			ValueError: if length of encoder_inputs, decoder_inputs, or
			target_weights disagrees with bucket size for the specified bucket_id.
		"""
	
		# Input feed: encoder inputs, decoder inputs, target_weights, as provided.
		input_feed = {}
		input_feed[self.learning_rate.name] = learning_rate
		input_feed[self.user_idxs.name] = user_idxs
		input_feed[self.product_idxs.name] = product_idxs
		input_feed[self.review_idxs.name] = review_idxs
		input_feed[self.word_idxs.name] = word_idxs
		if context_idxs != None:
			for i in xrange(2 * self.window_size):
				input_feed[self.context_word_idxs[i].name] = context_idxs[i]

		#add image features
		#input_feed[self.product_img_features.name] = product_img_features
		#input_feed[self.neg_img_features.name] = neg_img_features

		#negative sampling
		#input_feed[self.neg_product_idxs.name] = neg_product_idxs 
	
		# Output feed: depends on whether we do a backward step or not.
		if not forward_only:
			output_feed = [self.updates,	# Update Op that does SGD.
						 #self.norm,	# Gradient norm.
						 self.loss]	# Loss for this batch.
		else:
			if test_mode == 'output_embedding':
				output_feed = [self.user_emb, self.product_emb, self.word_emb, self.word_bias]
				if self.need_review:
					output_feed += [self.review_emb, self.review_bias]
				
				if self.need_context:
					output_feed += [self.context_emb, self.context_bias]
				
			else:
				output_feed = [self.product_scores] #negative instance output
	
		outputs = session.run(output_feed, input_feed)

		if not forward_only:
			return outputs[1], None	# loss, no outputs, Gradient norm.
		else:
			if test_mode == 'output_embedding':
				return outputs[:4], outputs[4:]
			else:
				return outputs[0], None	# product scores to input user

	def get_batch(self, data_set, review_idx, batch_word_idx):
		"""
		Randomly select a query, then randomly select one of its relevant document 
		as a positive instance. After that, randomly select N document from the whole
		data_set as negative instances. Repeat this process for batch_size times
	
		Args:
			data_set: a instance of data_util.Tensorflow_data
	
		Returns:
			The triple (word_idxs, context_word_idxs) for
			the constructed batch that has the proper format to call step(...) later.
		"""
		text_list = data_set.review_text[review_idx]
		text_length = len(text_list)
		word_idxs = []
		context_word_idxs = []
		need_context = self.need_context
		count_words = 0
		i = batch_word_idx
		while count_words < self.batch_size and i < text_length: #there is the possibility that the batch is empty
			if random.random() < data_set.sub_sampling_rate[text_list[i]]:
				word_idxs.append(text_list[i])
				count_words += 1

				if need_context:
					start_index = i - self.window_size if i - self.window_size > 0 else 0
					context_word_list = text_list[start_index:i] + text_list[i+1:i+self.window_size+1]
					while len(context_word_list) < 2 * self.window_size:
						context_word_list += text_list[start_index:start_index+2*self.window_size-len(context_word_list)]
					context_word_idxs.append(context_word_list)
				
			i += 1

		'''
		if len(word_idxs) < 1:
			i = text_length - 1
			word_idxs.append(text_list[i])
			count_words += 1

			if self.net_struct == 'hdc':
				start_index = i - self.window_size if i - self.window_size > 0 else 0
				context_word_list = text_list[start_index:i] + text_list[i+1:i+self.window_size+1]
				while len(context_word_list) < 2 * self.window_size:
					context_word_list += text_list[start_index:start_index+2*self.window_size-len(context_word_list)]
				context_word_idxs.append(context_word_list)	
			i += 1
		'''

		#reorganize context_word_idxs
		
		if need_context:
			length = len(word_idxs)
			batch_context_word_idxs = []
			for length_idx in xrange(2 * self.window_size):
				batch_context_word_idxs.append(np.array([context_word_idxs[batch_idx][length_idx]
						for batch_idx in xrange(length)], dtype=np.int64))

			return word_idxs, batch_context_word_idxs, i - batch_word_idx

		return word_idxs, None, i - batch_word_idx

	def setup_data_set(self, data_set, words_to_train):
		self.data_set = data_set
		self.words_to_train = words_to_train
		self.finished_word_num = 0
		if self.net_struct == 'hdc':
			self.need_context = True

	def intialize_epoch(self, training_seq):
		self.train_seq = training_seq
		self.review_size = len(self.train_seq)
		self.cur_review_i = 0
		self.cur_word_i = 0
		self.tested_user = set()

	def get_train_batch(self):
		user_idxs, product_idxs, review_idxs, word_idxs, context_word_idxs = [],[],[],[],[]
		#product_img_features, neg_product_idxs, neg_img_features = [],[],[] #add image
		learning_rate = self.init_learning_rate * max(0.0001, 
									1.0 - self.finished_word_num / self.words_to_train)
		review_idx = self.train_seq[self.cur_review_i]
		user_idx = self.data_set.review_info[review_idx][0]
		product_idx = self.data_set.review_info[review_idx][1]
		text_list = self.data_set.review_text[review_idx]
		text_length = len(text_list)

		while len(word_idxs) < self.batch_size:
			#print('review %d word %d word_idx %d' % (review_idx, self.cur_word_i, text_list[self.cur_word_i]))
			#if sample this word
			if random.random() < self.data_set.sub_sampling_rate[text_list[self.cur_word_i]]:
				user_idxs.append(user_idx)
				product_idxs.append(product_idx)
				review_idxs.append(review_idx)
				word_idxs.append(text_list[self.cur_word_i])
				if self.need_context:
					i = self.cur_word_i
					start_index = i - self.window_size if i - self.window_size > 0 else 0
					context_word_list = text_list[start_index:i] + text_list[i+1:i+self.window_size+1]
					while len(context_word_list) < 2 * self.window_size:
						context_word_list += text_list[start_index:start_index+2*self.window_size-len(context_word_list)]
					context_word_idxs.append(context_word_list)
				# add image
				#product_img_features.append(self.data_set.img_features[product_idx])

			#move to the next
			self.cur_word_i += 1
			self.finished_word_num += 1
			if self.cur_word_i == text_length:
				self.cur_review_i += 1
				if self.cur_review_i == self.review_size:
					break
				self.cur_word_i = 0
				review_idx = self.train_seq[self.cur_review_i]
				user_idx = self.data_set.review_info[review_idx][0]
				product_idx = self.data_set.review_info[review_idx][1]
				text_list = self.data_set.review_text[review_idx]
				text_length = len(text_list)

		batch_context_word_idxs = None
		length = len(word_idxs)
		if self.need_context:
			batch_context_word_idxs = []
			for length_idx in xrange(2 * self.window_size):
				batch_context_word_idxs.append(np.array([context_word_idxs[batch_idx][length_idx]
						for batch_idx in xrange(length)], dtype=np.int64))
		'''
		# add negative product samples
		while len(neg_product_idxs) < self.negative_sample:
			neg_product_idx = int(random.random() * self.product_size)
			neg_product_idxs.append(neg_product_idx)
			neg_img_features.append(self.data_set.img_features[neg_product_idx])
		'''
		has_next = False if self.cur_review_i == self.review_size else True
		return user_idxs, product_idxs, review_idxs, word_idxs, batch_context_word_idxs, learning_rate, has_next

	def get_test_batch(self):
		user_idxs, product_idxs, review_idxs, word_idxs, context_word_idxs = [],[],[],[],[]
		#product_img_features, neg_product_idxs, neg_img_features = [],[],[] #add image
		learning_rate = self.init_learning_rate * max(0.0001, 
									1.0 - self.finished_word_num / self.words_to_train)
		review_idx = self.train_seq[self.cur_review_i]
		user_idx = self.data_set.review_info[review_idx][0]

		# add image
		#product_img_features = self.data_set.img_features

		while len(user_idxs) < self.batch_size:
			if user_idx not in self.tested_user:
				product_idx = self.data_set.review_info[review_idx][1]
				text_list = self.data_set.review_text[review_idx]
				user_idxs.append(user_idx)
				product_idxs.append(product_idx)
				review_idxs.append(review_idx)
				word_idxs.append(text_list[0])
				if self.need_context:
					i = 0
					start_index = i - self.window_size if i - self.window_size > 0 else 0
					context_word_list = text_list[start_index:i] + text_list[i+1:i+self.window_size+1]
					while len(context_word_list) < 2 * self.window_size:
						context_word_list += text_list[start_index:start_index+2*self.window_size-len(context_word_list)]
					context_word_idxs.append(context_word_list)

				self.tested_user.add(user_idx)
			
			#move to the next review
			self.cur_review_i += 1
			if self.cur_review_i == self.review_size:
				break
			review_idx = self.train_seq[self.cur_review_i]
			user_idx = self.data_set.review_info[review_idx][0]

		batch_context_word_idxs = None
		length = len(word_idxs)
		if self.need_context:
			batch_context_word_idxs = []
			for length_idx in xrange(2 * self.window_size):
				batch_context_word_idxs.append(np.array([context_word_idxs[batch_idx][length_idx]
						for batch_idx in xrange(length)], dtype=np.int64))
		'''
		# add negative product samples
		while len(neg_product_idxs) < self.negative_sample:
			neg_product_idx = int(random.random() * self.product_size)
			neg_product_idxs.append(neg_product_idx)
			neg_img_features.append(self.data_set.img_features[neg_product_idx])
		'''

		has_next = False if self.cur_review_i == self.review_size else True
		return user_idxs, product_idxs, review_idxs, word_idxs, batch_context_word_idxs, learning_rate, has_next
