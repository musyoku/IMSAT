# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six, math, random, time, copy
from chainer import cuda, Variable, optimizers, serializers, function, optimizer, initializers
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L
import sequential

class Object(object):
	pass

def to_object(dict):
	obj = Object()
	for key, value in dict.iteritems():
		setattr(obj, key, value)
	return obj

class Params():
	def __init__(self, dict=None):
		if dict:
			self.from_dict(dict)

	def from_dict(self, dict):
		for attr, value in dict.iteritems():
			if hasattr(self, attr):
				setattr(self, attr, value)

	def to_dict(self):
		dict = {}
		for attr, value in self.__dict__.iteritems():
			if hasattr(value, "to_dict"):
				dict[attr] = value.to_dict()
			else:
				dict[attr] = value
		return dict

	def dump(self):
		for attr, value in self.__dict__.iteritems():
			print "	{}: {}".format(attr, value)

class ClassifierParams(Params):
	def __init__(self):
		self.num_clusters = 10
		self.weight_std = 0.01
		self.weight_initializer = "Normal"		# Normal, GlorotNormal or HeNormal
		self.nonlinearity = "relu"
		self.optimizer = "adam"
		self.learning_rate = 0.0001
		self.momentum = 0.9
		self.gradient_clipping = 1
		self.weight_decay = 0
		self.lam = 0.1
		self.mu = 5.0
		self.ip = 1

class Classifier():
	def __init__(self, params):
		self.params = copy.deepcopy(params)

		config = to_object(params["config"])
		self.classifier = sequential.chain.Chain(weight_initializer=config.weight_initializer, weight_std=config.weight_std)
		self.classifier.add_sequence(sequential.from_dict(self.params["model"]))
		self.classifier.setup_optimizers(config.optimizer, config.learning_rate, config.momentum)

		self.config = config
		self._gpu = False

	def update_learning_rate(self, lr):
		self.classifier.update_learning_rate(lr)

	def to_gpu(self):
		self.classifier.to_gpu()
		self._gpu = True

	@property
	def gpu_enabled(self):
		if cuda.available is False:
			return False
		return self._gpu

	@property
	def xp(self):
		if self.gpu_enabled:
			return cuda.cupy
		return np

	def to_variable(self, x):
		if isinstance(x, Variable) == False:
			x = Variable(x)
			if self.gpu_enabled:
				x.to_gpu()
		return x

	def to_numpy(self, x):
		if isinstance(x, Variable) == True:
			x = x.data
		if isinstance(x, cuda.ndarray) == True:
			x = cuda.to_cpu(x)
		return x

	def get_batchsize(self, x):
		return x.shape[0]

	def classify(self, x, test=False, apply_softmax=True, as_numpy=False):
		x = self.to_variable(x)
		p = self.classifier(x, test=test)
		if apply_softmax:
			p = F.softmax(p)
		if as_numpy:
			return self.to_numpy(p)
		return p

	def backprop(self, loss):
		self.classifier.backprop(loss)

	def load(self, dir=None):
		if dir is None:
			raise Exception()
		self.classifier.load(dir + "/classifier.hdf5")

	def save(self, dir=None):
		if dir is None:
			raise Exception()
		try:
			os.mkdir(dir)
		except:
			pass
		self.classifier.save(dir + "/classifier.hdf5")

	def compute_entropy(self, p):
		if p.ndim == 2:
			return -F.sum(p * F.log(p + 1e-16), axis=1)
		return -F.sum(p * F.log(p + 1e-16))

	def compute_marginal_entropy(self, p_batch):
		p = F.sum(p_batch, axis=0) / self.get_batchsize(p_batch)
		return self.compute_entropy(p)

	def compute_kld(self, p, q):
		assert self.get_batchsize(p) == self.get_batchsize(q)
		return F.reshape(F.sum(p * (F.log(p + 1e-16) - F.log(q + 1e-16)), axis=1), (-1, 1))

	def get_unit_vector(self, v):
		v /= (np.sqrt(np.sum(v ** 2, axis=1)).reshape((-1, 1)) + 1e-16)
		return v

	def compute_lds(self, x, xi=10, eps=1):
		x = self.to_variable(x)
		y1 = self.classify(x, apply_softmax=True)
		y1.unchain_backward()
		d = self.to_variable(self.get_unit_vector(np.random.normal(size=x.shape).astype(np.float32)))

		for i in xrange(self.config.ip):
			y2 = self.classify(x + xi * d, apply_softmax=True)
			kld = F.sum(self.compute_kld(y1, y2))
			kld.backward()
			d = self.to_variable(self.get_unit_vector(self.to_numpy(d.grad)))
		
		y2 = self.classify(x + eps * d, apply_softmax=True)
		return -self.compute_kld(y1, y2)

