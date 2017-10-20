import sys, os, chainer
import numpy as np
from chainer import functions, cuda
sys.path.append(os.path.join("..", ".."))
import imsat.nn as nn

class Model(nn.Module):
	def __init__(self, ndim_x=28*28, num_clusters=10, ndim_h=1200):
		self.ndim_x = ndim_x
		self.num_clusters = num_clusters
		self.ndim_h = ndim_h

		super(Model, self).__init__(
			nn.Linear(ndim_x, ndim_h),
			nn.ReLU(),
			nn.BatchNormalization(ndim_h),
			nn.Linear(ndim_h, ndim_h),
			nn.ReLU(),
			nn.BatchNormalization(ndim_h),
			nn.Linear(ndim_h, num_clusters),
		)

		for param in self.params():
			if param.name == "W":
				param.data[...] = np.random.normal(0, 0.01, param.data.shape)

	def to_numpy(self, x):
		if isinstance(x, chainer.Variable) == True:
			x = x.data
		if isinstance(x, cuda.ndarray) == True:
			x = cuda.to_cpu(x)
		return x

	def classify(self, x, apply_softmax=True, as_numpy=False):
		y = self(x)
		if apply_softmax:
			y = functions.softmax(y)
		if as_numpy:
			y = self.to_numpy(y)
		return y

	def compute_entropy(self, p):
		if p.ndim == 2:
			return -functions.sum(p * functions.log(p + 1e-16), axis=1)
		return -functions.sum(p * functions.log(p + 1e-16))

	def compute_marginal_entropy(self, p_batch):
		return self.compute_entropy(functions.mean(p_batch, axis=0))

	def compute_kld(self, p, q):
		assert p.shape[0] == q.shape[0]
		return functions.reshape(functions.sum(p * (functions.log(p + 1e-16) - functions.log(q + 1e-16)), axis=1), (-1, 1))

	def get_unit_vector(self, v):
		xp = cuda.get_array_module(v)
		if v.ndim == 4:
			return v / (xp.sqrt(xp.sum(v ** 2, axis=(1,2,3))).reshape((-1, 1, 1, 1)) + 1e-16)
		return v / (xp.sqrt(xp.sum(v ** 2, axis=1)).reshape((-1, 1)) + 1e-16)

	def compute_lds(self, x, xi=10, eps=1, Ip=1):
		xp = cuda.get_array_module(x)
		y1 = self.classify(x, apply_softmax=True)
		y1.unchain_backward()
		d = chainer.Variable(self.get_unit_vector(xp.random.normal(size=x.shape).astype(xp.float32)))
		for i in range(Ip):
			y2 = self.classify(x + xi * d, apply_softmax=True)
			kld = functions.sum(self.compute_kld(y1, y2))
			kld.backward()
			d = self.get_unit_vector(d.grad)
		
		y2 = self.classify(x + eps * d, apply_softmax=True)
		return -self.compute_kld(y1, y2)

