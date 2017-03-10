# -*- coding: utf-8 -*-
import math
import json, os, sys
from args import args
from chainer import cuda
sys.path.append(os.path.split(os.getcwd())[0])
from classifier import Classifier, ClassifierParams
from sequential import Sequential
from sequential.layers import Linear, BatchNormalization, MinibatchDiscrimination
from sequential.functions import Activation, dropout, gaussian_noise, softmax

# load params.json
try:
	os.mkdir(args.model_dir)
except:
	pass

# specify classifier
sequence_filename = args.model_dir + "/classifier.json"

if os.path.isfile(sequence_filename):
	print "loading", sequence_filename
	with open(sequence_filename, "r") as f:
		try:
			params = json.load(f)
		except Exception as e:
			raise Exception("could not load {}".format(sequence_filename))
else:
	config = ClassifierParams()
	config.num_clusters = 10
	config.weight_std = 0.01
	config.weight_initializer = "HeNormal"
	config.nonlinearity = "relu"
	config.optimizer = "adam"
	config.learning_rate = 0.0001
	config.momentum = 0.1
	config.gradient_clipping = 1
	config.weight_decay = 0
	config.lam = 0.1
	config.mu = 5.0
	config.ip = 1

	model = Sequential()
	model.add(Linear(None, 1200))
	model.add(Activation(config.nonlinearity))
	model.add(BatchNormalization(1200))
	model.add(Linear(None, 1200))
	model.add(Activation(config.nonlinearity))
	model.add(BatchNormalization(1200))
	model.add(Linear(None, config.num_clusters))

	params = {
		"config": config.to_dict(),
		"model": model.to_dict(),
	}

	with open(sequence_filename, "w") as f:
		json.dump(params, f, indent=4, sort_keys=True, separators=(',', ': '))

imsat = Classifier(params)
imsat.load(args.model_dir)

if args.gpu_device != -1:
	cuda.get_device(args.gpu_device).use()
	imsat.to_gpu()