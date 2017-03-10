# -*- coding: utf-8 -*-
import os
import numpy as np
import emnist_tools

def load_train_images():
	return emnist_tools.load_train_images()

def load_test_images():
	return emnist_tools.load_test_images()

def create_semisupervised(images, labels, num_labeled_data=10):
	training_images_l = []
	training_images_u = []
	training_labels = []
	indices_for_label = {}
	assert num_labeled_data % 10 == 0
	num_data_per_label = int(num_labeled_data / 10)
	num_unlabeled_data = len(images) - num_labeled_data

	indices = np.arange(len(images))
	np.random.shuffle(indices)

	def check(index):
		label = labels[index]
		if label not in indices_for_label:
			if num_data_per_label == 0:
				return False
			indices_for_label[label] = []
			return True
		if len(indices_for_label[label]) < num_data_per_label:
			for i in indices_for_label[label]:
				if i == index:
					return False
			return True
		return False

	for n in xrange(len(images)):
		index = indices[n]
		if check(index):
			indices_for_label[labels[index]].append(index)
			training_images_l.append(images[index])
			training_labels.append(labels[index])
		else:
			training_images_u.append(images[index])

	return training_images_l, training_labels, training_images_u

def sample_labeled_data(images, labels, batchsize):
	ndim_x = images[0].size
	image_batch = np.zeros((batchsize, ndim_x), dtype=np.float32)
	label_id_batch = np.zeros((batchsize,), dtype=np.int32)
	indices = np.random.choice(np.arange(len(images), dtype=np.int32), size=batchsize, replace=False)
	for j in range(batchsize):
		data_index = indices[j]
		image = images[data_index].astype(np.float32)
		image_batch[j] = image.reshape((ndim_x,))
		label_id_batch[j] = labels[data_index]
	return image_batch, label_id_batch

def sample_data(images, batchsize):
	ndim_x = images[0].size
	image_batch = np.zeros((batchsize, ndim_x), dtype=np.float32)
	indices = np.random.choice(np.arange(len(images), dtype=np.int32), size=batchsize, replace=False)
	for j in range(batchsize):
		data_index = indices[j]
		image = images[data_index].astype(np.float32)
		image_batch[j] = image.reshape((ndim_x,))
	return image_batch