import gzip, os, sys
from PIL import Image
import numpy as np

train_images_filename = "emnist-balanced-train-images-idx3-ubyte.gz"
train_labels_filename = "emnist-balanced-train-labels-idx1-ubyte.gz"
test_images_filename = "emnist-balanced-test-images-idx3-ubyte.gz"
test_labels_filename = "emnist-balanced-test-labels-idx1-ubyte.gz"
num_train = 112800
num_test = 18800
ndim_image = 28 * 28

def load_emnist(data_filename, label_filename, num_images):
	images = np.zeros((num_images, ndim_image), dtype=np.float32)
	label = np.zeros((num_images,), dtype=np.int32)
	with gzip.open(data_filename, "rb") as f_images, gzip.open(label_filename, "rb") as f_labels:
		f_images.read(16)
		f_labels.read(8)
		for i in range(num_images):
			label[i] = ord(f_labels.read(1))
			for j in range(ndim_image):
				image = ord(f_images.read(1)) / 255.0
				images[i, j] = image
	return images, label

def load_train_images():
	assert os.path.exists(train_images_filename), "{} not found. You can download it from https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist".format(train_images_filename)
	images, labels = load_emnist(train_images_filename, train_labels_filename, num_train)
	return images, labels

def load_test_images():
	assert os.path.exists(test_images_filename), "{} not found. You can download it from https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist".foramt(test_images_filename)
	images, labels = load_emnist(test_images_filename, test_labels_filename, num_test)
	return images, labels
