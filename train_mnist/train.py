import numpy as np
import os, sys, time, math
from chainer import cuda
from chainer import functions as F
from munkres import Munkres, print_matrix
sys.path.append(os.path.split(os.getcwd())[0])
import dataset
from progress import Progress
from mnist_tools import load_train_images, load_test_images
from model import imsat, params
from args import args

# load MNIST
train_images, train_labels = dataset.load_train_images()
test_images, test_labels = dataset.load_test_images()

# config
config = imsat.config

def compute_accuracy(images, labels_true):
	images = np.asanyarray(images, dtype=np.float32)
	labels_true = np.asanyarray(labels_true, dtype=np.int32)
	probs = F.softmax(imsat.classify(images, test=True, apply_softmax=True))
	probs.unchain_backward()
	probs = imsat.to_numpy(probs)
	labels_predict = np.argmax(probs, axis=1)
	predict_counts = np.zeros((10, config.num_clusters), dtype=np.float32)
	for i in xrange(len(images)):
		p = probs[i]
		label_predict = labels_predict[i]
		label_true = labels_true[i]
		predict_counts[label_true][label_predict] += 1

	probs = np.transpose(predict_counts) / np.reshape(np.sum(np.transpose(predict_counts), axis=1), (config.num_clusters, 1))
	indices = np.argmax(probs, axis=1)
	match_count = np.zeros((10,), dtype=np.float32)
	for i in xrange(config.num_clusters):
		assinged_label = indices[i]
		match_count[assinged_label] += predict_counts[assinged_label][i]

	accuracy = np.sum(match_count) / images.shape[0]
	return predict_counts.astype(np.int), accuracy
	
def main():

	# settings
	max_epoch = 1000
	num_updates_per_epoch = 500
	batchsize = 256

	# seed
	np.random.seed(args.seed)
	if args.gpu_device != -1:
		cuda.cupy.random.seed(args.seed)

	# training
	progress = Progress()
	for epoch in xrange(1, max_epoch + 1):
		progress.start_epoch(epoch, max_epoch)
		sum_loss = 0
		sum_entropy = 0
		sum_conditional_entropy = 0
		sum_rsat = 0

		for t in xrange(num_updates_per_epoch):
			x_u = dataset.sample_data(train_images, batchsize)
			p = imsat.classify(x_u, apply_softmax=True)
			hy = imsat.compute_marginal_entropy(p)
			hy_x = F.sum(imsat.compute_entropy(p)) / batchsize
			Rsat = -F.sum(imsat.compute_lds(x_u)) / batchsize

			loss = Rsat - config.lam * (config.mu * hy - hy_x)
			imsat.backprop(loss)

			sum_loss += float(loss.data)
			sum_entropy += float(hy.data)
			sum_conditional_entropy += float(hy_x.data)
			sum_rsat += float(Rsat.data)

			if t % 10 == 0:
				progress.show(t, num_updates_per_epoch, {})

		imsat.save(args.model_dir)

		counts_train, accuracy_train = compute_accuracy(train_images, train_labels)
		counts_test, accuracy_test = compute_accuracy(test_images, test_labels)
		progress.show(num_updates_per_epoch, num_updates_per_epoch, {
			"loss": sum_loss / num_updates_per_epoch,
			"hy": sum_entropy / num_updates_per_epoch,
			"hy_x": sum_conditional_entropy / num_updates_per_epoch,
			"Rsat": sum_rsat / num_updates_per_epoch,
			"acc_test": accuracy_test,
			"acc_train": accuracy_test,
		})
		print counts_train
		print counts_test


if __name__ == "__main__":
	main()
