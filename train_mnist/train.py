import numpy as np
import os, sys, time, math, pylab
import seaborn as sns
sns.set(font_scale=2.5)
import pandas as pd
import matplotlib.pyplot as plt
from chainer import cuda
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
import dataset
from progress import Progress
from model import imsat, params
from args import args

# load MNIST
train_images, train_labels = dataset.load_train_images()
train_images_l, train_labels_l, train_images_u = dataset.create_semisupervised(train_images, train_labels, num_labeled_data=args.num_labeled_data)
print "labeled: ", len(train_images_l)
print "unlabeled: ", len(train_images_u)
test_images, test_labels = dataset.load_test_images()

# config
config = imsat.config

def compute_accuracy(images, labels_true):
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

def plot(counts, filename):
	fig = pylab.gcf()
	fig.set_size_inches(20, 20)
	pylab.clf()
	dataframe = {}
	for label in xrange(10):
		dataframe[label] = {}
		for cluster in xrange(10):
			dataframe[label][cluster] = counts[label][cluster]

	dataframe = pd.DataFrame(dataframe)
	ax = sns.heatmap(dataframe, annot=False, fmt="f", linewidths=0)
	ax.tick_params(labelsize=30) 
	plt.yticks(rotation=0)
	plt.xlabel("ground truth")
	plt.ylabel("cluster")
	heatmap = ax.get_figure()
	heatmap.savefig("{}/{}.png".format(args.model_dir, filename))

def main():
	# settings
	max_epoch = 1000
	num_updates_per_epoch = 500
	batchsize_u = 256
	batchsize_l = min(100, args.num_labeled_data)

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
			x_u = dataset.sample_data(train_images_u, batchsize_u)
			p = imsat.classify(x_u, apply_softmax=True)
			hy = imsat.compute_marginal_entropy(p)
			hy_x = F.sum(imsat.compute_entropy(p)) / batchsize_u
			Rsat = -F.sum(imsat.compute_lds(x_u)) / batchsize_u

			# semi-supervised
			loss_semisupervised = 0
			if args.num_labeled_data > 0:
				x_l, t_l = dataset.sample_labeled_data(train_images_l, train_labels_l, batchsize_l)
				log_p = imsat.classify(x_l, apply_softmax=False)
				loss_semisupervised = F.softmax_cross_entropy(log_p, imsat.to_variable(t_l))

			loss = Rsat - config.lam * (config.mu * hy - hy_x) + config.sigma * loss_semisupervised
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
		plot(counts_train, "train")
		plot(counts_test, "test")


if __name__ == "__main__":
	main()
