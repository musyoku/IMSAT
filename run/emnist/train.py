from __future__ import division
from __future__ import print_function
import argparse, chainer, time, sys, copy
import numpy as np
import chainer.functions as F
from chainer import cuda
from model import Model
from imsat.optim import Optimizer, GradientClipping
from imsat.utils import printr, clear_console
from imsat.dataset import Dataset
import emnist_tools as emnist

def compute_accuracy(model, images, labels_true, num_clusters):
	with chainer.using_config("train", False) and chainer.no_backprop_mode():
		split_images = np.split(images, 50, axis=0)
		split_labels_true = np.split(labels_true, 50)
		predict_counts = np.zeros((47, num_clusters), dtype=np.float32)
		xp = model.xp

		for image_batch, label_true_batch in zip(split_images, split_labels_true):
			if xp is cuda.cupy:
				image_batch = cuda.to_gpu(image_batch)
			probs = F.softmax(model.classify(image_batch, apply_softmax=True))
			labels_predict = xp.argmax(probs.data, axis=1)
			for i in range(len(image_batch)):
				p = probs[i]
				label_predict = int(labels_predict[i])
				label_true = label_true_batch[i]
				predict_counts[label_true][label_predict] += 1

		probs = np.transpose(predict_counts) / np.reshape(np.sum(np.transpose(predict_counts), axis=1), (num_clusters, 1))
		indices = np.argmax(probs, axis=1)
		match_count = np.zeros((47,), dtype=np.float32)
		for i in range(num_clusters):
			assinged_label = indices[i]
			match_count[assinged_label] += predict_counts[assinged_label][i]

		accuracy = np.sum(match_count) / images.shape[0]
		return predict_counts.astype(np.int), accuracy

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--batchsize", "-b", type=int, default=64)
	parser.add_argument("--total-epochs", "-e", type=int, default=50)
	parser.add_argument("--gpu-device", "-g", type=int, default=0)
	parser.add_argument("--grad-clip", "-gc", type=float, default=5)
	parser.add_argument("--learning-rate", "-lr", type=float, default=0.01)
	parser.add_argument("--lr-decay", "-lr-decay", type=float, default=0.98)
	parser.add_argument("--momentum", "-mo", type=float, default=0.9)
	parser.add_argument("--num-clusters", "-cluster", type=int, default=47)
	parser.add_argument("--ndim-h", "-ndim-h", type=int, default=1200)
	parser.add_argument("--lam", "-lam", type=float, default=0.2)
	parser.add_argument("--mu", "-mu", type=float, default=4.0)
	parser.add_argument("--Ip", "-Ip", type=int, default=1)
	parser.add_argument("--optimizer", "-opt", type=str, default="msgd")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--model", "-m", type=str, default="model.hdf5")
	args = parser.parse_args()

	np.random.seed(args.seed)

	model = Model(num_clusters=args.num_clusters, ndim_h=args.ndim_h)
	model.load(args.model)

	images_train, labels_train = emnist.load_train_images()
	images_test, labels_test = emnist.load_test_images()

	dataset = Dataset(train=(images_train, labels_train), test=(images_test, labels_test))
	total_iterations_train = len(images_train) // args.batchsize

	# optimizers
	optimizer = Optimizer(args.optimizer, args.learning_rate, args.momentum)
	optimizer.setup(model)
	if args.grad_clip > 0:
		optimizer.add_hook(GradientClipping(args.grad_clip))

	using_gpu = False
	if args.gpu_device >= 0:
		cuda.get_device(args.gpu_device).use()
		model.to_gpu()
		using_gpu = True
	xp = model.xp

	training_start_time = time.time()
	for epoch in range(args.total_epochs):

		sum_loss = 0
		sum_entropy = 0
		sum_conditional_entropy = 0
		sum_rsat = 0

		epoch_start_time = time.time()
		dataset.shuffle()

		# training
		for itr in range(total_iterations_train):
			# update model parameters
			with chainer.using_config("train", True):
				# sample minibatch
				x_u, _ = dataset.sample_minibatch(args.batchsize, gpu=using_gpu)
				
				p = model.classify(x_u, apply_softmax=True)
				hy = model.compute_marginal_entropy(p)
				hy_x = F.sum(model.compute_entropy(p)) / args.batchsize
				Rsat = -F.sum(model.compute_lds(x_u)) / args.batchsize

				loss = Rsat - args.lam * (args.mu * hy - hy_x)

				model.cleargrads()
				loss.backward()
				optimizer.update()

				sum_loss += float(loss.data)
				sum_entropy += float(hy_x.data)
				sum_conditional_entropy += float(hy_x.data)
				sum_rsat += float(Rsat.data)

			printr("Training ... {:3.0f}% ({}/{})".format((itr + 1) / total_iterations_train * 100, itr + 1, total_iterations_train))

		model.save(args.model)
		
		counts_train, accuracy_train = compute_accuracy(model, images_train, labels_train, args.num_clusters)
		counts_test, accuracy_test = compute_accuracy(model, images_test, labels_test, args.num_clusters)

		clear_console()

		print(counts_train)
		print(counts_test)
		print("Epoch {} done in {} sec - loss {:.5g} - hy={:.5g} - hy_x={:.5g} - Rsat={:.5g} - acc: train={:.2f}%, test={:.2f}% - lr {:.5g} - total {} min".format(
			epoch + 1, int(time.time() - epoch_start_time), 
			sum_loss / total_iterations_train, 
			sum_entropy / total_iterations_train, 
			sum_conditional_entropy / total_iterations_train, 
			sum_rsat / total_iterations_train, 
			accuracy_train * 100,
			accuracy_test * 100,
			optimizer.get_learning_rate(),
			int((time.time() - training_start_time) // 60)))

		optimizer.decrease_learning_rate(args.lr_decay, 1e-5)


if __name__ == "__main__":
	main()
