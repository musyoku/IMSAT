import numpy as np
import os, sys, time, math
from chainer import cuda
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
import dataset
from progress import Progress
from mnist_tools import load_train_images, load_test_images
from model import imsat, params
from args import args

# load MNIST
train_images, train_labels = dataset.load_train_images()
test_images, test_labels = dataset.load_test_images()

def test():
	probs = F.softmax(imsat.classify(np.asarray(test_images, dtype=np.float32), apply_softmax=True))
	probs.unchain_backward()
	probs = imsat.to_numpy(probs)
	table = np.zeros((10, 10), dtype=np.int)
	for i in xrange(len(test_images)):
		p = probs[i]
		label_predict = np.argmax(p)
		label_true = test_labels[i]
		table[label_true][label_predict] += 1
	print table
	
def main():
	# config
	config = imsat.config

	# settings
	max_epoch = 100
	num_updates_per_epoch = 500
	batchsize = 256

	# seed
	np.random.seed(args.seed)
	if args.gpu_device != -1:
		cuda.cupy.random.seed(args.seed)

	# training
	test()
	progress = Progress()
	for epoch in xrange(1, max_epoch + 1):
		progress.start_epoch(epoch, max_epoch)
		sum_loss = 0

		for t in xrange(num_updates_per_epoch):
			x_u = dataset.sample_data(train_images, batchsize, binarize=False)
			log_p = imsat.classify(x_u, apply_softmax=False)
			p = F.softmax(log_p)
			hy = imsat.compute_marginal_entropy(p)
			hy_x = F.sum(imsat.compute_entropy(p)) / batchsize
			Rsat = -F.sum(imsat.compute_lds(x_u)) / batchsize

			loss = Rsat - config.lam * (5.0 * hy - hy_x)
			sum_loss += float(loss.data)
			imsat.backprop(loss)

			if t % 10 == 0:
				progress.show(t, num_updates_per_epoch, {})

		imsat.save(args.model_dir)

		progress.show(num_updates_per_epoch, num_updates_per_epoch, {
			"loss": sum_loss / num_updates_per_epoch,
		})

		# test
		test()

if __name__ == "__main__":
	main()
