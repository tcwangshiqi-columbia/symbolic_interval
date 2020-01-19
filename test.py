'''
 Test file for different sound propagation methods
 ** Top contributors:
 **   Shiqi Wang
 ** This file is part of the symbolic interval analysis library.
 ** Copyright (c) 2018-2019 by the authors listed in the file LICENSE
 ** and their institutional affiliations.
 ** All rights reserved.

 The basic network structures are inspired by the implementations of
 Eric Wong's convex polytope github available at 
 https://github.com/locuslab/convex_adversarial
'''


from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from symbolic_interval.symbolic_network import Interval_network
from symbolic_interval.symbolic_network import sym_interval_analyze
from symbolic_interval.symbolic_network import gen_interval_analyze
from symbolic_interval.symbolic_network import mix_interval_analyze
from symbolic_interval.symbolic_network import naive_interval_analyze

from utils_attacks import perturb_iterative

from bound_layers import BoundSequential, BoundLinear, BoundConv2d, BoundDataParallel, Flatten

import argparse

import time
import os

'''Flatten layers.
Please use this Flatten layer to flat the convolutional layers when 
design your own models.
'''
# class Flatten(nn.Module):
# 	def forward(self, x):
# 		return x.view(x.size(0), -1)

class Vlayer(nn.Module):
	def __init__(self, n):
		super(Vlayer, self).__init__()
		self.n = n
		self.w = torch.nn.Parameter(torch.ones((1,n)))
		#self.w = torch.nn.Parameter(torch.zeros((1,n)).uniform_(0,1))
		#print(self.w.sum()/self.n, self.w.max(), self.w.min())

	def forward(self, x):
		return x

def mnist_model_new(): 
	model = nn.Sequential(
		nn.Conv2d(1, 16, 4, stride=2, padding=1),
		nn.ReLU(),
		Vlayer(3136),
		nn.Conv2d(16, 32, 4, stride=2, padding=1),
		nn.ReLU(),
		Vlayer(1568),
		Flatten(),
		nn.Linear(32*7*7,100),
		nn.ReLU(),
		Vlayer(100),
		nn.Linear(100, 10)
	)
	return model

def transfer_model(model_new, model):

	import copy
	index = 0
	for i, layer in enumerate(model_new):
		#print(layer, isinstance(layer, nn.Linear))
		if 'Vlayer' in (str(layer.__class__.__name__)): 
			continue
		
		if isinstance(model[index], (nn.Conv2d, nn.Linear)) and\
					isinstance(layer, (nn.Conv2d, nn.Linear)):
			model_new[i] = copy.deepcopy(model[index])
		index += 1
	return model_new


'''Test MNIST models.
Please use nn.Sequential to build the models.
'''
def mnist_model(): 
	model = nn.Sequential(
		nn.Conv2d(1, 16, 4, stride=2, padding=1),
		nn.ReLU(),
		nn.Conv2d(16, 32, 4, stride=2, padding=1),
		nn.ReLU(),
		Flatten(),
		nn.Linear(32*7*7,100),
		nn.ReLU(),
		nn.Linear(100, 10)
	)
	return model


torch.manual_seed(7)

'''Test toy fully connected mnist models.
Please use nn.Sequential to build the models.
'''
def toy_model():
	model = nn.Sequential(
		Flatten(),
		nn.Linear(784,5),
		nn.ReLU(),
		nn.Linear(5,3),
		nn.ReLU(),
		nn.Linear(3,10),
		)
	return model


def mnist_loaders(batch_size, shuffle_test=False): 
	mnist_train = datasets.MNIST("./data", train=True, download=True,\
				transform=transforms.ToTensor())
	mnist_test = datasets.MNIST("./data", train=False, download=True,\
				transform=transforms.ToTensor())
	train_loader = torch.utils.data.DataLoader(mnist_train,\
				batch_size=batch_size, shuffle=True, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(mnist_test,\
				batch_size=batch_size, shuffle=shuffle_test,\
				pin_memory=True)
	return train_loader, test_loader


'''main function of this test script.
It compares the average tightness, loss, and efficiency of each
propagation methods.
You can test on you own models, desired epsilon, and batch_size.
'''
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=10)
	parser.add_argument('--method', default="sym")
	parser.add_argument('--proj', type=int, default=None)
	parser.add_argument('--epsilon', type=float, default=0.1)
	parser.add_argument('--PARALLEL', action='store_true', default=False)
	parser.add_argument('--compare_all', action='store_true', default=False)
	parser.add_argument('--norm', type=str, default="linf")
	parser.add_argument('--gpu', type=str, default="0")

	args = parser.parse_args()

	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	use_cuda = torch.cuda.is_available()

	# load model
	MODEL_NAME = "mnist_small_baseline.pth"
	model = mnist_model()
	if use_cuda:
		model.load_state_dict(torch.load(MODEL_NAME))
		model = model.cuda()
	else:
		model.load_state_dict(torch.load(MODEL_NAME, map_location={'cuda:0': 'cpu'}))

	#model = toy_model()
	epsilon = args.epsilon
	batch_size = args.batch_size
	PARALLEL = args.PARALLEL

	if not use_cuda:
		PARALLEL = False

	train_loader, test_loader = mnist_loaders(batch_size)	
	
	# Fetch test samples
	for i, (X,y) in enumerate(test_loader):
		
		if(use_cuda):
			X = X.cuda()
			y = y.cuda().long()
			model.cuda()
		break


	if args.method=="convexdual" or args.compare_all:
		from convex_adversarial import robust_loss
		
		# if(PARALLEL):
		# 	# The first run with parallel will take a few seconds
		# 	# to warm up.
		# 	eric_loss, eric_err = robust_loss(model,\
		# 				 epsilon, X, y, parallel=PARALLEL)
		# 	del eric_loss, eric_err

			
		start = time.time()
		eric_loss, eric_err = robust_loss(model,\
				epsilon, X, y,parallel=True,\
				bounded_input={0, 1})
				#norm_type="l1_median", proj=20)
		#eric_loss, eric_err = robust_loss(model,\
					#epsilon, X, y, parallel=PARALLEL)

		print ("eric loss", eric_loss)
		print ("eric err:", eric_err)
		print ("eric time per sample:", (time.time()-start)/X.shape[0])
		del eric_loss, eric_err
		print()

	if args.method == "crown" or args.compare_all:
		start = time.time()
		num_class = 10
		model = BoundSequential.convert(model,\
				{'same-slope': False, 'zero-lb': False, 'one-lb': False})
		c = torch.eye(num_class).type_as(X)[y].unsqueeze(1) -\
				torch.eye(num_class).type_as(X).unsqueeze(0) 
		I = (~(y.data.unsqueeze(1) ==\
					torch.arange(num_class).type_as(y.data).unsqueeze(0)))
		c = (c[I].view(X.size(0),num_class-1,num_class))

		data_ub = torch.clamp(X + epsilon, max=X.max())
		data_lb = torch.clamp(X - epsilon, min=X.min())
		ub, lb, relu_activity, unstable, dead, alive =\
						model(norm=np.inf, x_U=data_ub, x_L=data_lb, eps=epsilon,\
						C=c, method_opt="interval_range")

		sa = np.zeros((num_class, num_class - 1), dtype = np.int32)
		for i in range(sa.shape[0]):
			for j in range(sa.shape[1]):
				if j < i:
					sa[i][j] = j
				else:
					sa[i][j] = j + 1
		sa = torch.LongTensor(sa)
		lb_s = torch.zeros(X.size(0), num_class)
		sa_labels = sa[y]
		lb = lb_s.scatter(1, sa_labels, lb)
		# print(-lb)
		robust_ce = nn.CrossEntropyLoss()(-lb, y)

		print ("crown loss", robust_ce)
		print ("crown time per sample:", (time.time()-start)/X.shape[0])
		del robust_ce
		print()

	if args.method=="baseline" or args.compare_all:

		
		#if(method == BASELINE):
		start = time.time()

		f = model(X)
		loss = nn.CrossEntropyLoss()(f, y)
		print("baseline loss:", loss)
		print("baseline time per sample:",\
					(time.time()-start)/X.shape[0])
		print() 
		

	if args.method == "naive" or args.compare_all:
		start = time.time()

		iloss, ierr = naive_interval_analyze(model, epsilon,\
					X, y, use_cuda)

		#iloss.backward()
		#print(model[0].weight.grad.sum())

		print ("naive loss:", iloss)
		print ("naive err:", ierr)
		print ("naive time per sample:",\
					(time.time()-start)/X.shape[0])
		del iloss, ierr
		print()


	if args.method == "pgd" or args.compare_all:
		start = time.time()
		attack_iter = 20
		eps_step = args.epsilon/3.

		if args.norm == "linf":
			madry_ord = np.inf
		elif args.norm == "l2":
			madry_ord = 2
		elif args.norm == "l1":
			madry_ord = 1

		adv_x = perturb_iterative(X, y, model, attack_iter,\
					args.epsilon, eps_step, torch.nn.CrossEntropyLoss(),
					  delta_init=None, ord=madry_ord,
					  clip_min=X.min(), clip_max=X.max()
					  )
		adv_out = model(adv_x)
		# print(adv_out)
		pgd_loss = torch.nn.CrossEntropyLoss()(adv_out, y)

		pgd_err = (adv_out.max(1)[1]!=y).sum().item()/y.shape[0]

		print ("pgd loss:", pgd_loss)
		print ("pgd err:", pgd_err)
		print ("pgd time per sample:",\
					(time.time()-start)/X.shape[0])
		del pgd_loss, pgd_err
		print()
		

	if args.method == "sym" or args.compare_all:	
		
		start = time.time()

		iloss, ierr = sym_interval_analyze(model, epsilon,\
						X, y, use_cuda, parallel=PARALLEL,\
						proj=args.proj, norm=args.norm)

		#iloss.backward()
		#print(model[0].weight.grad.sum())

		print ("sym loss:", iloss)
		print ("sym err:", ierr)
		print("sym time per sample:",\
					(time.time()-start)/X.shape[0])
		del iloss, ierr
		print()

	if args.method == "gen" or args.compare_all:	
		
		start = time.time()

		iloss, ierr = gen_interval_analyze(model, [epsilon, epsilon],\
						X, y, use_cuda, parallel=PARALLEL,\
						proj=args.proj, norm=["l2", "l1"])

		#iloss.backward()
		#print(model[0].weight.grad.sum())

		print ("sym loss:", iloss)
		print ("sym err:", ierr)
		print("sym time per sample:",\
					(time.time()-start)/X.shape[0])
		del iloss, ierr
		print()

	if args.method == "mix" or args.compare_all:
		start = time.time()

		iloss, ierr = mix_interval_analyze(model, epsilon,\
						X, y, use_cuda, parallel=PARALLEL,\
						proj=args.proj, norm=args.norm)

		#iloss.backward()
		#print(model[0].weight.grad.sum())

		print ("mix loss:", iloss)
		print ("mix err:", ierr)
		print("mix time per sample:",\
					(time.time()-start)/X.shape[0])
		del iloss, ierr
		print()
		


