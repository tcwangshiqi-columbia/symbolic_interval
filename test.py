'''
Test file for different sound propagation methods
Top contributor: Shiqi Wang
The basic network structures are inspired by the implementations of
Eric Wong's convex polytope github available at:
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
from symbolic_interval.symbolic_network import naive_interval_analyze

from convex_adversarial import robust_loss	

import time


'''Flatten layers.
Please use this Flatten layer to flat the convolutional layers when 
design your own models.
'''
class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)

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

	use_cuda = torch.cuda.is_available()

	MODEL_NAME = "mnist_small_baseline.h5"
	model = mnist_model()
	
	if(not use_cuda):
		model = torch.load(MODEL_NAME, map_location={'cuda:0': 'cpu'})[0]
	else:
		model = torch.load(MODEL_NAME)[0]

	#model = toy_model()
	epsilon = 0.1
	batch_size = 10
	PARALLEL = False

	if(not use_cuda):
		PARALLEL = False

	train_loader, test_loader = mnist_loaders(batch_size)	
	
	for i, (X,y) in enumerate(test_loader):
		
		if(use_cuda):
			X = X.cuda()
			y = y.cuda().long()
			model.cuda()

		if(PARALLEL):
			# The first run with parallel will take a few seconds
			# to warm up.
			eric_loss, eric_err = robust_loss(model,\
						 epsilon, X, y, parallel=PARALLEL)
			del eric_loss, eric_err

		#if(method == ERIC_DUAL):
		start = time.time()
		eric_loss, eric_err = robust_loss(model,\
				epsilon, X, y,parallel=True,\
				bounded_input={0, 1})
				#norm_type="l1_median", proj=20)
		#eric_loss, eric_err = robust_loss(model,\
					#epsilon, X, y, parallel=PARALLEL)

		print ("eric loss", eric_loss)
		print ("eric err:", eric_err)
		print ("eric time per sample:", (time.time()-start))
		del eric_loss, eric_err
		print()


		#if(method == BASELINE):
		start = time.time()

		f = model(X)
		loss = nn.CrossEntropyLoss()(f, y)
		print("baseline loss:", loss)
		print("baseline time per sample:",\
					(time.time()-start)/X.shape[0])
		print() 
		

		#if(method == NAIVE_INTERVAL):
		start = time.time()

		iloss, ierr = naive_interval_analyze(model, epsilon,\
					X, y, use_cuda)

		print ("naive loss:", iloss)
		print ("naive err:", ierr)
		print ("naive time per sample:",\
					(time.time()-start)/X.shape[0])
		del iloss, ierr
		print()
		

		#if(method == SYM_INTERVAL):	
		start = time.time()

		iloss, ierr = sym_interval_analyze(model, epsilon,\
						X, y, use_cuda, parallel=PARALLEL)
			

		print ("sym loss:", iloss)
		print ("sym err:", ierr)
		print("sym time per sample:",\
					(time.time()-start))
		del iloss, ierr
		print()


		start = time.time()

		iloss, ierr = sym_interval_analyze(model, epsilon,\
					X, y, use_cuda, parallel=PARALLEL, proj=395)
			

		print ("sym loss:", iloss)
		print ("sym err:", ierr)
		print("sym time per sample:",\
					(time.time()-start))
		del iloss, ierr

		exit()





