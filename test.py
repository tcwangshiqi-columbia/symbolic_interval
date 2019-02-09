import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import setproctitle
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from Symb_interval.interval import Interval, Symbolic_interval
from Symb_interval.symbolic_network import *
from convex_adversarial import robust_loss	
import time


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

def mnist_loaders(batch_size, shuffle_test=False): 
	mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
	mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
	train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=shuffle_test, pin_memory=True)
	return train_loader, test_loader

NAIVE_INTERVAL = 0
SYM_INTERVAL = 1
CROWN = 2


if __name__ == '__main__':
	MODEL_NAME = "mnist_small_baseline.h5"
	model = mnist_model()
	model = torch.load(MODEL_NAME, map_location={'cuda:0': 'cpu'})[0]

	#method = NAIVE_INTERVAL
	epsilon = 0.1
	train_loader, test_loader = mnist_loaders(10)

	
	for i, (X,y) in enumerate(test_loader):
		
		#if(method == ERIC_DUAL):
		start = time.time()
		eric_bound, eric_loss, eric_err =  robust_loss(model, epsilon, X, y)

		print ("eric avg width per label", eric_bound.sum()/X.shape[0]/10)
		print ("eric loss", eric_loss)
		print ("eric time per sample:", (time.time()-start)/X.shape[0])
		print()
		
		#if(method == BASELINE):
		start = time.time()

		f = model(X)
		#print("concrete propagation:", model(X))
		loss = nn.CrossEntropyLoss()(f, y)
		print("baseline loss:", loss)
		print("baseline time per sample:", (time.time()-start)/X.shape[0])
		print() 

		#if(method == NAIVE_INTERVAL):
		start = time.time()

		iloss, ierr = naive_interval_analyze(model, epsilon, X, y)

		print ("naive loss:", iloss)
		print ("naive err:", ierr)
		print ("naive time per sample:", (time.time()-start)/X.shape[0])
		print()

		#if(method == SYM_INTERVAL):
			
		start = time.time()

		inet = Interval_network(model)
		iloss, ierr = sym_interval_analyze(model, epsilon, X, y)

		print ("sym loss:", iloss)
		print ("sym err:", ierr)
		print("sym time per sample:", (time.time()-start)/X.shape[0])
		
		exit()





