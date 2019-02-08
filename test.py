import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import setproctitle
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from Symb_interval.interval import Interval, Symbolic_interval
from Symb_interval.symbolic_network import Interval_network
from Symb_interval import robust_loss	
import time


class Interval_Dense(nn.Module):
	def __init__(self, layer):
		nn.Module.__init__(self)
		self.layer = layer
		#print ("linear:", self.layer.weight.shape)

	def forward(self, ix):
		
		if(isinstance(ix, Symbolic_interval)):
			c = ix.c
			idep = ix.idep
			edep = ix.edep
			#print (ix.c.shape, self.layer.weight.shape)
			ix.c = F.linear(c, self.layer.weight, bias=self.layer.bias)

			ix.idep = F.linear(idep, self.layer.weight)
			ix.edep = F.linear(edep, self.layer.weight)
			ix.shape = list(ix.c.shape[1:])
			ix.n = list(ix.c[0].reshape(-1).size())[0]

			ix.concretize()

			return ix

		if(isinstance(ix, Interval)):
			c = ix.c
			e = ix.e
			#print (ix.c.shape, self.layer.weight.shape)
			c = F.linear(c, self.layer.weight, bias=self.layer.bias)
			e = F.linear(e, self.layer.weight.abs())
			#return F.linear(x, self.layer.weight, bias=self.layer.bias)
			ix.update_lu(c-e, c+e)
			return ix




class Interval_Conv2d(nn.Module):
	def __init__(self, layer):
		nn.Module.__init__(self)
		self.layer = layer
		#print ("conv2d:", self.layer.weight.shape)

	def forward(self, ix):
		
		if(isinstance(ix, Symbolic_interval)):
			ix.shrink()
			c = ix.c
			idep = ix.idep
			edep = ix.edep
			ix.c = F.conv2d(c, self.layer.weight, 
						   stride=self.layer.stride,
						   padding=self.layer.padding, 
						   bias=self.layer.bias)
			ix.idep = F.conv2d(idep, self.layer.weight, 
						   stride=self.layer.stride,
						   padding=self.layer.padding)
			ix.edep = F.conv2d(edep, self.layer.weight, 
						   stride=self.layer.stride,
						   padding=self.layer.padding)
			ix.shape = list(ix.c.shape[1:])
			ix.n = list(ix.c[0].reshape(-1).size())[0]

			ix.concretize()

			return ix

		if(isinstance(ix, Interval)):
			c = ix.c
			e = ix.e
			c = F.conv2d(c, self.layer.weight, 
						   stride=self.layer.stride,
						   padding=self.layer.padding, 
						   bias=self.layer.bias)
			e = F.conv2d(e, self.layer.weight.abs(), 
						   stride=self.layer.stride,
						   padding=self.layer.padding)
			ix.update_lu(c-e, c+e)
			return ix



class Interval_ReLU(nn.Module):
	def __init__(self, layer):
		nn.Module.__init__(self)
		self.layer = layer

	def forward(self, ix):
		if(isinstance(ix, Symbolic_interval)):
			lower = ix.l
			upper = ix.u
			appr_condition = ((lower<0) * (upper>0)).type(torch.Tensor)

			mask = appr_condition*((upper)/(upper-lower+0.000001))
			mask = mask + 1 - appr_condition
			mask = mask*(upper>0).type(torch.Tensor)
			m = int(appr_condition.sum())
			ix.mask.append(mask[0])

			appr = (appr_condition*mask)[0]

			appr_ind = (appr).nonzero()

			appr_err = (appr*(-lower[0]))/2.0

			error_row = torch.zeros((m, ix.n)).scatter_(1,\
						appr_ind, appr_err[appr_ind])

			ix.c = ix.c*mask+appr_err.reshape(-1)
			ix.edep = ix.edep*mask
			ix.idep = ix.idep*mask
			ix.edep = torch.cat((ix.edep, error_row), 0)

			return ix

		if(isinstance(ix, Interval)):
			lower = ix.l
			upper = ix.u
			appr_condition = ((lower<0) * (upper>0)).type(torch.Tensor)

			mask = appr_condition*((upper)/(upper-lower+0.000001))
			mask = mask + 1 - appr_condition
			mask = mask*(upper>0).type(torch.Tensor)
			ix.mask.append(mask)
			ix.update_lu(F.relu(ix.l), F.relu(ix.u))
			return ix
		


class Interval_Flatten(nn.Module):
	def __init__(self):
		nn.Module.__init__(self)

	def forward(self, ix):
		if(isinstance(ix, Symbolic_interval)):
			ix.extend()
			return ix
		if(isinstance(ix, Interval)):
			ix.update_lu(ix.l.view(ix.l.size(0), -1),\
				ix.u.view(ix.u.size(0), -1))
			return ix


class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)

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


class Interval_network(nn.Module):
	def __init__(self, model):
		nn.Module.__init__(self)
		#self.net = [layer for layer in model]
		
		self.net = []
		for layer in model:
			if(isinstance(layer, nn.Linear)):
				self.net.append(Interval_Dense(layer))
			if(isinstance(layer, nn.ReLU)):
				self.net.append(Interval_ReLU(layer))
			if(isinstance(layer, nn.Conv2d)):
				self.net.append(Interval_Conv2d(layer))
			if 'Flatten' in (str(layer.__class__.__name__)): 
				self.net.append(Interval_Flatten())
		
	def forward(self, ix):
		for i, layer in enumerate(self.net):
			ix = layer(ix)

		return ix


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

	method = NAIVE_INTERVAL
	epsilon = 0.1
	train_loader, test_loader = mnist_loaders(5)

	#print (model[5].weight.shape)

	inet = Interval_network(model)

	for i, (X,y) in enumerate(test_loader):
		start = time.time()
		robust_loss(model, epsilon, X, y)
		print ("eric time:", time.time()-start)
		#print (X.shape, X.min(),X.max())
		start = time.time()
		print("concrete propagation:", model(X))
		print("baseline time:", time.time()-start)

		if(method == NAIVE_INTERVAL):
			ix = Interval(torch.clamp(X[0:1,:]-epsilon, 0.0, 1.0),\
					torch.clamp(X[0:1,:]+epsilon, 0.0, 1.0))

			ix = (inet(ix))
			print (ix.mask)
		
		if(method == SYM_INTERVAL):
			start = time.time()
			for xi, yi in zip(X,y):
				six = Symbolic_interval(torch.clamp(xi.unsqueeze(0)-epsilon,0.0,1.0),\
							torch.clamp(xi.unsqueeze(0)+epsilon,0.0,1.0))
				ix = inet(six)
				print (ix.worst_case(yi.unsqueeze(0)))
				# You can get the mask for each layer in ix.mask
				#print (ix.mask)

			print("sym time:", time.time()-start)

		exit()





