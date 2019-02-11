'''
Interval networks and symbolic interval propagations.
Top contributor: Shiqi Wang
The basic network structures are inspired by the implementations of
Eric Wong's convex polytope github available at:
https://github.com/locuslab/convex_adversarial

Usage: 
for symbolic interval anlysis:
	from symbolic_interval.symbolic_network import sym_interval_analyze
for naive interval analysis:
	from symbolic_interval.symbolic_network import sym_interval_analyze
'''

from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .interval import Interval, Symbolic_interval


class Interval_network(nn.Module):
	'''Convert a nn.Sequential model to a network support symbolic
	interval propagations/naive interval propagations.
	'''
	def __init__(self, model):
		nn.Module.__init__(self)

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

	'''Forward intervals for each layer.

	* :attr:`ix` is the input fore each layer. If ix is a naive
	interval, it will propagate naively. If ix is a symbolic interval,
	it will propagate symbolicly.
	'''
	def forward(self, ix):
		for i, layer in enumerate(self.net):
			ix = layer(ix)

		return ix


class Interval_Dense(nn.Module):
	def __init__(self, layer):
		nn.Module.__init__(self)
		self.layer = layer

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
			if(ix.use_cuda):
				appr_condition = ((lower<0) * (upper>0)).type(\
							torch.Tensor).cuda()
			else:
				appr_condition = ((lower<0) * (upper>0)).type(\
							torch.Tensor)

			mask = appr_condition*((upper)/(upper-lower+0.000001))
			mask = mask + 1 - appr_condition

			if(ix.use_cuda):
				mask = mask*((upper>0).type(torch.Tensor).cuda())
			else:
				mask = mask*(upper>0).type(torch.Tensor)

			m = int(appr_condition.sum())
			ix.mask.append(mask[0])

			appr = (appr_condition*mask)[0]

			appr_ind = (appr).nonzero()

			appr_err = (appr*(-lower[0]))/2.0

			if(ix.use_cuda):
				error_row = torch.zeros((m, ix.n))
				error_row = error_row.cuda()
				if(m!=0):
					error_row = error_row.scatter_(1,\
						appr_ind, appr_err[appr_ind])
			else:
				
				error_row = torch.zeros((m, ix.n))
				if(m!=0):
					error_row = error_row.scatter_(1,\
						appr_ind, appr_err[appr_ind])

			ix.c = ix.c*mask+appr_err.reshape(-1)
			ix.edep = ix.edep*mask
			ix.idep = ix.idep*mask
			ix.edep = torch.cat((ix.edep, error_row), 0)

			return ix

		if(isinstance(ix, Interval)):
			lower = ix.l
			upper = ix.u

			if(ix.use_cuda):
				appr_condition = ((lower<0) * (upper>0)).type(\
							torch.Tensor).cuda()
			else:
				appr_condition = ((lower<0) * (upper>0)).type(\
							torch.Tensor)

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


'''Naive interval propagations.

Args:
	model: regular nn.Sequential models
	epsilon: desired input ranges
	X and y: samples and lables

Return:
	iloss: robust loss provided by naive interval analysis
	ierr: verifiable robust error provided by naive interval analysis
'''
def naive_interval_analyze(model, epsilon, X, y, use_cuda=False):

	# Transfer original model to interval models
	inet = Interval_network(model)

	ix = Interval(torch.clamp(X-epsilon, 0.0, 1.0),\
					torch.clamp(X+epsilon, 0.0, 1.0))

	ix = (inet(ix))
	wc =  ix.worst_case(y)
	#print (wc)
	#print (ix.mask)

	iloss = nn.CrossEntropyLoss()(wc, y)
	ierr = (wc.max(1)[1]!=y).type(torch.Tensor)
	ierr = ierr.sum().item()/X.shape[0]

	print ("naive avg width per label:", wc.sum()/X.shape[0]/10)

	return iloss, ierr


'''Symbolic interval propagations.

Args:
	model: regular nn.Sequential models
	epsilon: desired input ranges
	X and y: samples and lables

Return:
	iloss: robust loss provided by symbolic interval analysis
	ierr: verifiable robust error provided by symbolic interval analysis
'''
def sym_interval_analyze(model, epsilon, X, y, use_cuda=False):

	# Transfer original model to interval models
	inet = Interval_network(model)

	iloss = 0
	ierr = 0
	width_per_label = 0

	minimum = 0.0
	maximum = 1.0

	for xi, yi in zip(X,y):

		ix = Symbolic_interval(\
			   torch.clamp(xi.unsqueeze(0)-epsilon, minimum, maximum),\
			   torch.clamp(xi.unsqueeze(0)+epsilon, minimum, maximum),\
			   use_cuda
		     )
		#ix = Symbolic_interval(xi.unsqueeze(0)-epsilon,\
			#xi.unsqueeze(0)+epsilon)

		ix = inet(ix)

		# wc is the worst case output returned by symbolic 
		# interval analysis
		wc = (ix.worst_case(yi.unsqueeze(0)))

		width_per_label += wc.sum()

		# You can get the mask for each layer in ix.mask
		#print (ix.mask)

		iloss += nn.CrossEntropyLoss()(wc, yi.unsqueeze(0))
		ierr += (wc.max(1)[1]!=yi).type(torch.Tensor)

	iloss /= X.shape[0]
	ierr /= X.shape[0]

	print ("sym avg width per label:", width_per_label/X.shape[0]/10)
	if(use_cuda): 
		#iloss = iloss.cpu().data.numpy()
		ierr = ierr.cpu().data.numpy()[0]
	else:
		#iloss = iloss.data.numpy()
		ierr = ierr.data.numpy()[0]

	return iloss, ierr




