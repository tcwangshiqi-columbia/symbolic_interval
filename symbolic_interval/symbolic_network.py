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
from .interval import Symbolic_interval_proj1, Symbolic_interval_proj2
import time


class Interval_network(nn.Module):
	'''Convert a nn.Sequential model to a network support symbolic
	interval propagations/naive interval propagations.
	'''
	def __init__(self, model):
		nn.Module.__init__(self)

		self.net = []
		first_layer = True
		for layer in model:
			if(isinstance(layer, nn.Linear)):
				self.net.append(Interval_Dense(layer, first_layer))
				first_layer = False
			if(isinstance(layer, nn.ReLU)):
				self.net.append(Interval_ReLU(layer))
			if(isinstance(layer, nn.Conv2d)):
				self.net.append(Interval_Conv2d(layer, first_layer))
				first_layer = False
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
	def __init__(self, layer, first_layer=False):
		nn.Module.__init__(self)
		self.layer = layer
		self.first_layer = first_layer

	def forward(self, ix):
		
		if(isinstance(ix, Symbolic_interval)):
			c = ix.c
			idep = ix.idep
			edep = ix.edep
			#print (ix.c.shape, self.layer.weight.shape)
			ix.c = F.linear(c, self.layer.weight, bias=self.layer.bias)
			ix.idep = F.linear(idep, self.layer.weight)
			for i in range(len(edep)):
				ix.edep[i] = F.linear(edep[i], self.layer.weight)
			ix.shape = list(ix.c.shape[1:])
			ix.n = list(ix.c[0].view(-1).size())[0]
			ix.concretize()
			return ix

		if(isinstance(ix, Symbolic_interval_proj1)):
			c = ix.c
			idep = ix.idep
			edep = ix.edep

			ix.c = F.linear(c, self.layer.weight, bias=self.layer.bias)
			ix.idep = F.linear(idep, self.layer.weight)
			ix.idep_proj = F.linear(ix.idep_proj, self.layer.weight.abs())

			for i in range(len(edep)):
				ix.edep[i] = F.linear(edep[i], self.layer.weight)
			ix.shape = list(ix.c.shape[1:])
			ix.n = list(ix.c[0].view(-1).size())[0]
			ix.concretize()
			return ix

		if(isinstance(ix, Symbolic_interval_proj2)):
			c = ix.c
			idep = ix.idep
			edep = ix.edep

			ix.c = F.linear(c, self.layer.weight, bias=self.layer.bias)
			ix.idep = F.linear(idep, self.layer.weight)
			ix.idep_proj = F.linear(ix.idep_proj, self.layer.weight.abs())

			ix.edep = F.linear(ix.edep, self.layer.weight.abs())

			ix.shape = list(ix.c.shape[1:])
			ix.n = list(ix.c[0].view(-1).size())[0]
			ix.concretize()
			return ix

		if(isinstance(ix, Interval)):
			c = ix.c
			e = ix.e
			c = F.linear(c, self.layer.weight, bias=self.layer.bias)
			e = F.linear(e, self.layer.weight.abs())
			#print("naive e", e)
			#print("naive c", c)
			ix.update_lu(c-e, c+e)
			
			return ix


class Interval_Conv2d(nn.Module):
	def __init__(self, layer, first_layer=False):
		nn.Module.__init__(self)
		self.layer = layer
		self.first_layer = first_layer
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

			for i in range(len(edep)):
				ix.edep[i] = F.conv2d(edep[i], self.layer.weight, 
						   stride=self.layer.stride,
						   padding=self.layer.padding)
			ix.shape = list(ix.c.shape[1:])
			ix.n = list(ix.c[0].reshape(-1).size())[0]
			ix.concretize()
			return ix

		if(isinstance(ix, Symbolic_interval_proj1)):
			#print(ix.idep.shape)
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
			ix.idep_proj = F.conv2d(ix.idep_proj, self.layer.weight.abs(), 
						   stride=self.layer.stride,
						   padding=self.layer.padding)

			for i in range(len(edep)):
				ix.edep[i] = F.conv2d(edep[i], self.layer.weight, 
						   stride=self.layer.stride,
						   padding=self.layer.padding)
			ix.shape = list(ix.c.shape[1:])
			ix.n = list(ix.c[0].reshape(-1).size())[0]
			ix.concretize()
			return ix

		if(isinstance(ix, Symbolic_interval_proj2)):
			#print(ix.idep.shape)
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
			ix.idep_proj = F.conv2d(ix.idep_proj, self.layer.weight.abs(), 
						   stride=self.layer.stride,
						   padding=self.layer.padding)

			ix.edep = F.conv2d(ix.edep, self.layer.weight.abs(), 
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
		#print(ix.u)
		#print(ix.l)
		if(isinstance(ix, Symbolic_interval)):
			lower = ix.l
			upper = ix.u
			#print("sym u", upper)
			#print("sym l", lower)

			appr_condition = ((lower<0)*(upper>0)).detach()
			mask = (lower>0).type_as(lower)
			mask[appr_condition] = upper[appr_condition]/\
					(upper[appr_condition]-lower[appr_condition]).detach()

			m = int(appr_condition.sum().item())

			appr_ind = appr_condition.view(-1,ix.n).nonzero()

			appr_err = mask*(-lower)/2.0

			if (m!=0):

				if(ix.use_cuda):
					error_row = torch.zeros((m, ix.n), device=lower.get_device())
				else:
					error_row = torch.zeros((m, ix.n))

				error_row = error_row.scatter_(1,\
					appr_ind[:,1,None], appr_err[appr_condition][:,None])

				edep_ind = lower.new(appr_ind.size(0),lower.size(0)).zero_()
				edep_ind = edep_ind.scatter_(1, appr_ind[:,0][:,None], 1)

			ix.c = ix.c*mask+appr_err*appr_condition.type_as(lower)

			for i in range(len(ix.edep)):
				ix.edep[i] = ix.edep[i] * ix.edep_ind[i].mm(mask)

			ix.idep = ix.idep*mask.view(ix.batch_size, 1, ix.n)

			if(m!=0):

				ix.edep = ix.edep + [error_row]
				ix.edep_ind = ix.edep_ind + [edep_ind]

			return ix


		if(isinstance(ix, Symbolic_interval_proj1)):
			lower = ix.l
			upper = ix.u
			#print("sym u", upper)
			#print("sym l", lower)

			appr_condition = ((lower<0)*(upper>0)).detach()
			mask = (lower>0).type_as(lower)
			mask[appr_condition] = upper[appr_condition]/\
					(upper[appr_condition]-lower[appr_condition]).detach()

			m = int(appr_condition.sum().item())
			appr_ind = appr_condition.view(-1,ix.n).nonzero()

			appr_err = mask*(-lower)/2.0
			
			if (m!=0):

				if(ix.use_cuda):
					error_row = torch.zeros((m, ix.n), device=lower.get_device())
				else:
					error_row = torch.zeros((m, ix.n))

				error_row = error_row.scatter_(1,\
						appr_ind[:,1,None], appr_err[appr_condition][:,None])

				edep_ind = lower.new(appr_ind.size(0),lower.size(0)).zero_()
				edep_ind = edep_ind.scatter_(1, appr_ind[:,0][:,None], 1)

			ix.c = ix.c*mask+appr_err*appr_condition.type_as(lower)

			for i in range(len(ix.edep)):
				ix.edep[i] = ix.edep[i] * ix.edep_ind[i].mm(mask)

			ix.idep = ix.idep*mask.view(ix.batch_size, 1, ix.n)
			ix.idep_proj = ix.idep_proj*mask.view(ix.batch_size, ix.n)

			if (m!=0):
				ix.edep = ix.edep + [error_row]
				ix.edep_ind = ix.edep_ind + [edep_ind]

			return ix


		if(isinstance(ix, Symbolic_interval_proj2)):
			lower = ix.l
			upper = ix.u
			#print("sym u", upper)
			#print("sym l", lower)
			appr_condition = ((lower<0)*(upper>0)).detach()
			mask = (lower>0).type_as(lower)
			m = int(appr_condition.sum().item())
			#appr_ind = appr_condition.view(-1,ix.n).nonzero()

			appr_err = upper/2.0
			appr_err = appr_err * (appr_condition.type_as(appr_err))

			ix.edep = ix.edep * mask
			ix.edep = ix.edep + appr_err

			ix.c = ix.c * mask + appr_err

			ix.idep = ix.idep * mask.view(ix.batch_size, 1, ix.n)
			ix.idep_proj = ix.idep_proj * mask.view(ix.batch_size, ix.n)

			return ix


		if(isinstance(ix, Interval)):
			lower = ix.l
			upper = ix.u

			#print("naive", upper)

			if(ix.use_cuda):
				appr_condition = ((lower<0) * (upper>0)).type(\
					torch.Tensor).cuda(device=ix.c.get_device())
			else:
				appr_condition = ((lower<0) * (upper>0)).type(\
							torch.Tensor)

			mask = appr_condition*((upper)/(upper-lower+0.000001))
			mask = mask + 1 - appr_condition
			if(ix.use_cuda):
				mask = mask*((upper>0).type(torch.Tensor).\
					cuda(device=ix.c.get_device()))
			else:
				mask = mask*(upper>0).type(torch.Tensor)
			ix.mask.append(mask)
			#print(ix.e.shape)
			ix.update_lu(F.relu(ix.l), F.relu(ix.u))
			return ix


class Interval_Flatten(nn.Module):
	def __init__(self):
		nn.Module.__init__(self)

	def forward(self, ix):
		if(isinstance(ix, Symbolic_interval) or\
				isinstance(ix, Symbolic_interval_proj1) or\
				isinstance(ix, Symbolic_interval_proj2)):
			ix.extend()
			return ix
		if(isinstance(ix, Interval)):
			ix.update_lu(ix.l.view(ix.l.size(0), -1),\
				ix.u.view(ix.u.size(0), -1))
			return ix


class Interval_Bound(nn.Module):
	def __init__(self, net, epsilon, method="sym",\
					proj=None, use_cuda=True):
		nn.Module.__init__(self)
		self.net = net
		self.epsilon = epsilon
		self.use_cuda = use_cuda
		self.proj = proj
		if(proj is not None):
			assert proj>0, "project dimension has to be larger than 0"

		assert method in ["sym", "naive"],\
				"No such interval methods!"
		self.method = method
			

	def forward(self, X, y):
		minimum = X.min()
		maximum = X.max()
		out_features = self.net[-1].out_features

		# Transfer original model to interval models
		inet = Interval_network(self.net)

		# Create symbolic inteval classes from X
		if(self.method == "naive"):
			ix = Interval(torch.clamp(X-self.epsilon, minimum, maximum),\
					torch.clamp(X+self.epsilon, minimum, maximum),\
					self.use_cuda\
			 	)

		if(self.method == "sym"):
			if(self.proj is None):
				ix = Symbolic_interval(\
					   torch.clamp(X-self.epsilon, minimum, maximum),\
					   torch.clamp(X+self.epsilon, minimum, maximum),\
					   self.use_cuda\
					 )
			else:
				input_size = list(X[0].reshape(-1).size())[0]
				if(self.proj>(input_size/2)):
					ix = Symbolic_interval_proj1(\
						   torch.clamp(X-self.epsilon, minimum, maximum),\
						   torch.clamp(X+self.epsilon, minimum, maximum),\
						   self.proj,\
						   self.use_cuda\
						 )
				else:
					ix = Symbolic_interval_proj2(\
						   torch.clamp(X-self.epsilon, minimum, maximum),\
						   torch.clamp(X+self.epsilon, minimum, maximum),\
						   self.proj,\
						   self.use_cuda\
						 )

		# Propagate symbolic interval through interval networks
		ix = inet(ix)
		#print(ix.u)
		#print(ix.l)

		# Calculate the worst case outputs
		wc = ix.worst_case(y, out_features)
		#print(wc)

		return wc


'''Naive interval propagations.

Args:
	model: regular nn.Sequential models
	epsilon: desired input ranges
	X and y: samples and lables

Return:
	iloss: robust loss provided by naive interval analysis
	ierr: verifiable robust error provided by naive interval analysis
'''
def naive_interval_analyze(net, epsilon, X, y,\
					use_cuda=True, parallel=False):

	# Transfer original model to interval models

	if(parallel):
		wc = nn.DataParallel(Interval_Bound(net, epsilon,
				method="naive", use_cuda=use_cuda))(X, y)
	else:
		wc = Interval_Bound(net, epsilon, method="naive",\
						use_cuda=use_cuda)(X, y)

	iloss = nn.CrossEntropyLoss()(wc, y)
	ierr = (wc.max(1)[1]!=y).type(torch.Tensor)
	ierr = ierr.sum().item()/X.shape[0]

	return iloss, ierr


'''Symbolic interval propagations.

Args:
	model: regular nn.Sequential models
	epsilon: desired input ranges
	X and y: samples and lables
	use_cuda: whether we want to use cuda
	parallel: whehter we want to run on multiple gpus

Return:
	iloss: robust loss provided by symbolic interval analysis
	ierr: verifiable robust error provided by symbolic
	interval analysis
'''
def sym_interval_analyze(net, epsilon, X, y,\
					use_cuda=True, parallel=False, proj=None):

	if(parallel):
		wc = nn.DataParallel(Interval_Bound(net, epsilon,\
					method="sym", proj=proj, use_cuda=use_cuda))(X, y)
	else:
		wc = Interval_Bound(net, epsilon,\
					method="sym", proj=proj, use_cuda=use_cuda)(X, y)

	iloss = nn.CrossEntropyLoss()(wc, y)
	ierr = (wc.max(1)[1] != y)
	ierr = ierr.sum().item()/X.size(0)
	
	return iloss, ierr




