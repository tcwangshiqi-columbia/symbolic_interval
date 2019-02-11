'''
Interval class definitions
Top contributor: Shiqi Wang
'''

from __future__ import print_function

import numpy as np
import torch


class Interval():
	'''Naive interval class

	Naive interval propagation is low-cost (only around two times slower 
	than regular NN propagation). However, the output range provided is 
	loose. This is because the dependency of inputs are ignored.
	See ReluVal https://arxiv.org/abs/1804.10829 for more details of
	the tradeoff.

	Naive interval propagation are used for many existing training
	schemes:
	(1) DiffAi: http://proceedings.mlr.press/v80/mirman18b/mirman18b.pdf
	(2) IBP: https://arxiv.org/pdf/1810.12715.pdf
	These training schemes are fast but the robustness of trained models
	suffers from the loose estimations of naive interval propagation.
	
	Args:
		lower: numpy matrix of the lower bound for each layer nodes
		upper: numpy matrix of the upper bound for each layer nodes
		lower and upper should have the same shape of input for 
		each layer
		no upper value should be less than corresponding lower value

	* :attr:`l` and `u` keeps the upper and lower values of the
	  interval. Naive interval propagation using them to propagate.

	* :attr:`c` and `e` means the center point and the error range 
	  of the interval. Symbolic interval propagation using to propagate
	  since it can keep the dependency more efficiently. 

	* :attr:`mask` is used to keep the estimation information for each
	  hidden node. It has the same shape of the ReLU layer input. 
	  for each hidden node, before going through ReLU, let [l,u] denote
	  a ReLU's input range. It saves the value u/(u-l), which is the
	  slope of estimated output dependency. 0 means, given the input
	  range, this ReLU's input will always be negative and the output 
	  is always 0. 1 indicates, it always stays positive and the
	  output will not change. Otherwise, this node is estimated during 
	  interval propagation and will introduce overestimation error. 
	'''
	def __init__(self, lower, upper, use_cuda=False):
		assert not ((upper-lower)<0).any(), "upper less than lower"
		self.l = lower
		self.u = upper
		self.c = (lower+upper)/2
		self.e = (upper-lower)/2
		self.mask = []
		self.use_cuda = use_cuda

	def update_lu(self, lower, upper):
		'''Update this interval with new lower and upper numpy matrix

		Args:
			lower: numpy matrix of the lower bound for each layer nodes
			upper: numpy matrix of the upper bound for each layer nodes
		'''
		assert not ((upper-lower)<0).any(), "upper less than lower"
		self.l = lower
		self.u = upper
		self.c = (lower+upper)/2
		self.e = (upper-lower)/2

	def update_ce(self, center, error):
		'''Update this interval with new error and center numpy matrix

		Args:
			lower: numpy matrix of the lower bound for each layer nodes
			upper: numpy matrix of the upper bound for each layer nodes
		'''
		assert not (error<0).any(), "upper less than lower"
		self.c = center
		self.e = error
		self.u = self.c+self.e
		self.l = self.c+self.e

	def __str__(self):
		'''Print function
		'''
		string = "interval shape:"+str(self.c.shape)
		string += "\nlower:"+str(self.l)
		string += "\nupper:"+str(self.u)
		return string
	
	def worst_case(self, y):
		'''Calculate the wrost case of the analyzed output ranges.
		In details, it returns the upper bound of other label minus 
		the lower bound of the target label. If the returned value is 
		less than 0, it means the worst case provided by interval
		analysis will never be larger than the target label y's. 
		'''
		assert y.shape[0] == self.l.shape[0] == self.u.shape[0],\
				"wrong input shape"
		u = torch.zeros(self.u.shape)
		if(self.use_cuda): u = u.cuda()
		for i in range(y.shape[0]):
			t = self.l[i, y[i]]
			u[i] = self.u[i]-t
			u[i, y[i]] = 0.0
		return u	


class Symbolic_interval(Interval):
	'''Symbolic interval class

	Symbolic interval analysis is a state-of-the-art tight output range 
	analyze method. It captured the dependencies ignored by naive
	interval propagation. As the tradeoff, the cost is much higher than
	naive interval and regular propagations. To maximize the tightness,
	symbolic linear relaxation is used. More details can be found in 
	Neurify: https://arxiv.org/pdf/1809.08098.pdf

	There are several similar methods which can provide close tightness
	(1) Convex polytope: https://arxiv.org/abs/1711.00851
	(2) FastLin: https://arxiv.org/abs/1804.09699
	(3) DeepZ: https://files.sri.inf.ethz.ch/website/papers/DeepZ.pdf
	This lib implements symbolic interval analysis, which can provide
	one of the tightest and most efficient analysis among all these 
	methods.

	Symbolic interval analysis is used to verifiably robust train the
	networks in MixTrain, providing state-of-the-art efficiency and 
	verifiable robustness. See https://arxiv.org/abs/1811.02625 for more
	details.
	Similar training methods include:
	(1) Scaling defense: https://arxiv.org/abs/1805.12514
	(2) DiffAI: http://proceedings.mlr.press/v80/mirman18b/mirman18b.pdf
	
	* :attr:`shape` is the input shape of ReLU layers.

	* :attr:`n` is the number of hidden nodes in each layer.

	* :attr:`idep` keeps the input dependencies.

	* :attr:`edep` keeps the error dependency introduced by each
	  overestimated nodes.
	'''
	def __init__(self, lower, upper, use_cuda=False):
		assert lower.shape[0]==upper.shape[0]==1, "each symbolic"+\
					"should only contain one sample"
		Interval.__init__(self, lower, upper)
		self.shape = list(self.c.shape[1:])
		self.n = list(self.c[0].reshape(-1).size())[0]

		self.idep = torch.eye(self.n)
		self.edep = torch.zeros((1, self.n))
		self.use_cuda = use_cuda
		if(self.use_cuda):
			self.idep = self.idep.cuda()
			self.edep = self.edep.cuda()
		self.idep *= self.e.reshape(-1,1)
		
	'''Calculating the upper and lower matrix for symbolic intervals.
	To make concretize easier, convolutional layer nodes will be 
	extended first.
	'''
	def concretize(self):
		self.extend()
		#print(self.idep.abs().sum(dim=0).shape,\
				#self.edep.abs().sum(dim=0).shape)
		e  = self.idep.abs().sum(dim=0)+self.edep.abs().sum(dim=0)
		if(self.use_cuda):
			e = e.cuda()
		self.l = self.c - e
		self.u = self.c + e
		return self

	'''Extending convolutional layer nodes to a two-dimensional vector.
	'''
	def extend(self):
		self.c = self.c.reshape(-1, self.n)
		self.idep = self.idep.reshape(-1, self.n)
		self.edep = self.edep.reshape(-1, self.n)

	'''Convert the extended layer back to the shape stored in `shape`.
	'''
	def shrink(self):
		#print(self.shape)
		self.c = self.c.reshape(tuple([-1]+self.shape))
		self.idep = self.idep.reshape(tuple([-1]+self.shape))
		self.edep = self.edep.reshape(tuple([-1]+self.shape))


	'''Calculate the wrost case of the analyzed output ranges.
	Return the upper bound of other output dependency minus target's
	output dependency. If the returned value is less than 0, it means
	the worst case provided by interval analysis will never be larger
	than the target label y's. 
	'''
	def worst_case(self, y):
		if(self.use_cuda):
			y = y.cuda().long()
		assert y.shape[0] == self.l.shape[0] == self.u.shape[0],\
					"wrong input shape"
		#print (self.c.shape, self.idep.shape, self.edep.shape)
		c_t = self.c[:,y]
		self.c = self.c - c_t
		idep_t = self.idep[:, y]
		self.idep = self.idep-idep_t
		edep_t = self.edep[:, y]
		self.edep = self.edep-edep_t
		self.concretize()
		return self.u 



class Crown(Interval):
	def __init__(self, lower, upper):
		Interval.__init__(self, lower, upper)
		self.shape = list(self.c.shape[1:])
		self.n = list(self.c[0].reshape(-1).size())[0]

		self.idep = torch.eye(self.n)*self.e.reshape(-1,1)

		self.edep = torch.zeros((1, self.n))



class Zonotope(Interval):
	def __init__(self, lower, upper):
		Interval.__init__(self, lower, upper)
		raise NotImplementedError


