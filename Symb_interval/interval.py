import numpy as np
import torch

class Interval():
	def __init__(self, lower, upper):
		assert not ((upper-lower)<0).any(), "upper less than lower"
		self.l = lower
		self.u = upper
		self.c = (lower+upper)/2
		self.e = (upper-lower)/2
		self.mask = []

	def update_lu(self, lower, upper):
		assert not ((upper-lower)<0).any(), "upper less than lower"
		self.l = lower
		self.u = upper
		self.c = (lower+upper)/2
		self.e = (upper-lower)/2

	def update_ce(self, center, error):
		assert not (error<0).any(), "upper less than lower"
		self.c = center
		self.e = error
		self.u = self.c+self.e
		self.l = self.c+self.e

	def __str__(self):
		string = "interval shape:"+str(self.c.shape)
		string += "\nlower:"+str(self.l)
		string += "\nupper:"+str(self.u)
		return string
	
	def worst_case(self, y):
		assert y.shape[0] == self.l.shape[0] == self.u.shape[0], "wrong input shape"
		u = torch.zeros(self.u.shape)
		for i in range(y.shape[0]):
			t = self.l[i, y[i]]
			u[i] = self.u[i]-t
			u[i, y[i]] = 0.0
		return u	


class Symbolic_interval(Interval):
	def __init__(self, lower, upper):
		Interval.__init__(self, lower, upper)
		self.shape = list(self.c.shape[1:])
		self.n = list(self.c[0].reshape(-1).size())[0]

		self.idep = torch.eye(self.n)*self.e.reshape(-1,1)
		self.edep = torch.zeros((1, self.n))
		

	def concretize(self):
		self.extend()
		#print(self.idep.abs().sum(dim=0).shape, self.edep.abs().sum(dim=0).shape)
		e  = self.idep.abs().sum(dim=0)+self.edep.abs().sum(dim=0)
		self.l = self.c - e
		self.u = self.c + e
		return self

	def extend(self):
		self.c = self.c.reshape(-1, self.n)
		self.idep = self.idep.reshape(-1, self.n)
		self.edep = self.edep.reshape(-1, self.n)

	def shrink(self):
		#print(self.shape)
		self.c = self.c.reshape(tuple([-1]+self.shape))
		self.idep = self.idep.reshape(tuple([-1]+self.shape))
		self.edep = self.edep.reshape(tuple([-1]+self.shape))

	def worst_case(self, y):
		assert y.shape[0] == self.l.shape[0] == self.u.shape[0], "wrong input shape"
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



		


if __name__ == '__main__':
	
	i = Interval(np.arange(10), np.arange(10)+1)
	print (i)
	i.u[9]=0
	print (i)