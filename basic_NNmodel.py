import numpy as np

# we should make a class that has our layer concepts
class FCNetLayer():
	def __init__(self,in_size):
		#print('INPUT SIZE',in_size)
		# upon initialization we generate weight array
		#self.wsize = (in_size[1],num_neur)
		self.wsize = in_size
		print('W SIZE',self.wsize)
		self.ws = np.random.normal(0,1,self.wsize) # using a normal dist
		self.bs = np.random.normal(0,1,self.wsize[1])
		self.preamp = np.zeros((1,self.wsize[1]))
		self.output = np.zeros((1,self.wsize[1]))

	def fire(self,x_in):
		# forward probagation
		self.preamp = np.matmul(x_in,self.ws)+self.bs
		self.output = 1/(1+np.exp(-self.preamp))
		return self.output

	def dfire(self,xin):
		# change the dfire first
		# only use after firing
		if np.sum(self.output) == 0:
			print('layer has not been fired yet')
		# a different form of derivative of sigmoid
		out = self.output*(1-self.output)

		return out

	def show_data(self):
		print('weights size: ',self.ws.shape)

# # here is a single 1 layer thing
# # insize = (1,4)
# x0 = np.ones((1,4))
# lay1 = FCNetLayer(10,(4,10))
# lay2 = FCNetLayer(20,(10,20))
# print('lay2 wiehts',lay2.ws.shape)
# yhat = lay1.fire(x0)
# print('first layer',yhat)
# yhat = lay2.fire(yhat)
# print('second layer',yhat)

# Here is an example that has multiple layers
# create an input vector
x0 = np.ones((1,4))
y0 = (0.5)

nlist = [4,4,5,10,20,50,20,10,5,1]
w_sizes = [(a,b) for a,b in zip(nlist[0:-1],nlist[1::])]
print(w_sizes)

# create the other layers in the network
layers = [FCNetLayer(A) for A in w_sizes]
print('connection points: {}'.format(len(layers)))

# now we want to get the firing for each point
acts = []
act = layers[0].fire(x0)
acts.append(act)
#now forward propogate through the network. -- this could be our firing function
for lay in layers[1::]:
	act = lay.fire(act)
	acts.append(act)
	print('layer output shape',act.shape)
print(len(acts))

print(layers[1].ws)

eta = 10
nabla_b = [np.zeros(lay.bs.shape) for lay in layers]
nabla_w = [np.zeros(lay.ws.shape) for lay in layers]

batch_size = 1 # we chose only 1 example
def cost(activity,y):
	# changing this changes the minimization
	return (activity-y)

derr = cost(layers[-1].output,y0)
# get the error from output
delta =  derr * layers[-1].dfire(layers[-1].preamp)
print('delta',delta.shape,delta)
nabla_b[-1] = delta
# acti is transpose of weights thats why I assign transpose
print('acti shape:{}'.format(layers[-2].output.shape))
nabla_w[-1] = np.matmul(delta,layers[-2].output).T

for j in range(2,len(layers)):
	delta = np.matmul(delta,layers[-j+1].ws.T) * layers[-j].dfire(layers[-j].preamp)
	print('new delta {}'.format(delta.shape))
	nabla_b[-j] = delta
	nabla_w[-j] = np.matmul(delta.T,layers[-j-1].output).T


# backprop complete, update the weights
# in this case they are little deltas
# so at the begining we have a zeros nabla_b and w,
# as the minibatch continues we add our delta nabla
# to them then update the w and b
