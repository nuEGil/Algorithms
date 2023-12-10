import numpy as np
import matplotlib.pyplot as plt
# we should make a class that has our layer concepts
##TODO change to Relu activation and cost

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

class Network():
	def __init__(self,layer_list):

		# write down general training params
		self.eta = 0.1 # arbitrailiy picked paramters.
		self.batch_size = 10
		self.epochs = 5

		# generate a list of layer sizes given the number of neurons in each layer
		w_sizes = [(a,b) for a,b in zip(nlist[0:-1],nlist[1::])]
		print(w_sizes)

		# create the other layers in the network
		self.layers = [FCNetLayer(A) for A in w_sizes]
		print('connection points: {}'.format(len(self.layers)))

		print('network initialization complete')

		#current INPUT
		self.currx0 = None

	def evaluate(self,x0):
		self.currx0 = x0
		#print('evaluating network')
		# now we want to get the firing for each point

		act = self.layers[0].fire(x0)
		#now forward propogate through the network. -- this could be our firing function
		# so j starts from 0  goes to len(layers)-1, that's
		# why we need to use j+1 to start at the right spot, input into that layer is j
		# so it works out
		for lay in self.layers[1::]:
			act = lay.fire(act)

		#print('network evaluation complete')
		#print('Network output: {}'.format(self.acts[-1]))
		# give final output
		return self.layers[-1].output

	def cost(self,activity,y):
		# changing this changes the minimization
		return (activity-y)

	def backprop(self,derr):
		# this stuff
		nabla_b = [np.zeros(lay.bs.shape) for lay in self.layers]
		nabla_w = [np.zeros(lay.ws.shape) for lay in self.layers]


		# get the error from output
		delta =  derr * self.layers[-1].dfire(self.layers[-1].preamp)
		#print('delta',delta.shape,delta)
		# first bias and weight update
		nabla_b[-1] = delta
		nabla_w[-1] = np.matmul(delta,self.layers[-2].output).T
		for j in range(2,len(self.layers)):
			delta = np.matmul(delta,self.layers[-j+1].ws.T) * self.layers[-j].dfire(self.layers[-j].preamp)
			#print('new delta {}'.format(delta.shape))
			nabla_b[-j] = delta
			nabla_w[-j] = np.matmul(delta.T,self.layers[-j-1].output).T
		return (nabla_b,nabla_w)

	def train_sgd(self,eta,batch_size,epochs,train_x,train_y):
		self.eta = eta
		self.batch_size = batch_size
		self.epochs = epochs
		derr = []
		errors = []

		print('Parameters updated, Training network now')
		for jj in range(epochs):
			# first dshuffle the training data
			inds = np.random.permutation(train_x.shape[0])
			train_x = train_x[inds] # shuffling all rows in array
			train_y = train_y[inds]
			print('data shuffled \nxshape: {}\yshape:{}'.format(train_x.shape,train_y.shape))

			# itterate through mini batches
			for ii in range(0,train_x.shape[0]-self.batch_size,self.batch_size):
				# reset nablas
				nabla_b = [np.zeros(lay.bs.shape) for lay in self.layers]
				nabla_w = [np.zeros(lay.ws.shape) for lay in self.layers]
				for it in range(ii,ii+self.batch_size):
					# itterate through examples in minibatch
					outt = self.evaluate(train_x[ii,:].reshape(1,train_x[ii,:].size))
					derri = self.cost(outt,train_y[ii])
					# save the derri
					derr.extend(derri)
					errors.extend(0.5*(train_y[ii]-outt)**2)
					# new d_nablas
					dnabb,dnabw = self.backprop(derri)
					nabla_b = [nb+dnb for nb,dnb in zip(nabla_b, dnabb)]
					nabla_w = [nw+dnw for nw,dnw in zip(nabla_w, dnabw)]
				# after the minibatch is done we update ws and bs
				for i,lay in enumerate(self.layers):
					self.layers[i].ws = self.layers[i].ws - (eta/batch_size)*nabla_w[i]
					self.layers[i].bs = self.layers[i].bs - (eta/batch_size)*nabla_b[i]

		derr = np.array(derr)
		errors = np.array(errors)
		plt.figure()
		plt.plot(errors)
		plt.show()

if __name__ == '__main__':
	# list of neuron counts for each layer -- first number is input layer
	nlist = [4,20,50,150,50,20,4,1]
	eta = 0.1 # learning rate
	batch_size = 10 #number of examples to evaluate per weight update
	epochs = 50 # number of times we want to get through the training set.

	samples = 500
	t_ = np.linspace(-50,50,samples)
	t_ = np.reshape(t_,(samples,1))
	t = np.repeat(t_,4,axis=1)

	yout = np.sum(t**2,axis = 1)
	yout = (yout-np.min(yout))/np.max(yout)
	print('ytaining ys shape :{}'.format(yout.shape))
	print(np.min(yout),np.max(yout),yout.shape)

	bNet = Network(nlist)
	#bNet.evaluate(x0)
	bNet.train_sgd(eta,batch_size,epochs,t,yout)

	# # print message for bug checks
	# for j,(layr,act) in enumerate(zip(bNet.layers,bNet.acts)):
		# print('Layer:{}\nSize: {}\nActiv size: {}'.format(j, layr.ws.shape,act.shape))
