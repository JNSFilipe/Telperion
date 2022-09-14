import numpy as np

class HeartWood:
	def __init__(self, num_feat, lr=0.1):
		self.lr = lr
		
		self.W = np.random.randn(num_feat, 1) * np.sqrt(1./1)
		self.k = np.random.normal()
		
	def f(self, x, derivative=False):
		if derivative=='W':
			# y = x*self.__sigmoid(np.dot(x, self.W) - self.k, derivative=True)
			y = x*self.__tanh(np.dot(x, self.W) - self.k, derivative=True)
		elif derivative=='b':
			# y = -1*self.__sigmoid(np.dot(x, self.W) - self.k, derivative=True)
			y = -1*self.__tanh(np.dot(x, self.W) - self.k, derivative=True)
		else:
			# y = self.__sigmoid(np.dot(x, self.W) - self.k, derivative=False)
			y = self.__tanh(np.dot(x, self.W) - self.k, derivative=False)
		return y
		
	def forward(self, x):
		return np.squeeze((np.dot(x, self.W) > self.k).astype(float))
		
	def __next_batch(self, x, y, batchSize):
		for i in np.arange(0, x.shape[0], batchSize):
			yield (x[i:i + batchSize], y[i:i + batchSize])
			
	def __lr(self, iter, total_iter, fixed=False):
		if fixed:
			return self.lr
		else:
			return int(self.lr*total_iter)/(iter+total_iter)
	
	def __sigmoid(self, x, derivative=False):
		if derivative:
			return (np.exp(-x))/((np.exp(-x)+1)**2)
		return 1/(1 + np.exp(-x))
		
	def __tanh(self, x, derivative=False):
		if derivative:
			return 0.5/(np.cosh(x)**2)
		else:
			return (np.tanh(x)+1)/2
		
	def __mse(self, y, y_hat):
		return np.mean((y-y_hat)**2)
		
	def __backprop(self, x, y, lr=None):
		if lr == None:
			lr = self.lr
		
		y_hat = self.f(x)
		y = y.reshape(y_hat.shape)
		c = self.__mse(y, y_hat)
		e = y - y_hat
		
		# repeat until convergence{
		# 	w = w - (learning_rate * (dJ/dw))
		# 	b = b - (learning_rate * (dJ/db))
		# }
		
		#d = e * self.f(x, derivative=True)
		dW = -2*np.dot(self.f(x, derivative='W').T, e) / y.shape[0]
		db = -2*np.dot(self.f(x, derivative='b').T, e) / y.shape[0]
		
		self.W -= lr * dW
		self.k -= lr * db
		
		return c
		
	def accuracy(self, x, y):
		y_hat = self.forward(x)
		return np.mean(y_hat==y)
		
	def fit(self, x, y, batch_size, epochs):
		epoch_loss = []
		epoch_accuracy = []
		for epoch in np.arange(0, epochs):
			batch_loss = []
			batch_accuracy = []
			
			lr = self.__lr(epoch, epochs, fixed=True)
			
			for x_batch, y_batch in self.__next_batch(x, y, batch_size):
				
				loss = self.__backprop(x_batch, y_batch, lr=lr)
				batch_loss.append(loss)
				batch_accuracy.append(self.accuracy(x_batch, y_batch))
				
			epoch_loss.append(np.array(batch_loss).mean())
			epoch_accuracy.append(np.array(batch_accuracy).mean())
			print("Epoch {}: Loss: {:.6f} Accuracy: {:.2f}".format(epoch+1, epoch_loss[-1], epoch_accuracy[-1]*100))