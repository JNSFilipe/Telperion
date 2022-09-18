import numpy as np

from tqdm import tqdm
class HeartWood:
  def __init__(self, num_feat, lr=0.1):
    self.lr = lr
    
    self.W = np.random.randn(num_feat, 1) * np.sqrt(1./1)
    self.k = np.random.normal()

    self.true_branch   = lambda: 1
    self.false_branch  = lambda: 0

    self.multidim = True

  ## Mathematical rewrite of a tree
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

  ## General Utils
  def accuracy(self, x, y):
    y_hat = self.forward(x)
    return np.mean(y_hat==y)

  ## Prediction Functions
  def fastforward(self, x):
    return np.squeeze((np.dot(x, self.W) > self.k).astype(float))

  def forward(self, x):
    y_hat = self.fastforward(x)
    return np.array([self.true_branch() if i==1 else self.false_branch() for i in y_hat])

  ## Traditional Training Functions
  def __info_metrics(self, y, metric='gini'):
    p = y.sum()/len(y)
    if metric == 'gini':
      gini = 1-np.sum(p**2)
      return gini
    if metric == 'entropy':
      entropy = np.sum(-p*np.log2(p+1e-9))
      return entropy

  def _impurity(self, trues, falses, metric='gini'):
    wt = len(trues) / (len(trues)+len(falses))
    wf = len(falses) / (len(trues)+len(falses))
    
    return (wt * self.__info_metrics(trues, metric=metric) + wf * self.__info_metrics(falses, metric=metric))

  def __get_best_unidimensonal_split(self, x, y, metric='gini'):
    max_info_gain = -float('inf')
    best_split = ()

    num_feat = x.shape[1]

    parent_imp = self.__info_metrics(y, metric=metric)

    for f in range(num_feat):

      split_candiadates = np.unique(x[:,f])
      
      for sc in split_candiadates:
        y_hat  = x[:,f]>sc
        trues  = y[y_hat]
        falses = y[~y_hat]

        if len(trues) > 0 and len(falses) > 0:
          info_gain = parent_imp - self._impurity(trues, falses, metric=metric)
          if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_split = (f,sc)

    self.W = np.zeros((num_feat,1))
    self.W[best_split[0], 0] = 1.0
    self.k = best_split[1]

  ## Backpropagation Training Functions
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

  def __train(self, x, y, batch_size=128, epochs=50):
    epoch_loss = []
    epoch_accuracy = []
    pbar = tqdm(np.arange(0, epochs))
    for epoch in pbar:
      batch_loss = []
      batch_accuracy = []
      
      lr = self.__lr(epoch, epochs, fixed=True)
      
      for x_batch, y_batch in self.__next_batch(x, y, batch_size):
        
        loss = self.__backprop(x_batch, y_batch, lr=lr)
        batch_loss.append(loss)
        batch_accuracy.append(self.accuracy(x_batch, y_batch))
        
      epoch_loss.append(np.array(batch_loss).mean())
      epoch_accuracy.append(np.array(batch_accuracy).mean())
      # print("Epoch {}: Loss: {:.6f} Accuracy: {:.2f}".format(epoch+1, epoch_loss[-1], epoch_accuracy[-1]*100))
      pbar.set_description("E: {} | L: {:.6f} | A: {:.2f}\% ||".format(epoch+1, epoch_loss[-1], epoch_accuracy[-1]*100))
    
  def fit(self, x, y, batch_size=128, epochs=50, metric='gini'):

    parent_imp = self.__info_metrics(y, metric=metric)

    self.__get_best_unidimensonal_split(x, y, metric=metric)
    W_back = self.W.copy()
    k_back = self.k
    y_hat = self.fastforward(x).astype(bool)
    trues  = y[y_hat]
    falses = y[~y_hat]
    uni_info_gain = parent_imp - self._impurity(trues, falses, metric=metric)

    self.__train(x, y, batch_size=batch_size, epochs=epochs)
    y_hat = self.fastforward(x).astype(bool)
    trues  = y[y_hat]
    falses = y[~y_hat]

    multi_info_gain = -float('inf')
    if len(trues) > 0 and len(falses) > 0:
      multi_info_gain = parent_imp - self._impurity(trues, falses, metric=metric)

    if uni_info_gain > multi_info_gain:
      self.multidim = False
      self.W = W_back
      self.k = k_back

if __name__ == '__main__':
  x = np.random.uniform(low=0, high=1, size=(50000,5))
  y = np.array([1 if j < i - 0 else 0 for i, j in zip(x[:,0],x[:,1])])

  hw = HeartWood(x.shape[1])
  hw.fit(x, y, 128, 100)