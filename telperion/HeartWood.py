import numpy as np
from tqdm import tqdm
from collections.abc import Iterable
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class HeartWood(BaseEstimator, ClassifierMixin):
  def __init__(self, classical_backend='sklearn', lr=0.1, leaf=True):
    self.classical_backend = classical_backend
    self.lr = lr
    
    self.W = None
    self.k = None

    self.true_branch   = lambda X: 1
    self.false_branch  = lambda X: 0

    self.leaf = leaf

    self._linear_combo = True

    # TODO manage estimator tags (https://scikit-learn.org/stable/developers/develop.html#estimator-tags)
    # TODO implement Pipeline compatibility (https://scikit-learn.org/stable/developers/develop.html#pipeline-compatibility)
    # TODO implement GridSearch compatibility (https://scikit-learn.org/stable/developers/develop.html#parameters-and-init)

  def __call__(self, X):
    return self.forward(X)

  ## Mathematical rewrite of a tree
  def __f(self, X, derivative=False):
    if derivative=='W':
      # y = X*self.__sigmoid(np.dot(X, self.W) - self.k, derivative=True)
      y = X*self.__tanh(np.dot(X, self.W) - self.k, derivative=True)
    elif derivative=='b':
      # y = -1*self.__sigmoid(np.dot(X, self.W) - self.k, derivative=True)
      y = -1*self.__tanh(np.dot(X, self.W) - self.k, derivative=True)
    else:
      # y = self.__sigmoid(np.dot(X, self.W) - self.k, derivative=False)
      y = self.__tanh(np.dot(X, self.W) - self.k, derivative=False)
    return y

  ## General Utils
  def accuracy(self, X, y):
    y_hat = self.forward(X)
    return np.mean(y_hat==y)

  ## Prediction Functions
  def fastforward(self, X):
    return np.squeeze((np.dot(X, self.W) > self.k).astype(float))

  def forward(self, X):
    y_hat = self.fastforward(X).astype(bool).tolist()
    if not isinstance(y_hat, Iterable):
      y_hat = [y_hat]
      X = np.expand_dims(X, axis=0)
    pred = [self.true_branch(X[i,:]) if y_hat[i] else self.false_branch(X[i,:]) for i in range(len(y_hat))]
    return np.array(pred)

  def predict(self, X):
    # Check if fit has been called
    check_is_fitted(self)
    # Input validation
    X = check_array(X)
    return self.forward(X)

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

  def __get_best_unidimensonal_split(self, X, y, metric='gini', backend='sklearn'):
    
    num_feat = X.shape[1]

    if backend == 'sklearn' or backend == 'sklearn-dt':
      clf = DecisionTreeClassifier(criterion=metric, max_depth=1)
      clf.fit(X, y)
      self.W = np.zeros((num_feat,1))
      self.W[clf.tree_.feature[0], 0] = 1.0
      self.k = clf.tree_.threshold[0]
    elif backend == 'telperion':
      max_info_gain = -float('inf')
      best_split = ()

      parent_imp = self.__info_metrics(y, metric=metric)

      for f in range(num_feat):

        split_candiadates = np.unique(X[:,f])
        
        for sc in split_candiadates:
          y_hat  = X[:,f]>sc
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
    else:
      raise Exception("ERROR: Unvalid backend specified")

  ## Backpropagation Training Functions
  def __next_batch(self, X, y, batchSize):
    for i in np.arange(0, X.shape[0], batchSize):
      yield (X[i:i + batchSize], y[i:i + batchSize])
      
  def __lr(self, iter, total_iter, fixed=False):
    if fixed:
      return self.lr
    else:
      return int(self.lr*total_iter)/(iter+total_iter)
  
  def __sigmoid(self, X, derivative=False):
    if derivative:
      return (np.exp(-X))/((np.exp(-X)+1)**2)
    return 1/(1 + np.exp(-X))
    
  def __tanh(self, X, derivative=False):
    if derivative:
      return 0.5/(np.cosh(X)**2)
    else:
      return (np.tanh(X)+1)/2
    
  def __mse(self, y, y_hat):
    return np.mean((y-y_hat)**2)
    
  def __backprop(self, X, y, lr=None):
    if lr == None:
      lr = self.lr
    
    y_hat = self.__f(X)
    y = y.reshape(y_hat.shape)
    c = self.__mse(y, y_hat)
    e = y - y_hat
    
    # repeat until convergence{
    # 	w = w - (learning_rate * (dJ/dw))
    # 	b = b - (learning_rate * (dJ/db))
    # }
    
    dW = -2*np.dot(self.__f(X, derivative='W').T, e) / y.shape[0]
    db = -2*np.dot(self.__f(X, derivative='b').T, e) / y.shape[0]
    
    self.W -= lr * dW
    self.k -= lr * db
    
    return c

  def __train(self, X, y, batch_size=128, epochs=50):
    epoch_loss = []
    epoch_accuracy = []
    pbar = tqdm(np.arange(0, epochs))
    for epoch in pbar:
      batch_loss = []
      batch_accuracy = []
      
      lr = self.__lr(epoch, epochs, fixed=True)
      
      for x_batch, y_batch in self.__next_batch(X, y, batch_size):
        
        loss = self.__backprop(x_batch, y_batch, lr=lr)
        batch_loss.append(loss)
        batch_accuracy.append(self.accuracy(x_batch, y_batch))
        
      epoch_loss.append(np.array(batch_loss).mean())
      epoch_accuracy.append(np.array(batch_accuracy).mean())
      # print("Epoch {}: Loss: {:.6f} Accuracy: {:.2f}".format(epoch+1, epoch_loss[-1], epoch_accuracy[-1]*100))
      pbar.set_description("|| E: {} | L: {:.6f} | A: {:.2f}% ||".format(epoch+1, epoch_loss[-1], epoch_accuracy[-1]*100))
    
  def fit(self, X, y, batch_size=128, epochs=50, metric='gini'):
    
    # Check that X and y have correct shape
    X, y = check_X_y(X, y)

    # Store the classes seen during fit
    self.classes_ = unique_labels(y)

    num_feat=X.shape[1]

    self.W = np.random.randn(num_feat, 1) * np.sqrt(1./1)
    self.k = np.random.normal()

    parent_imp = self.__info_metrics(y, metric=metric)

    self.__get_best_unidimensonal_split(X, y, metric=metric, backend=self.classical_backend)
    W_back = self.W.copy()
    k_back = self.k
    y_hat = self.fastforward(X).astype(bool)
    trues  = y[y_hat].copy()
    falses = y[~y_hat].copy()
    uni_info_gain = parent_imp - self._impurity(trues, falses, metric=metric)

    self.__train(X, y, batch_size=batch_size, epochs=epochs)
    y_hat = self.fastforward(X).astype(bool)
    trues  = y[y_hat].copy()
    falses = y[~y_hat].copy()

    multi_info_gain = -float('inf')
    if len(trues) > 0 and len(falses) > 0:
      multi_info_gain = parent_imp - self._impurity(trues, falses, metric=metric)

    if uni_info_gain > multi_info_gain:
      self._linear_combo = False
      self.W = W_back
      self.k = k_back

    if self.leaf:
      y_hat = self.fastforward(X).astype(bool)
      if len(y[y_hat]) > 0:
        true_value = round(y[y_hat].mean())
      else:
        true_value = 1.0

      if len(y[~y_hat]) > 0:
        false_value = round(y[~y_hat].mean())
      else:
        false_value = 0.0
      self.true_branch  = lambda v: true_value
      self.false_branch = lambda v: false_value

    # Return the classifier
    return self


if __name__ == '__main__':
  np.random.seed(76)
  X = np.random.uniform(low=0, high=1, size=(10000,2))
  y = np.array([1 if j < i - 0 else 0 for i, j in zip(X[:,0],X[:,1])])

  hw = HeartWood()
  hw.fit(X, y, 128, 100)

  from sklearn.metrics import accuracy_score
  print('Final Accuracy: {:.2f}%'.format(accuracy_score(y, hw.predict(X))*100))

  print('Done!')