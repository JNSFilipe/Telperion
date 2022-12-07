import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from HeartWood import HeartWood

class SapWood(BaseEstimator, ClassifierMixin):
  def __init__(self, min_samples_split=2, max_depth=5, imp_metric='gini', classical_backend='sklearn', lr=0.1):
    self.min_samples_split = min_samples_split
    self.classical_backend = classical_backend
    self.imp_metric = imp_metric
    self.max_depth = max_depth
    self.lr = lr

    self.root = None

    # TODO manage estimator tags (https://scikit-learn.org/stable/developers/develop.html#estimator-tags)
    # TODO implement Pipeline compatibility (https://scikit-learn.org/stable/developers/develop.html#pipeline-compatibility)
    # TODO implement GridSearch compatibility (https://scikit-learn.org/stable/developers/develop.html#parameters-and-init)

  def __sow(self, X, y, curr_depth=1, prev_impurity=5000, metric='gini', leaf=False, classical_backend='sklearn', lr=0.1):
    n_rows, n_cols = X.shape
    if self.max_depth <= 1:
      leaf = True
        
    # Check to see if a node should be leaf node
    if n_rows >= self.min_samples_split:
      # Create and train new node
      node = HeartWood(classical_backend=classical_backend, lr=lr, leaf=leaf)
      node.fit(X, y)
      y_hat = node.fastforward(X)
      y_hat_idx = y_hat.astype(bool)

      curr_impurity = node._impurity(y[y_hat_idx], y[~y_hat_idx], metric=metric)
      # If the split isn't pure
      # if prev_impurity-curr_impurity > 0: # TODO Make this earl stop mechanism work. It should 
      if curr_depth + 1 <= self.max_depth:
        leaf = curr_depth + 1 == self.max_depth
        
        # Build a tree on the left
        # TODO Implement min_sample_protection more elegantly
        if len(y[y_hat_idx].tolist()) >= self.min_samples_split:
          node.true_branch = self.__sow(
              X[y_hat_idx,:], 
              y[y_hat_idx],
              prev_impurity=curr_impurity,
              curr_depth=curr_depth+1,
              leaf=leaf,
              metric=metric,
              classical_backend=classical_backend,
              lr=lr
          )

        if len(y[~y_hat_idx].tolist()) >= self.min_samples_split:
          node.false_branch = self.__sow(
              X[~y_hat_idx,:], 
              y[~y_hat_idx],
              prev_impurity=curr_impurity,
              curr_depth=curr_depth+1,
              leaf=leaf,
              metric=metric,
              classical_backend=classical_backend,
              lr=lr
          )

      return node

  def fit(self, X, y):
    # Check that X and y have correct shape
    X, y = check_X_y(X, y)

    # Store the classes seen during fit
    self.classes_ = unique_labels(y)

    self.root = self.__sow(X, y, metric=self.imp_metric, classical_backend=self.classical_backend, lr=self.lr)

  def predict(self, X):
    # Check if fit has been called
    check_is_fitted(self)
    # Input validation
    X = check_array(X)
    return self.root.predict(X).squeeze()

  def print_tree(self, tree=None, indent=" "):
    if not tree:
      tree = self.root

    # TODO Fix this for the case where only one of the branches is leaf
    if tree.leaf:
      X = np.zeros(1)
      print("X_"+str(np.squeeze(tree.W)), ">", tree.k)
      print("%s True:\t" % (indent), end="")
      print(tree.true_branch(X))
      print("%s False:\t" % (indent), end="")
      print(tree.false_branch(X))
    else:
      print("X_"+str(np.squeeze(tree.W)), ">", tree.k)
      print("%s True:\t" % (indent), end="")
      self.print_tree(tree.true_branch, indent + indent)
      print("%s False\t:" % (indent), end="")
      self.print_tree(tree.false_branch, indent + indent)

  def __prune(self):
    pass


if __name__ == '__main__':

  X = np.random.uniform(low=0, high=1, size=(50000,3))
  # y = np.array([1 if (j-0.5)**2 + (i-0.5)**2 < 0.2 else 0 for i, j in zip(X[:,0],X[:,1])])
  y = np.array([1 if (j)**2 + (i)**2 < 1 else 0 for i, j in zip(X[:,0],X[:,1])])

  from sklearn.datasets import load_breast_cancer

  data = load_breast_cancer()
  X = data['data']
  y = data['target']

  X = np.random.uniform(low=0, high=1, size=(10000,2))
  y = np.array([1 if (j)**2 + (i)**2 < 0.7 else 0 for i, j in zip(X[:,0],X[:,1])])

  X = np.random.uniform(low=0, high=1, size=(10000,2))
  y = np.array([1 if (j-0.5)**2 + (i-0.5)**2 < 0.1 else 0 for i, j in zip(X[:,0],X[:,1])])


  x = np.random.uniform(low=0, high=1, size=(10000,2))
  y = np.array([1 if (j-0.5)**2 + (i-0.5)**2 < 0.1 else 0 for i, j in zip(x[:,0],x[:,1])])
  y = np.array([1 if (j)**2 + (i)**2 < 0.7 else 0 for i, j in zip(x[:,0],x[:,1])])
  # X = np.array([
  #     [1, 12, 0],
  #     [1, 87, 1],
  #     [0, 44, 0],
  #     [1, 19, 2],
  #     [0, 32, 1],
  #     [0, 14, 0],
  #   ])
  # y = np.array([1, 1, 0, 0, 1, 1])

  # X = np.random.uniform(low=0, high=1, size=(50000,5))
  # y = np.array([1 if j < i - 0 else 0 for i, j in zip(X[:,0],X[:,1])])

  sw = SapWood(max_depth=3)
  sw.fit(X,y)

  y_hat = sw.predict(X)

  from sklearn.metrics import accuracy_score
  print('Final Accuracy: {:.2f}%'.format(accuracy_score(y, sw.predict(X))*100))

  # sw.print_tree()

  print('Done!')