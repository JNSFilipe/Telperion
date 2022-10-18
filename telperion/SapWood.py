import numpy as np

from HeartWood import HeartWood

class SapWood:
  def __init__(self, min_samples_split=2, max_depth=5):
    self.min_samples_split = min_samples_split
    self.max_depth = max_depth

    self.root = None

  def __sow(self, x, y, curr_depth=1, prev_impurity=5000, metric='gini', leaf=False):
    n_rows, n_cols = x.shape
    if self.max_depth <= 1:
      leaf = True
        
    # Check to see if a node should be leaf node
    if n_rows >= self.min_samples_split:
      # Create and train new node
      node = HeartWood(n_cols, leaf=leaf)
      node.fit(x, y)
      y_hat = node.fastforward(x)
      y_hat_idx = y_hat.astype(bool)

      curr_impurity = node._impurity(y[y_hat_idx], y[~y_hat_idx], metric=metric)
      # If the split isn't pure
      # if prev_impurity-curr_impurity > 0: # TODO Make this earl stop mechanism work. It should 
      if curr_depth + 1 <= self.max_depth:
        leaf = curr_depth + 1 == self.max_depth
        
        # Build a tree on the left
        node.true_branch = self.__sow(
            x[y_hat_idx,:], 
            y[y_hat_idx],
            prev_impurity=curr_impurity,
            curr_depth=curr_depth+1,
            leaf=leaf
        )
        node.false_branch = self.__sow(
            x[~y_hat_idx,:], 
            y[~y_hat_idx],
            prev_impurity=curr_impurity,
            curr_depth=curr_depth+1,
            leaf=leaf
        )

      return node

  def fit(self, x, y, metric='gini'):
    self.root = self.__sow(x, y, metric=metric)

  def predict(self, x):
    y_hat = self.root.predict(x)
    return y_hat

  def print_tree(self, tree=None, indent=" "):
    if not tree:
      tree = self.root

    if tree.leaf:
      x = np.zeros(1)
      print("X_"+str(np.squeeze(tree.W)), ">", tree.k)
      print("%s True:\t" % (indent), end="")
      print(tree.true_branch(x))
      print("%s False:\t" % (indent), end="")
      print(tree.false_branch(x))
    else:
      print("X_"+str(np.squeeze(tree.W)), ">", tree.k)
      print("%s True:\t" % (indent), end="")
      self.print_tree(tree.true_branch, indent + indent)
      print("%s False\t:" % (indent), end="")
      self.print_tree(tree.false_branch, indent + indent)

  def __prune(self):
    pass


if __name__ == '__main__':

  x = np.random.uniform(low=0, high=1, size=(50000,3))
  # y = np.array([1 if (j-0.5)**2 + (i-0.5)**2 < 0.2 else 0 for i, j in zip(x[:,0],x[:,1])])
  y = np.array([1 if (j)**2 + (i)**2 < 1 else 0 for i, j in zip(x[:,0],x[:,1])])

  from sklearn.datasets import load_breast_cancer

  data = load_breast_cancer()
  x = data['data']
  y = data['target']

  x = np.random.uniform(low=0, high=1, size=(10000,2))
  y = np.array([1 if (j)**2 + (i)**2 < 0.7 else 0 for i, j in zip(x[:,0],x[:,1])])

  x = np.random.uniform(low=0, high=1, size=(10000,2))
  y = np.array([1 if (j-0.5)**2 + (i-0.5)**2 < 0.1 else 0 for i, j in zip(x[:,0],x[:,1])])

  # x = np.array([
  #     [1, 12, 0],
  #     [1, 87, 1],
  #     [0, 44, 0],
  #     [1, 19, 2],
  #     [0, 32, 1],
  #     [0, 14, 0],
  #   ])
  # y = np.array([1, 1, 0, 0, 1, 1])

  # x = np.random.uniform(low=0, high=1, size=(50000,5))
  # y = np.array([1 if j < i - 0 else 0 for i, j in zip(x[:,0],x[:,1])])

  sw = SapWood(max_depth=6)
  sw.fit(x,y)

  y_hat = sw.predict(x)

  sw.print_tree()

  print('Done!')