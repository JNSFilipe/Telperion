import numpy as np

from HeartWood import HeartWood

class SapWood:
  def __init__(self, min_samples_split=2, max_depth=5):
    self.min_samples_split = min_samples_split
    self.max_depth = max_depth

    self.root = None

  def __info_metrics(self, y, metric='gini'):
    p = y.sum()/len(y)
    if metric == 'gini':
      gini = 1-np.sum(p**2)
      return gini
    if metric == 'entropy':
      entropy = np.sum(-p*np.log2(p+1e-9))
      return entropy

  def __impurity(self, trues, falses, metric='gini'):
    wt = len(trues) / (len(trues)+len(falses))
    wf = len(falses) / (len(trues)+len(falses))
    
    return (wt * self.__info_metrics(trues, metric=metric) + wf * self.__info_metrics(falses, metric=metric))

  def __sow(self, x, y, curr_depth=0, prev_impurity=5000, metric='gini'):
    n_rows, n_cols = x.shape
        
    # Check to see if a node should be leaf node
    if n_rows >= self.min_samples_split and curr_depth <= self.max_depth:
      # Create and train new node
      node = HeartWood(n_cols)
      node.fit(x, y)
      y_hat = node.fastforward(x)
      y_hat_idx = y_hat.astype(bool)

      curr_impurity = self.__impurity(y[y_hat_idx], y[~y_hat_idx], metric=metric)
      # If the split isn't pure
      if prev_impurity-curr_impurity > 0:

        # Build a tree on the left
        node.true_branch = self.__sow(
            x[y_hat_idx,:], 
            y[y_hat_idx],
            prev_impurity=curr_impurity,
            curr_depth = curr_depth + 1
        )
        node.false_branch = self.__sow(
            x[~y_hat_idx,:], 
            y[~y_hat_idx],
            prev_impurity=curr_impurity,
            ccurr_depth=curr_depth + 1
        )

        return node

  def fit(self, x, y, metric='gini'):
    self.root = self.__sow(x, y, metric='gini')


if __name__ == '__main__':

  x = np.random.uniform(low=0, high=1, size=(50000,5))
  y = np.array([1 if (j-0.5)**2 + (i-0.5)**2 < 0.2 else 0 for i, j in zip(x[:,0],x[:,1])])

  x = np.array([
      [1, 12, 0],
      [1, 87, 1],
      [0, 44, 0],
      [1, 19, 2],
      [0, 32, 1],
      [0, 14, 0],
    ])
  y = np.array([1, 1, 0, 0, 1, 1])

  x = np.random.uniform(low=0, high=1, size=(50000,5))
  y = np.array([1 if j < i - 0 else 0 for i, j in zip(x[:,0],x[:,1])])

  sw = SapWood()
  sw.fit(x,y)