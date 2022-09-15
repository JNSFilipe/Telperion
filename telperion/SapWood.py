import numpy as np

from HeartWood import HeartWood

class CrossEntropy:
    def __init__(self):
        pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)
    

class SapWood:
  def __init__(self, num_feat, n_estimators=5, lr=0.1, zero=0, one=1):

    self.estimators = [HeartWood(num_feat) for _ in range(n_estimators)]

    self.loss = CrossEntropy()

    self.learning_rate = lr
    self.n_estimators = n_estimators
    self.estimators = [HeartWood(num_feat) for _ in range(n_estimators)]

  def __initial_guess(self, y):
    log_odds = np.log(len(y)/y.sum())
    return np.exp(log_odds)/(np.exp(log_odds) + 1)

    
  def fit(self, x, y):
    y_pred = np.full(np.shape(y), np.mean(y, axis=0))
    for i in range(self.n_estimators):
        gradient = self.loss.gradient(y, y_pred)
        self.estimators[i].fit(x, gradient)
        update = self.estimators[i].f(x)
        # Update y prediction
        y_pred -= np.squeeze(np.multiply(self.learning_rate, update))

    
  def predict(self, x):
    pass
    # y_hat = self.__heartwood(x)
    
    # if y_hat == 0:
    # 	return self.zero
    # else:
    # 	return self.one

if __name__ == '__main__':
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score

    x = np.random.uniform(low=0, high=1, size=(50000,5))
    y = np.array([1 if (j-0.5)**2 + (i-0.5)**2 < 0.2 else 0 for i, j in zip(x[:,0],x[:,1])])

    sw = SapWood(x.shape[1], n_estimators=50, lr=0.2)

    sw.fit(x, y)

    m = GradientBoostingClassifier(n_estimators=50)
    m.fit(x,y)
    print(accuracy_score(m.predict(x), y))