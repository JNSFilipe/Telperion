import copy
import torch
from torch import nn, optim
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import LRScheduler, EarlyStopping
from telperion.utils import gini, accuracy, train_nn
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class HeartWood(BaseEstimator, ClassifierMixin):
    def __init__(self, leaf=True):

        self.fc = None
        # self.U = nn.Tanh()
        # self.U = nn.Hardsigmoid()
        self.U = nn.Sigmoid()

        self.true_branch = lambda X: 1
        self.false_branch = lambda X: 0

        self.leaf = leaf

        self._multidim = False

        # TODO manage estimator tags (https://scikit-learn.org/stable/developers/develop.html#estimator-tags)
        # TODO implement Pipeline compatibility (https://scikit-learn.org/stable/developers/develop.html#pipeline-compatibility)
        # TODO implement GridSearch compatibility (https://scikit-learn.org/stable/developers/develop.html#parameters-and-init)
        # TODO implement possibility of using entropy instead of gini to chose between multidim and unidim
        # TODO costumize LRScheduler and EarlyStopping parameters

    def __call__(self, X):
        return self.forward(X)

    def __f(self, X):
        # y = 0.5 * self.U(self.fc(X).squeeze()) + 0.5
        y = self.U(self.fc(X).squeeze())
        return y

    def __F(self, X):
        y = torch.heaviside(self.fc(X).squeeze(), torch.tensor([0.0]))
        return y

    # Prediction Functions
    def forward(self, X):

        y_hat = self.__F(X).squeeze()
        return y_hat

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        if not torch.is_tensor(X):
            X = check_array(X)
            X = torch.Tensor(X)

        return self.forward(X)

    def __train_unidim(self, X, y, metric='gini'):

        num_feat = X.shape[1]

        clf = DecisionTreeClassifier(criterion=metric, max_depth=1)
        clf.fit(X, y)

        w = clf.tree_.feature[0]
        b = clf.tree_.threshold[0]

        wt = torch.zeros(1, num_feat, dtype=torch.float)
        wt[:, w] = 1.0
        self.fc.weight.data = wt

        bt = torch.tensor([-b], dtype=torch.float)
        self.fc.bias.data = bt

    # Backpropagation Training Functions
    def __train_multidim(self, X, y, lr=0.01, batch_size=128, epochs=50, verbose=0, backend='skorch'):

        opt = optim.Adam
        loss = nn.CrossEntropyLoss
        if backend == 'skorch':

            m = NeuralNetBinaryClassifier(
                self.fc, max_epochs=epochs, batch_size=batch_size, lr=lr, optimizer=opt,
                callbacks=[LRScheduler(), EarlyStopping()])
            m.fit(X, y)

        else:
            self.fc = train_nn(self.fc, self.U, X, y, optimizer=opt, loss=loss,
                               lr=lr, epochs=epochs, batch_size=batch_size, verbose=verbose)

        if verbose > 2:
            print(
                f"Final Accuracy: {(self.__F(X)==y).sum()/len(y)*100.0:.2f}%")

    def fit(self, X, y, lr=0.01, batch_size=128, epochs=50, metric='gini', method='both', backend='skorch', verbose=0):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y, )

        if not torch.is_tensor(X):
            X = torch.Tensor(X)

        if not torch.is_tensor(y):
            y = torch.Tensor(y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        num_feat = X.shape[1]
        unidim_impurity = 1.0

        self.fc = nn.Linear(num_feat, 1, bias=True)

        if method == 'unidim' or method == 'both':
            self.__train_unidim(X, y, metric=metric)

        if method == 'both':
            y_pred = self.predict(X)
            unidim_impurity = gini(y, y_pred)
            unidim_fc = copy.deepcopy(self.fc)
            if verbose > 1:
                print("Unidim Accuracy: {:.2f}%".format(accuracy(y, y_pred)))

        if method == 'multidim' or method == 'both':
            self.__train_multidim(
                X, y, lr=lr, batch_size=batch_size, epochs=epochs, backend=backend, verbose=verbose)
            self._multidim = True

        if method == 'both':
            y_pred = self.predict(X)
            multidim_impurity = gini(y, y_pred)
            if verbose > 1:
                print("Multidim Accuracy: {:.2f}%".format(accuracy(y, y_pred)))
            if multidim_impurity >= unidim_impurity:
                self.fc = unidim_fc
                self._multidim = False

        return self

    def is_multidim(self):
        return self._multidim
