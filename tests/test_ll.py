import torch
import pytest
from telperion.Mallorn import Mallorn
from telperion.Lothlorien import Lothlorien
from telperion.utils import accuracy
from sklearn.tree import DecisionTreeClassifier

torch.manual_seed(42)

BATCH_SIZE = 1024
EPOCHS = 100
LR = 0.01


def test_lothlorien():
    max_depth = 5
    w = torch.Tensor([1.0, 1.0])
    b = 0.8

    X = torch.Tensor(10000, 2).uniform_(0, 1)
    y = (torch.matmul(w, X.square().t()) > b).float().squeeze()

    ml = Mallorn(max_depth=max_depth)
    ml.fit(X, y)
    y_ml = ml.predict(X)
    acc_ml = accuracy(y, y_ml)

    ll = Lothlorien(max_depth=max_depth, n_estimators=5)
    ll.fit(X, y)
    y_ll = ll.predict(X)
    acc_ll = accuracy(y, y_ll)

    print('Mallorn Accuracy:\t {:.2f}%'.format(acc_ml))
    print('Lothlorien Accuracy:\t {:.2f}%'.format(acc_ll))

    assert acc_ml <= acc_ll
