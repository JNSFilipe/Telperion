import torch
import pytest
from telperion.Mallorn import Mallorn
from telperion.utils import accuracy
from sklearn.tree import DecisionTreeClassifier

torch.manual_seed(42)

BATCH_SIZE = 1024
EPOCHS = 100
LR = 0.01


def test_unidim():
    max_depth = 2
    w = torch.Tensor([1.0, 1.0])
    b = 0.8

    X = torch.Tensor(10000, 2).uniform_(0, 1)
    y = (torch.matmul(w, X.square().t()) > b).float().squeeze()

    ml = Mallorn(max_depth=max_depth)
    ml.fit(X, y)
    y_ml = ml.predict(X)
    acc_ml = accuracy(y, y_ml)

    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X, y)
    y_dt = dt.predict(X)
    acc_dt = accuracy(y, torch.Tensor(y_dt))

    # TODO verify in notebooks if ml trees are working correctly. For ms=5, dt are yielding better, which should never be the case!!!
    print('Mallorn Accuracy:\t {:.2f}%'.format(acc_ml))
    print('SKl DTs Accuracy:\t {:.2f}%'.format(acc_dt))

    assert acc_ml >= acc_dt


def test_print():
    max_depth = 5
    w = torch.Tensor([1.0, 1.0])
    b = 0.8

    X = torch.Tensor(10000, 2).uniform_(0, 1)
    y = (torch.matmul(w, X.square().t()) > b).float().squeeze()

    ml = Mallorn(max_depth=max_depth)
    ml.fit(X, y)

    ml.print_tree()

    # Check program reaached here
    assert True


def test_prunning():
    max_depth = 3
    b = 0.7

    X = torch.Tensor(10000, 2).uniform_(0, 1)
    y = torch.Tensor([1 if (j)**2 + (i)**2 < b else 0 for i,
                     j in zip(X[:, 0], X[:, 1])])

    ml = Mallorn(max_depth=max_depth)
    ml.fit(X, y, prune=False, verbose=0)
    y_pred = ml.predict(X)
    acc = accuracy(y, y_pred)
    ml.print_tree()

    print('\n\n')

    ml.prune_tree()
    y_pred = ml.predict(X)
    acc_pred = accuracy(y, y_pred)
    ml.print_tree()

    assert acc == acc_pred

    print('\n\n')
    ml = Mallorn(max_depth=max_depth)
    ml.fit(X, y, prune=True, verbose=0)
    y_pred = ml.predict(X)
    acc_pred_arg = accuracy(y, y_pred)
    ml.print_tree()

    assert acc == pytest.approx(acc_pred_arg, rel=0.05)
