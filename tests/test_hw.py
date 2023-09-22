import torch
import pytest
from telperion.HeartWood import HeartWood
from telperion.utils import accuracy

torch.manual_seed(42)

BATCH_SIZE = 1024
EPOCHS = 100
LR = 0.01


def test_unidim():
    w = torch.Tensor([1.0, 0.0])
    b = 0.8

    # np.random.seed(76)
    X = torch.Tensor(10000, 2).uniform_(0, 1)
    y = (torch.matmul(w, X.t()) > b).float().squeeze()

    hw = HeartWood()
    hw.fit(X, y, metric='gini', method='unidim', verbose=6)

    y_pred = hw.predict(X)
    print('Final Accuracy: {:.2f}%'.format(accuracy(y, y_pred)))

    w_pred = hw.fc.weight.data
    b_pred = hw.fc.bias.data.tolist()[0]

    b_pred = (b_pred/torch.max(torch.abs(w_pred))).tolist()
    w_pred = (w_pred/torch.max(torch.abs(w_pred))).squeeze().tolist()

    b_ref = (b/torch.max(torch.abs(w))).tolist()
    w_ref = (w/torch.max(torch.abs(w))).squeeze().tolist()

    assert b_ref == pytest.approx(abs(b_pred), rel=0.1)
    assert w_ref == [abs(i) for i in w_pred]


def test_multidim():
    w = torch.Tensor([3.0, 6.0])
    b = 4.5

    # np.random.seed(76)
    X = torch.Tensor(10000, 2).uniform_(0, 1)
    y = (torch.matmul(w, X.t()) > b).float().squeeze()

    hw = HeartWood()
    hw.fit(X, y, lr=LR, batch_size=BATCH_SIZE, epochs=EPOCHS,
           metric='gini', method='multidim', verbose=6)
    y_pred = hw.predict(X)
    print('Final Accuracy: {:.2f}%'.format(accuracy(y, y_pred)))

    w_pred = hw.fc.weight.data
    b_pred = hw.fc.bias.data.tolist()[0]

    b_pred = (b_pred/torch.max(torch.abs(w_pred))).tolist()
    w_pred = (w_pred/torch.max(torch.abs(w_pred))).squeeze().tolist()

    b_ref = (b/torch.max(torch.abs(w))).tolist()
    w_ref = (w/torch.max(torch.abs(w))).squeeze().tolist()

    assert b_ref == pytest.approx(abs(b_pred), rel=0.1)
    assert w_ref == pytest.approx([abs(i) for i in w_pred], rel=0.1)


def test_train():
    w = torch.Tensor([1.0, 0.0])
    b = 0.8

    X = torch.Tensor(10000, 2).uniform_(0, 1)
    y = (torch.matmul(w, X.t()) > b).float().squeeze()

    hw = HeartWood()
    hw.fit(X, y, lr=LR, batch_size=BATCH_SIZE, epochs=EPOCHS,
           metric='gini', method='both', verbose=6)

    y_pred = hw.predict(X)
    print('Final Accuracy: {:.2f}%'.format(accuracy(y, y_pred)))

    w_pred = hw.fc.weight.data
    b_pred = hw.fc.bias.data.tolist()[0]

    b_pred = (b_pred/torch.max(torch.abs(w_pred))).tolist()
    w_pred = (w_pred/torch.max(torch.abs(w_pred))).squeeze().tolist()

    b_ref = (b/torch.max(torch.abs(w))).tolist()
    w_ref = (w/torch.max(torch.abs(w))).squeeze().tolist()

    assert b_ref == pytest.approx(abs(b_pred), rel=0.1)
    assert w_ref == [abs(i) for i in w_pred]
    assert not hw.is_multidim()

    w = torch.Tensor([3.0, 6.0])
    b = 4.5

    X = torch.Tensor(10000, 2).uniform_(0, 1)
    y = (torch.matmul(w, X.t()) > b).float().squeeze()

    hw = HeartWood()
    hw.fit(X, y, lr=LR, batch_size=BATCH_SIZE, epochs=EPOCHS,
           metric='gini', method='both', verbose=6)
    y_pred = hw.predict(X)
    print('Final Accuracy: {:.2f}%'.format(accuracy(y, y_pred)))

    w_pred = hw.fc.weight.data
    b_pred = hw.fc.bias.data.tolist()[0]

    b_pred = (b_pred/torch.max(torch.abs(w_pred))).tolist()
    w_pred = (w_pred/torch.max(torch.abs(w_pred))).squeeze().tolist()

    b_ref = (b/torch.max(torch.abs(w))).tolist()
    w_ref = (w/torch.max(torch.abs(w))).squeeze().tolist()

    assert b_ref == pytest.approx(abs(b_pred), rel=0.2)
    assert w_ref == pytest.approx([abs(i) for i in w_pred], rel=0.2)
    assert hw.is_multidim()
