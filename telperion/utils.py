import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


def accuracy(y, y_pred):

    # if not isinstance(y, torch.Tensor):
    #     y = torch.tensor(y)
    # if not isinstance(y_pred, torch.Tensor):
    #     y_pred = torch.tensor(y_pred)

    return (y == y_pred).float().sum()/len(y)*100


def impurity(y):
    # Ensure y is a tensor
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)

    # Calculate proportion of positive instances
    p = torch.mean(y.float())

    # Calculate the Gini impurity
    impurity = 2 * p * (1 - p)

    return impurity


def gini(y, y_pred):
    # Ensure both are tensors
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)

    # Split y based on y_pred values
    y_left = y[y_pred == 0]
    y_right = y[y_pred == 1]

    # Calculate the Gini impurity for each split
    gini_left = impurity(y_left)
    gini_right = impurity(y_right)

    # Calculate the weighted Gini impurity for the split
    weight_left = len(y_left) / len(y)
    weight_right = len(y_right) / len(y)

    gini_index = weight_left * gini_left + weight_right * gini_right

    if torch.isnan(gini_index):
        gini_index = torch.tensor(1)

    return gini_index


def train_nn(model, activation, X, y, optimizer=optim.Adam, loss=nn.CrossEntropyLoss, lr=0.01, epochs=10, batch_size=128, verbose=6):
    if not torch.is_tensor(X):
        X = torch.Tensor(X)

    if not torch.is_tensor(y):
        y = torch.Tensor(y)

    dataloader = torch.utils.data.DataLoader(
        list(zip(X, y)), batch_size=batch_size, shuffle=True)

    loss_fn = loss()  # Mean squared error loss
    opt = optimizer(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0.0
        for Xb, target in dataloader:
            # Zero the gradients
            opt.zero_grad()

            # Forward pass
            output = activation(model(Xb)).squeeze()
            loss = loss_fn(output, target)

            # Backward pass and optimization
            loss.backward()
            opt.step()

            if verbose > 1:
                total_loss += loss.item()

        # Print the average loss for this epoch
        if verbose > 1:
            average_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")

    return model


def plot_decision_domains(X, y, classifier, resolution=0.02, margin=None, subsample=None):
    # Set Seaborn style
    sns.set()
    palette = sns.color_palette("muted", n_colors=np.unique(y).shape[0])
    cmap = ListedColormap(palette[:len(np.unique(y))])

    # Get plot boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create a mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))

    # Predict class for each point in the mesh grid
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    if isinstance(Z, torch.Tensor):
        Z = Z.detach()

    # Plot decision regions using contourf
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

    # Subsample data if needed
    if subsample:
        indices = np.random.choice(len(X), int(
            len(X) * subsample), replace=False)
        X_subsampled = X[indices]
        y_subsampled = y[indices]
    else:
        X_subsampled = X
        y_subsampled = y

    # Plot data points
    for idx, cl in enumerate(np.unique(y_subsampled)):
        plt.scatter(x=X_subsampled[y_subsampled == cl, 0],
                    y=X_subsampled[y_subsampled == cl, 1],
                    c=[palette[idx]],
                    label=cl,
                    edgecolor='black',
                    s=100)  # Adjusted size for better visibility

    if not (margin is None):
        plt.xlim(X[:, 0].min()-margin, X[:, 0].max()+margin)
        plt.ylim(X[:, 1].min()-margin, X[:, 1].max()+margin)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Regions')
    plt.legend(loc='upper left')
