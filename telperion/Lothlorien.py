import torch
from tqdm import tqdm
from joblib import Parallel, delayed
from telperion.Mallorn import Mallorn


class Lothlorien:
    def __init__(self, n_estimators=100, max_depth=3, bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.trees = []

    def _train_tree(self, X, y, **kwargs):
        # If bootstrap is True, sample with replacement
        if self.bootstrap:
            indices = torch.randint(0, len(X), (len(X),))
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample = X
            y_sample = y

        tree = Mallorn(max_depth=self.max_depth)
        tree.fit(X_sample, y_sample, **kwargs)
        return tree

    def fit(self, X, y, n_jobs=-1, **kwargs):
        with tqdm(total=self.n_estimators, desc="Training Trees") as pbar:
            self.trees = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(self._train_tree)(X, y, **kwargs) for _ in range(self.n_estimators)
            )
            pbar.update()

    def predict(self, X):
        # Get predictions from all trees
        predictions = torch.stack([tree.predict(X) for tree in self.trees])
        # Majority vote for final prediction
        majority_vote = torch.round(predictions.float().mean(dim=0))
        return majority_vote

    def score(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y).float().mean().item()
