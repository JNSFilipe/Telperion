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

    def generate_cpp_function(self, threshold=0.5, add_libs=True):
        """
        Generate a C++ function from the trained forest.
        """
        indent = '    '

        libs = ""
        if add_libs:
            libs += "#include <vector>\n"
            libs += "#include <iostream>\n"
            libs += "#include <numeric>\n\n"

        function_header = f"int predict(const std::vector<float>& x, float threshold={threshold}) {{\n"
        function_header += f"{indent}std::vector<int> votes;\n"

        function_body = ""
        for t in self.trees:
            function_body += t.to_cpp(standalone=False)

        function_footer = f"{indent}float average = accumulate(votes.begin(), votes.end(), 0.0)/((float)votes.size());\n"
        function_footer += f"{indent}return average >= threshold ? 1 : 0;\n"
        function_footer += "}\n"

        return libs + function_header + function_body + function_footer
