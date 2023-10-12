import torch
from telperion.HeartWood import HeartWood
from telperion.SapWood import SapWood

# TODO make inherit sklearn base class
# TODO make method to print the tree
# TODO make method to covert tree to C++


class Mallorn:
    def __init__(self, max_depth=3, min_samples=5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples
        self.root = None

    def _fit(self, X, y, lr=0.01, batch_size=128, epochs=50, metric='gini', method='both', backend='skorch', depth=0, verbose=0):
        # Base cases for recursion
        reach_max_depth = False
        not_enough_samples = False

        # 5 is the minimum number of samples required by skorch
        if depth == self.max_depth or len(y) <= max(self.min_samples_leaf, 5):
            reach_max_depth = True

        if not reach_max_depth:
            # Train a HeartWood (stump) on the data
            stump = HeartWood()
            stump.fit(X, y, lr=lr, batch_size=batch_size, epochs=epochs,
                      metric=metric, method=method, backend=backend, verbose=verbose)

            # Split data based on stump's decision
            predictions = stump.predict(X)
            left_indices = predictions == 0
            right_indices = predictions == 1

            # If stump can't split data further, create a leaf node
            # 5 is the minimum number of samples required by skorch
            if len(y[left_indices]) < 5 or len(y[right_indices]) < 5:
                not_enough_samples = True

            if not not_enough_samples:
                # Recursively build the tree
                node = SapWood()
                node.stump = stump
                node.left = self._fit(
                    X[left_indices], y[left_indices],
                    lr=lr, batch_size=batch_size, epochs=epochs,
                    metric=metric, method=method, backend=backend,
                    depth=depth+1, verbose=verbose)

                node.right = self._fit(
                    X[right_indices], y[right_indices],
                    lr=lr, batch_size=batch_size, epochs=epochs,
                    metric=metric, method=method, backend=backend,
                    depth=depth+1, verbose=verbose)

        if reach_max_depth or not_enough_samples:
            if verbose > 2:
                if reach_max_depth:
                    print("Stopped at depth {depth} due to reach max depth")
                elif not_enough_samples:
                    print("Stopped at depth {depth} due to not enough samples")
            leaf_node = SapWood()
            leaf_node.is_leaf = True
            leaf_node.value = 1.0 if sum(
                y) / len(y) > 0.5 else 0.0  # Majority class
            return leaf_node

        return node

    def fit(self, X, y, lr=0.01, batch_size=128, epochs=50, metric='gini', method='both', backend='skorch', prune=True, verbose=0):
        self.root = self._fit(X, y,
                              lr=lr, batch_size=batch_size, epochs=epochs,
                              metric=metric, method=method, backend=backend,
                              depth=0, verbose=verbose)

        if prune:
            self.prune_tree()

    def _predict_single(self, node, x):
        if node.is_leaf:
            return node.value
        decision = node.stump.predict(x)
        if decision == 0:
            return self._predict_single(node.left, x)
        else:
            return self._predict_single(node.right, x)

    def predict(self, X):
        return torch.Tensor([self._predict_single(self.root, x) for x in X])

    def print_tree(self, node=None, depth=0, prefix="Root:"):
        if node is None:
            node = self.root

        # Print the current node
        if node.is_leaf:
            print(f"{'  ' * depth}{prefix} Leaf => Class: {node.value}")
        else:
            # print(f"{'  ' * depth}{prefix} Decision Stump (Feature Index, Threshold)")
            w = node.stump.fc.weight.data.squeeze().tolist()
            b = node.stump.fc.bias.data.tolist()[0]
            print(
                f"{'  ' * depth}{prefix} Decision Stump ( {[round(i,2) for i in w]} * X > {-b:.2f} )")

        # Recursively print left and right children
        if node.left:
            self.print_tree(node.left, depth + 1, "Left:  ")
        if node.right:
            self.print_tree(node.right, depth + 1, "Right: ")

    def prune_tree(self, node=None):
        """
        Prune the tree by removing branches where both left and right children 
        are leaf nodes with the same value.
        """
        if node is None:
            node = self.root

        # Base case: if the node is a leaf, return its value
        if node.is_leaf:
            return node.value

        # Recursively prune left and right children
        left_value = self.prune_tree(node.left)
        right_value = self.prune_tree(node.right)

        # If both children are leaves and have the same value, prune them
        if left_value == right_value and left_value is not None:
            node.is_leaf = True
            node.value = left_value
            node.left = None
            node.right = None
            node.stump = None
            return node.value

        return None

    def to_cpp(self, node=None, depth=0, standalone=True):
        """
        Convert the trained tree into a C++ function.
        """
        if node is None:
            node = self.root

        indent = '    ' * (depth+1)
        cpp_code = ""

        # If the node is a leaf, return its value
        if node.is_leaf:
            if standalone:
                cpp_code += f"{indent}return {int(node.value)};\n"
            else:
                cpp_code += f"{indent}votes.push_back({int(node.value)});\n"
            return cpp_code

        # Generate the decision code for the stump
        w = node.stump.fc.weight.data.squeeze().tolist()
        b = node.stump.fc.bias.data.tolist()[0]
        # Skip indices with w[i] == 0
        decision_terms = [
            f"({w[i]} * x[{i}])" for i in range(len(w)) if w[i] != 0]
        decision_code = " + ".join(decision_terms)

        cpp_code += f"{indent}if ({decision_code} > {-b}) {{\n"
        cpp_code += self.to_cpp(node.right, depth + 1, standalone=standalone)
        cpp_code += f"{indent}}} else {{\n"
        cpp_code += self.to_cpp(node.left, depth + 1, standalone=standalone)
        cpp_code += f"{indent}}}\n"

        return cpp_code

    def generate_cpp_function(self):
        """
        Generate a C++ function from the trained tree.
        """
        function_header = "int predict(const std::vector<float>& x) {\n"
        function_body = self.to_cpp()
        function_footer = "}\n"
        return function_header + function_body + function_footer
