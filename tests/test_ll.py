import csv
import torch
import pytest
import subprocess
from telperion.Mallorn import Mallorn
from telperion.Lothlorien import Lothlorien
from telperion.utils import accuracy
from sklearn.tree import DecisionTreeClassifier

torch.manual_seed(42)

BATCH_SIZE = 1024
EPOCHS = 100
LR = 0.01

cpp_libs = """#include <vector>
#include <numeric>
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>

"""


cpp_main = """
int main() {{
    std::vector<float> x(2);
    int y;
    int correct = 0;
    int total = 0;
    std::ifstream infile("/tmp/test_data.csv");
    std::string line;

    while (std::getline(infile, line)) {{
        std::istringstream iss(line);
        std::string val;

        std::getline(iss, val, ',');
        x[0] = std::stof(val);

        std::getline(iss, val, ',');
        x[1] = std::stof(val);

        std::getline(iss, val, ',');
        y = std::stoi(val);

        if (predict(x) == y)
            correct++;
        total++;
     }}

    std::cout << "Accuracy: " << (float)correct / total << std::endl;
    return 0;
}}
"""



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

@pytest.mark.parametrize("n_estimators,max_depth,b", [(3, 3, 0.7)])
def test_lothlorien_cpp_accuracy(n_estimators, max_depth, b):
    # 1. Train the Mallorn model using the provided data
    X = torch.Tensor(10000, 2).uniform_(0, 1)
    y = torch.Tensor([1 if (j)**2 + (i)**2 < b else 0 for i,
                     j in zip(X[:, 0], X[:, 1])])

    forest = Lothlorien(n_estimators=n_estimators,max_depth=max_depth)
    forest.fit(X, y)
    #tree.print_tree()

    # 2. Write the test data to a CSV in /tmp
    csv_path = "/tmp/test_data.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for xi, yi in zip(X, y):
            writer.writerow([xi[0].item(), xi[1].item(), int(yi.item())])

    # 3. Generate the C++ code for the trained model and write it to a file in /tmp
    cpp_code = forest.generate_cpp_function(add_libs=False)
    cpp_path = "/tmp/lothlorien_model.cpp"
    with open(cpp_path, 'w') as cppfile:
        # Add main function to read CSV, run predictions, and print accuracy
        cppfile.write(cpp_libs)
        cppfile.write(cpp_code)
        cppfile.write(cpp_main)

    # 4. Compile and run the C++ code using the test data
    compile_command = "g++ -std=c++11 -o /tmp/lothlorien_test " + cpp_path
    run_command = "/tmp/lothlorien_test"

    subprocess.call(compile_command, shell=True)
    cpp_output = subprocess.check_output(
        run_command, shell=True).decode('utf-8')
    cpp_accuracy = float(cpp_output.split(":")[1].strip())

    # 5. Compare the accuracy of the Python and C++ versions
    python_predictions = forest.predict(X)
    python_accuracy = (python_predictions == y).float().mean().item()

    assert python_accuracy == pytest.approx(cpp_accuracy)
    print(f"Python accuracy: {python_accuracy}, C++ accuracy: {cpp_accuracy}")
