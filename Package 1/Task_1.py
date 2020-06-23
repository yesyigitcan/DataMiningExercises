import pandas as pd
import numpy
from copy import deepcopy
dataset = pd.read_csv('C:\\Users\\YigitCan\\Desktop\\Data Mining\\Homeworks\\Exercise 1\\bank_customer.csv')


# Part a
print("Unique Job List")
print(dataset["job"].unique())

dataset["job"] = dataset["job"].replace("management","white-collar")
dataset["job"] = dataset["job"].replace("admin.","white-collar")

dataset["job"] = dataset["job"].replace("services","pink-collar")
dataset["job"] = dataset["job"].replace("housemaid","pink-collar")

dataset["job"] = dataset["job"].replace("retired","other")
dataset["job"] = dataset["job"].replace("student","other")
dataset["job"] = dataset["job"].replace("unemployed","other")
dataset["job"] = dataset["job"].replace("unknown","other")

print("\n")
print("Unique Poutcome List")
print(dataset["poutcome"].unique())

dataset["poutcome"] = dataset["poutcome"].replace("other", "unknown")


# Part b
columns = list(dataset.keys())
columns_type = list(dataset.dtypes)

print("\n")
print("Attributes List")
print(columns)
print("Attributes Type List")
print(columns_type)

from sklearn.preprocessing import LabelEncoder


encoder = {}
for i in range(len(columns)):
    if columns_type[i] == numpy.dtype('O'):
        encoder[columns[i]] = LabelEncoder()
        dataset[columns[i]] = encoder[columns[i]].fit_transform(dataset[columns[i]])

# Part c, d
from sklearn.model_selection import train_test_split
dataset_1_features = ["age", "job", "marital", "education", "balance", "housing", "duration", "poutcome"]
dataset_2_features = ["job", "marital", "education", "housing"]
target_feature = "deposit"

data_1_x_train, data_1_x_test, data_y_train, data_y_test = train_test_split(dataset[dataset_1_features], dataset[target_feature], test_size=0.3, random_state=42)
data_2_x_train, data_2_x_test, data_y_train, data_y_test = train_test_split(dataset[dataset_2_features], dataset[target_feature], test_size=0.3, random_state=42)



# Part e
from sklearn.tree import DecisionTreeClassifier
dtce1 = DecisionTreeClassifier(criterion="entropy", random_state=42)
dtce1.fit(data_1_x_train, data_y_train)
dtce2 = DecisionTreeClassifier(criterion="entropy", random_state=42)
dtce2.fit(data_2_x_train, data_y_train)

dtcg1 = DecisionTreeClassifier(criterion="gini", random_state=42)
dtcg1.fit(data_1_x_train, data_y_train)
dtcg2 = DecisionTreeClassifier(criterion="gini", random_state=42)
dtcg2.fit(data_2_x_train, data_y_train)

from sklearn.metrics import accuracy_score
accuracy_ent_1 = accuracy_score(data_y_test, dtce1.predict(data_1_x_test))
accuracy_ent_2 = accuracy_score(data_y_test, dtce2.predict(data_2_x_test))

accuracy_gin_1 = accuracy_score(data_y_test, dtcg1.predict(data_1_x_test))
accuracy_gin_2 = accuracy_score(data_y_test, dtcg2.predict(data_2_x_test))

print("\n\n")

print("Accuracy Data 1 Entropy:", accuracy_ent_1)
print("Accuracy Data 2 Entropy:", accuracy_ent_2)

print("Accuracy Data 1 Gini:", accuracy_gin_1)
print("Accuracy Data 2 Gini:", accuracy_gin_2)

ed1 = dtce1.get_depth()
ed2 = dtce2.get_depth()

gd1 = dtcg1.get_depth()
gd2 = dtcg2.get_depth()

print("\n\n")

print("The depth of the decision tree classifier of data 1 by entropy:", ed1)
print("The depth of the decision tree classifier of data 2 by entropy:", ed2)

print("The depth of the decision tree classifier of data 1 by gini:", gd1)
print("The depth of the decision tree classifier of data 2 by gini:", gd2)

# Part g

def pruning(tree, x_train, y_train, x_test, y_test, criterion):
    length = tree.get_depth()
    for depth in range(length - 1, -1, -1):
        previous_tree = deepcopy(tree)
        previous_accuracy = accuracy_score(y_test, tree.predict(x_test))
        tree = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=42)
        tree.fit(x_train, y_train)
        accuracy = accuracy_score(y_test, tree.predict(x_test))
        accuracy_change = accuracy - previous_accuracy
        if accuracy_change < 0:
            return previous_tree
    return tree

print("\n\n")

print("The pruning of decision tree of data 1 by entropy")
new_dtce1 = pruning(dtce1, data_1_x_train, data_y_train, data_1_x_test, data_y_test, "entropy")
print("The depth after pruning:", ed1, " -> ", new_dtce1.get_depth())
new_accuracy_ent_1 = accuracy_score(data_y_test, new_dtce1.predict(data_1_x_test))
print("Accuracy after pruning:", accuracy_ent_1, " -> ", new_accuracy_ent_1)

print("\n")

print("The pruning of decision tree of data 2 by entropy")
new_dtce2 = pruning(dtce2, data_2_x_train, data_y_train, data_2_x_test, data_y_test, "entropy")
print("The depth after pruning:", ed2, " -> ", new_dtce2.get_depth())
new_accuracy_ent_2 = accuracy_score(data_y_test, new_dtce2.predict(data_2_x_test))
print("Accuracy after pruning:", accuracy_ent_2, " -> ", new_accuracy_ent_2)

print("\n")

print("The pruning of decision tree of data 1 by gini")
new_dtcg1 = pruning(dtcg1, data_1_x_train, data_y_train, data_1_x_test, data_y_test, "gini")
print("The depth after pruning:", gd1, " -> ", new_dtcg1.get_depth())
new_accuracy_gin_1 = accuracy_score(data_y_test, new_dtcg1.predict(data_1_x_test))
print("Accuracy after pruning:", accuracy_gin_1, " -> ", new_accuracy_gin_1)

print("\n")

print("The pruning of decision tree of data 2 by gini")
new_dtcg2 = pruning(dtcg2, data_2_x_train, data_y_train, data_2_x_test, data_y_test, "gini")
print("The depth after pruning:", gd2, " -> ", new_dtcg2.get_depth())
new_accuracy_gin_2 = accuracy_score(data_y_test, new_dtcg2.predict(data_2_x_test))
print("Accuracy after pruning:", accuracy_gin_2, " -> ", new_accuracy_gin_2)

print("\n")

print("The depth of new decision tree classifier of data 1 by entropy:", new_dtce1.get_depth())
print("The depth of new decision tree classifier of data 2 by entropy:", new_dtce2.get_depth())
print("The depth of new decision tree classifier of data 1 by gini:", new_dtcg1.get_depth())
print("The depth of new decision tree classifier of data 2 by gini:", new_dtcg2.get_depth())

# Part h
def calculatePValues(acc, n, z):
    temp = float(2 * n * acc + z**2)
    temp2 = float(z**2 + 4 * n * acc - 4 * n * acc**2)
    temp3 = numpy.sqrt(temp2)
    temp4 = float(2 * (n + z**2))

    p_lower = (temp - temp3) / temp4
    p_upper = (temp + temp3) / temp4

    return p_lower, p_upper


n = len(data_y_test)

p_lower_ent_1, p_upper_ent_1 = calculatePValues(accuracy_ent_1, n, 1.96)
p_lower_ent_2, p_upper_ent_2 = calculatePValues(accuracy_ent_2, n, 1.96)

p_lower_gin_1, p_upper_gin_1 = calculatePValues(accuracy_gin_1, n, 1.96)
p_lower_gin_2, p_upper_gin_2 = calculatePValues(accuracy_gin_2, n, 1.96)

print("\n\n")

print("95% confidence interval (z = 1.96) for accuracy values")
print("\t\t\t\t\t", "Lower P \t\t\t\t Upper P")
print("Accuracy Entropy 1\t", p_lower_ent_1, "\t", p_upper_ent_1)
print("Accuracy Entropy 2\t", p_lower_ent_2, "\t", p_upper_ent_2)
print("Accuracy Gini 1\t\t", p_lower_gin_1, "\t", p_upper_gin_1)
print("Accuracy Gini 2\t\t", p_lower_gin_2, "\t", p_upper_gin_2)


# Part i
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

print("\n\n")

import sys
import subprocess
def openImage(path):
    imageViewerFromCommandLine = {'linux':'xdg-open',
                                  'win32':'explorer',
                                  'darwin':'open'}[sys.platform]
    subprocess.run([imageViewerFromCommandLine, path])

input("Press any key to plot decision tree with data 1 by entropy")
print("Tree will be displayed. Please wait")
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (25,10), dpi=500)
plot_tree(dtce1, feature_names=dataset_1_features, class_names=["false", "true"])
try:
    fig.savefig('dtce1.png')
    openImage('dtce1.png')
except:
    plt.show()

print("")



input("Press any key to plot decision tree with data 2 by entropy")
print("Tree will be displayed. Please wait")
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (25,10), dpi=500)
plot_tree(dtce2, feature_names=dataset_2_features, class_names=["false", "true"])
try:
    fig.savefig('dtce2.png')
    openImage('dtce2.png')
except:
    plt.show()

print("")

input("Press any key to plot decision tree with data 1 by gini")
print("Tree will be displayed. Please wait")
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (25,10), dpi=500)
plot_tree(dtcg1, feature_names=dataset_1_features, class_names=["false", "true"])
try:
    fig.savefig('dtcg1.png')
    openImage('dtcg1.png')
except:
    plt.show()

print("")

input("Press any key to plot decision tree with data 2 by gini")
print("Tree will be displayed. Please wait")
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (25,10), dpi=500)
plot_tree(dtcg2, feature_names=dataset_2_features, class_names=["false", "true"])
try:
    fig.savefig('dtcg2.png')
    openImage('dtcg2.png')
except:
    plt.show()

sys.exit(0)