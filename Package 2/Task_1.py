import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.tree import DecisionTreeClassifier
from wittgenstein import RIPPER
from datetime import datetime

# Part a
selectedFeatures = ["age", "cp", "trestbps", "thalach", "chol", "target"]
targetFeature = "target"

df = pandas.read_csv('heart_data.csv')[selectedFeatures]
print(df.describe())
print(df.info())

selectedFeatures.remove(targetFeature)

# Part b
df.loc[df["age"] > 50, "age"] = "older person"
df.loc[df["age"] != "older person", "age"] = "younger person"

# Part c
LabelEncoder_age = LabelEncoder()
df["age"] = LabelEncoder_age.fit_transform(df["age"]) # older person: 0, younger person: 1

# Part d
X_train, X_test, y_train, y_test = train_test_split(df[selectedFeatures], df[targetFeature], test_size=0.2, random_state=42)

# Part e
model1 = RIPPER()
start = datetime.now()
model1.fit(X_train, y_train)
end = datetime.now()
time1 = end - start
fpr, tpr, thresholds = roc_curve(y_test, model1.predict(X_test))
auc_score1 = auc(fpr, tpr)

try:
    X_train = X_train.drop(targetFeature, axis=1)
except:
    pass # Target value not in X_train

# Part f
model2 = DecisionTreeClassifier()
start = datetime.now()
model2.fit(X_train, y_train)
end = datetime.now()
time2 = end - start
fpr, tpr, thresholds = roc_curve(y_test, model2.predict(X_test))
auc_score2 = auc(fpr, tpr)

# Part g
print("Running Times (ms)")
print("------------------")
print("RIPPER:", round(time1.total_seconds() * 1000, 5), "ms")
print("Decision Tree:", round(time2.total_seconds() * 1000, 5), "ms")
print("AUC Score")
print("---------")
print("RIPPER:", auc_score1)
print("Decision Tree:", auc_score2)
