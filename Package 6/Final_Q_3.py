import pandas
from copy import deepcopy
import numpy
import math
import random
import operator
from sklearn.model_selection import train_test_split
class preprocessing:

    # input : pandas Series
    @staticmethod
    def zero_mean_standardize(input):
        if type(input) != pandas.core.series.Series:
            raise Exception("This function only works with pandas series")
        return input.apply(lambda x:(x-input.mean())/input.std())

    @staticmethod
    def shuffle(input, random_seed=0):
        if type(input) != pandas.core.frame.DataFrame:
            raise Exception("This function only works with pandas dataframe")
        random.seed(random_seed)
        length = len(input)
        indexList = df.index.values
        for preIndex in range(length):
            nextIndex = random.randint(0, length-1)
            indexList[preIndex], indexList[nextIndex] = indexList[nextIndex], indexList[preIndex]
        output = input.reindex(indexList)
        return output.sort_index(axis=0)

    @staticmethod
    def kFold(input, n=10, shuffle=False):
        if type(input) != pandas.core.frame.DataFrame:
            raise Exception("This function only works with pandas dataframe")
        length = len(input)
        foldLength = math.ceil(length/n)
        foldList = list()
        for foldNumber in range(n):
            startPoint = foldLength * foldNumber
            endPoint = startPoint + foldLength
            if endPoint > length:
                startPoint = length - foldLength
                endPoint = length
            foldList.append(input.iloc[range(startPoint, endPoint)])
        return tuple(foldList)

class model:
    class NaiveBayes:
        def __init__(self):
            self.X = None
            self.y = None
            self.uniqueTargets = list()
            self.numericDict = dict() # store mean and std of numeric columns {0: {'feature1': (mean, std)}, 1: {'feature1': (mean, std)}}
            self.probDict = dict() # store probabilities for non numeric columns {0: {'feature':{'value1':'p1'}, }}

        def fit(self, X, y):
            if type(X) != pandas.core.frame.DataFrame:
                raise Exception("This function only works with pandas dataframe in X")
            if type(y) != pandas.core.series.Series:
                raise Exception("This function only works with pandas series in y")

            self.X = X
            self.y = y
            self.uniqueTargets = list(y.unique()) 
            for targetValue in self.uniqueTargets:
                self.numericDict.update({targetValue:{}})
                self.probDict.update({targetValue:None})
                for key in X.keys():
                    self.probDict[targetValue] = {key:None}

            for targetValue in self.uniqueTargets:
                for key in X.keys():
                    indexList = list(y[y==targetValue].index.values)
                    X_temp = X.loc[indexList]
                    if(pandas.api.types.is_numeric_dtype(X[key])):
                        # column numeric
                        mean = X_temp[key].mean()
                        std = X_temp[key].std()
                        self.numericDict[targetValue].update({key:(mean, std)})
                    else:
                        # column non numeric
                        X_col_temp = X_temp[key]
                        length = len(X_col_temp)
                        for uniqueValue in list(X_col_temp.unique()):
                            self.probDict[targetValue][key] = {uniqueValue: len(X_col_temp[X_col_temp==uniqueValue])/length}
                    

        def predict(self, X_test):
            if type(self.X) == None or type(self.y) == None:
                raise Exception("This model is not trained. You cannot make prediction via this model")
            if type(X_test) != pandas.core.frame.DataFrame:
                raise Exception("This function only works with pandas dataframe in X")
            return X_test.apply(self.predictRow, axis=1)

        def predictRow(self, x):
            if not self.numericDict or not self.probDict:
                raise Exception("This model is not trained. You cannot make prediction via this model")
            prob = dict()
            for targetValue in self.uniqueTargets:
                p_total = 1
                for key in x.keys():
                    if(pandas.api.types.is_numeric_dtype(x[key])):
                        p = self.likelihood(x[key], self.numericDict[targetValue][key][0], self.numericDict[targetValue][key][1])
                    else:
                        p = self.probDict[targetValue][key][x[key]]
                    p_total *= p
                prob.update({targetValue:p})
            return max(prob.items(), key=operator.itemgetter(1))[0]

        def likelihood(self, x, mean, std):
            part_1 = 1.0/(std * math.sqrt(2 * math.pi))
            part_2 = math.exp(1)**(-((x-mean)**2 / (2*std**2)))
            return part_1 * part_2

class metrics:
    @staticmethod
    def f1_score(y_test, predict):
        precision = metrics.precision(y_test, predict)
        recall = metrics.recall(y_test, predict)
        return 2.0 * (precision * recall) / (precision + recall)

    @staticmethod
    def recall(y_test, predict):
        conf = metrics.binary_confusion_matrix(y_test, predict)
        return  conf["tp"] / float(conf["tp"] + conf["fn"])

    @staticmethod
    def precision(y_test, predict):
        conf = metrics.binary_confusion_matrix(y_test, predict)
        return  conf["tp"] / float(conf["tp"] + conf["fp"])

    @staticmethod
    def binary_confusion_matrix(y_test, predict):
        conf = dict({"tp": 0, "fp": 0, "tn": 0, "fn": 0})
        for valueList in zip(list(y_test), list(predict)):
            if valueList[0] == 1 and valueList[1] == 1:
                conf["tp"] += 1
            elif valueList[0] == 0 and valueList[1] == 1:
                conf["fp"] += 1
            elif valueList[0] == 0 and valueList[1] == 0:
                conf["tn"] += 1
            elif valueList[0] == 1 and valueList[1] == 0:
                conf["fn"] += 1
            else:
                raise Exception("Unexpected class value")
        return conf

if __name__ == '__main__':
    df = pandas.read_csv('diabetes.csv')
    features = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
    target = "Outcome"

    df[features] = df[features].apply(preprocessing.zero_mean_standardize, axis=0)
    df = preprocessing.shuffle(df)
    
    y = df[target]
    
    
    folds = preprocessing.kFold(df)
    f1List = list()
    
    for fold in folds:
        X_train, X_test, y_train, y_test = train_test_split(fold[features], fold[target], test_size=0.33, random_state=42)
        nBayesModel = model.NaiveBayes()
        nBayesModel.fit(X_train, y_train)
        predict = nBayesModel.predict(X_test)
        f1Score = metrics.f1_score(y_test, predict)
        f1List.append(f1Score)

    print("Average F1 Score: ", numpy.mean(f1List))
    print("Best fold according to F1 Score value ", max(f1List), " is k=", f1List.index(max(f1List)))    
