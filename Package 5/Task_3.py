import pandas
import itertools
import numpy
from collections import defaultdict
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier
class Rule:
    def __init__(self):
        self.attNameList = list()
        self.attValueList = list()
        self.className = None
    def append(self, name, value):
        self.attNameList.append(name)
        self.attValueList.append(value)
    def setClass(self, className):
        self.className = className
    def getAttributes(self):
        output = list()
        for i in range(len(self.attNameList)):
            output.append((self.attNameList[i], self.attValueList[i]))
        return output
    def getClass(self):
        return self.className
    def toString(self):
        if self.className == None:
            raise Exception("Class name not defined")
        temp = ""
        for i in range(len(self.attNameList)):
            temp += self.attNameList[i] + "=" + self.attValueList[i] + "^"
        temp = temp[:-1] + "->" + self.className
        return temp
    def __str__(self):
        return self.toString()

class RuleBasedClassifier:
    def __init__(self):
        self.X = None           # Dataframe
        self.X_keys = None  # 1D Tuple
        self.y = None           # Dataframe
        
    def fit(self, X, y):
        self.X = X
        self.X_keys = tuple(X.keys())
        self.size = len(self.X_keys)    # Number of features
        self.recordNumber = len(self.X)
        self.y = y
        self.uniqueClassName = self.y.unique()
        self.uniqueValueDict = dict()
        for key in self.X_keys:
            self.uniqueValueDict.update({key:tuple(self.X[key].unique())})
        self.singleRules = list()
        self.doubleRules = list()
        self.tripleRules = list()
        self.totalRules = list()
        self.createRules()
    
    def createRules(self):
        # Creating rules for one attribute

        for key in self.X_keys:
            for uniqueValue in self.uniqueValueDict[key]:
                rule = Rule()
                rule.append(key, uniqueValue)
                countDict = dict.fromkeys(self.uniqueClassName, 0)
                indexList = self.X[self.X[key] == uniqueValue].index.tolist()
                resultClasses = tuple(self.y[indexList])
                for className in self.uniqueClassName:
                    if className in resultClasses:
                        tempRule = deepcopy(rule)
                        tempRule.setClass(className)
                        self.singleRules.append(tempRule)
                        self.totalRules.append(tempRule)
                
        # Creating rules for two and three attributes
        for combination in self.feature_combinations():
            if len(combination) == 2:
                for valueComb in itertools.product(self.uniqueValueDict[combination[0]], self.uniqueValueDict[combination[1]]):
                    rule = Rule()
                    rule.append(combination[0], valueComb[0])
                    rule.append(combination[1], valueComb[1])
                    countDict = dict.fromkeys(self.uniqueClassName, 0)
                    temp = self.X[self.X[combination[0]] == valueComb[0]]
                    indexList = temp[temp[combination[1]] == valueComb[1]].index.tolist()
                    resultClasses = tuple(self.y[indexList])
                    for className in self.uniqueClassName:
                        if className in resultClasses:
                            tempRule = deepcopy(rule)
                            tempRule.setClass(className)
                            self.doubleRules.append(tempRule)
                            self.totalRules.append(tempRule)
            elif len(combination) == 3:
                for valueComb in itertools.product(self.uniqueValueDict[combination[0]], self.uniqueValueDict[combination[1]], self.uniqueValueDict[combination[2]]):
                    rule = Rule()
                    rule.append(combination[0], valueComb[0])
                    rule.append(combination[1], valueComb[1])
                    rule.append(combination[2], valueComb[2])
                    countDict = dict.fromkeys(self.uniqueClassName, 0)
                    temp = self.X[self.X[combination[0]] == valueComb[0]]
                    temp = temp[temp[combination[1]] == valueComb[1]]
                    indexList = temp[temp[combination[2]] == valueComb[2]].index.tolist()
                    resultClasses = tuple(self.y[indexList])
                    for className in self.uniqueClassName:
                        if className in resultClasses:
                            tempRule = deepcopy(rule)
                            tempRule.setClass(className)
                            self.tripleRules.append(tempRule)
                            self.totalRules.append(tempRule)
            else:
                raise Exception("Unexpected length of combinations. Maximum triple allowed")
            
    def feature_combinations(self):
        output = list()
        if type(self.X) == None or type(self.y) == None:
            raise Exception("You need to fit model to see combinations")
        for i in range(self.size):
            if i+1 < self.size:
                for j in range(i+1, self.size):
                    output.append((self.X_keys[i], self.X_keys[j]))
            else:
                continue
            if i+2 < self.size:
                temp = [self.X_keys[i]]
                for j in range(i+1, self.size):
                    temp.append(self.X_keys[j])
                    for k in range(j+1, self.size):
                        temp.append(self.X_keys[k])
                        output.append(tuple(temp))
                        temp.pop(-1)
                    temp.pop(-1)
        del temp
        return output

    def getSingleRules(self):
        return self.singleRules
    def getDoubleRules(self):
        return self.doubleRules
    def getTripleRules(self):
        return self.tripleRules
    def getTotalRules(self):
        return self.totalRules

    def top(self, method="coverage"):
        if method == "coverage":
            ruleDict = dict()
            for rule in self.totalRules:
                ruleDict.update({rule:self.coverage(rule)})
            if len(self.totalRules) >= 10:
                rulesSorted = sorted(ruleDict.items(), key=lambda x: x[1], reverse=True)[:10]
            else:
                rulesSorted = sorted(ruleDict.items(), key=lambda x: x[1], reverse=True)
            return [rule[0] for rule in rulesSorted]
        elif method == "accuracy":
            ruleDict = dict()
            for rule in self.totalRules:
                ruleDict.update({rule:self.accuracy(rule)})
            if len(self.totalRules) >= 10:
                rulesSorted = sorted(ruleDict.items(), key=lambda x: x[1], reverse=True)[:10]
            else:
                rulesSorted = sorted(ruleDict.items(), key=lambda x: x[1], reverse=True)
            return [rule[0] for rule in rulesSorted]
        else:
            raise Exception("Unexpected method")

    def coverage(self, rule):
        attributes = rule.getAttributes() # [ (att1, val1), (att2, val2), ... ]
        temp = self.X
        for attribute in attributes:
            temp = temp[temp[attribute[0]] == attribute[1]]
        indexList = temp.index.tolist()
        a = len(self.y[indexList])
        return float(a) / self.recordNumber

    def accuracy(self, rule):
        attributes = rule.getAttributes() # [ (att1, val1), (att2, val2), ... ]
        temp = self.X
        for attribute in attributes:
            temp = temp[temp[attribute[0]] == attribute[1]]
        indexList = temp.index.tolist()
        a = len(self.y[indexList])
        true = 0
        for realClassName in self.y[indexList]:
            if rule.getClass() == realClassName:
                true += 1
        return true / float(a)


if __name__ == '__main__':
    df = pandas.read_csv('vertebrates.csv')
    x_keys = ["Blood_Type","Give_Birth","Can_Fly","Live_Water"]
    model = RuleBasedClassifier()
    model.fit(df[x_keys], df["Class"])
    
    print("a) Print ​number of classes​,​ number of attributes ​and generate all possible combinations of attributes​ (twice, triple).")
    print("Number of classes: ", len(df["Class"].unique()))
    print("Number of attributes: ", len(x_keys))
    print("Combinations\n")
    for comb in model.feature_combinations():
        if type(comb) == tuple:
            print(comb)
    print()
    
    print("b) Before obtain rules with multiple attributes, firstly obtain rules with single attribute​, one attribute has to be one value in a rule")
    for rule in model.getSingleRules():
        print(rule)
    print()

    print("c) Create rules with ​two attributes​")
    for rule in model.getDoubleRules():
        print(rule)
    print()

    print("d) Create rules with ​tree attributes")
    for rule in model.getTripleRules():
        print(rule)
    print()

    print("e) Total rules")
    ruleset = model.getTotalRules()
    print("Total Rule Count: ", len(ruleset))
    print("----------- All Rules -----------")
    for rule in ruleset:
        print(rule)
    print()

    print("f) Show first 10 rules with coverage and accuracy values")
    print("Coverage")
    for rule in model.top(method="coverage"):
        print(rule)
    print("\nAccuracy")
    for rule in model.top(method="accuracy"):
        print(rule)
    print()
    
