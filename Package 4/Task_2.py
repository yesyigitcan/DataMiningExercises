import pandas
import math

def entropy(y):
    valueTotalCount = float(len(y))
    score = 0.0
    for classCount in y.value_counts():
        p = classCount / valueTotalCount
        score += p * math.log2(p)
    score = -1 * score
    return score

def gini(y):
    valueTotalCount = float(len(y))
    score = 0.0
    for classCount in y.value_counts():
        p = classCount / valueTotalCount
        score += p**2
    score = 1 - score
    return score

def entropy_weighted(collection, target, column = None):
    if type(collection) == pandas.core.frame.DataFrame:
        if column == None:
            # column not specified, general
            output = {}
            for key in collection.keys():
                output.update( { key : entropy_weighted(collection, target, column = key) } )
            return output
        else:
            if column not in collection.keys():
                raise Exception("Specified column does not exist in input data frame")
            else:
                # column specified, special
                # age
                if column == target:
                    y = collection[target]
                    output = entropy(y)
                else:
                    output = 0.0
                    totalCount = float(collection[column].count())
                    for value in collection[column].unique():
                        # older 
                        y = collection.loc[collection[column] == value, target]
                        score = entropy(y)
                        output += (len(y) / totalCount) * score

                return output
    else:
        raise Exception("This function only works with pandas dataframe")

def gini_weighted(collection, target, column = None):
    if type(collection) == pandas.core.frame.DataFrame:
        if column == None:
            # column not specified, general
            output = {}
            for key in collection.keys():
                output.update( { key : gini_weighted(collection, target, column = key) } )
            return output
        else:
            if column not in collection.keys():
                raise Exception("Specified column does not exist in input data frame")
            else:
                # column specified, special
                # age
                if column == target:
                    y = collection[target]
                    output = gini(y)
                else:
                    output = 0.0
                    totalCount = float(collection[column].count())
                    for value in collection[column].unique():
                        # older 
                        y = collection.loc[collection[column] == value, target]
                        score = gini(y)
                        output += (len(y) / totalCount) * score

                return output
    else:
        raise Exception("This function only works with pandas dataframe")


def information_gain(collection, target, column = None):
    if type(collection) == pandas.core.frame.DataFrame:
        if column == None:
            # column not specified, general
            output = {}
            for key in collection.keys():
                output.update( { key : information_gain(collection, target, column = key) } )
            return output
        else:
            if column not in collection.keys():
                raise Exception("Specified column does not exist in input data frame")
            else:
                parentEntropy = entropy_weighted(collection, target, target)
                childEntropy = entropy_weighted(collection, target, column)
                return parentEntropy - childEntropy
    else:
        raise Exception("This function only works with pandas dataframe")

if __name__ == "__main__":
    filepath = 'heart_summary.csv'
    targetClass = 'target'
    df = pandas.read_csv(filepath)
    df_entropy = entropy_weighted(df, targetClass) 
    df_gini = gini_weighted(df, targetClass) # “measure how often a randomly chosen element from the set would be incorrectly labeled”
    df_gain = information_gain(df, targetClass)
    
    print("a. Compute the E and GI f or the overall collection of training examples.")
    print("E: ", df_entropy['target'])
    print("GI: ", df_gini['target'], end="\n\n")

    print("b. Compute the E and GI for the age attribute.")
    print("E: ", df_entropy['age'])
    print("GI: ", df_gini['age'], end="\n\n")

    print("c. Compute the E and GI for the cp attribute.")
    print("E: ", df_entropy['cp'])
    print("GI: ", df_gini['cp'], end="\n\n")

    print("d. Compute the E and GI for the trestbps attribute.")
    print("E: ", df_entropy['trestbps'])
    print("GI: ", df_gini['trestbps'], end="\n\n")

    print("e. Which attribute is better according to calculations?")
    print("We see that gini index and entropy of cp is lower than other attributes. ")
    print("So cp is better than others.", end="\n\n")

    print("f. Which attribute can be chosen as the root ? Explain why.")
    print("Cp can be chosen as the root.")
    print("Because it has the lowest gini and maximum information gain score among the other features.")
    print("IG:  Age: ", df_gain['age'], " Cp: ", df_gain['cp'], " Trestbps: ", df_gain['trestbps'], end="\n\n")

    print("Data chunks that are used for the information above")
    print("Entropy")
    print(df_entropy)
    print("Gini")
    print(df_gini)
    print("Information Gain")
    print(df_gain)

    