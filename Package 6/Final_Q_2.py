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

if __name__ == '__main__':
    a1 = ["T", "F", "F", "T", "T", "F", "T", "F", "T", "T"]
    a2 = ["T", "T", "T", "T", "F", "F", "F", "F", "F", "F"]
    a3 = [6.0, 4.0, 3.0, 7.0, 1.0, 8.0, 5.0, 4.0, 2.0, 9.0]
    tr = [1, 0, 0, 1, 1, 0, 1, 1, 1, 0]
    df = pandas.DataFrame({"a1":a1, "a2":a2, "a3":a3, "Target":tr})
    df_gain = information_gain(df, "Target")
    print(df_gain)
    print()
    print("Information gain a1: ", df_gain["a1"])
    print("Information gain a2: ", df_gain["a2"])