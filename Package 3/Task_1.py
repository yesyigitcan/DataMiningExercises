import numpy
import pandas
import matplotlib.pyplot as plt

class MinMaxScaler():
    # input: 2D List
    def fit(self, input):
        self.minVList = []
        self.rangeVList = []
        numpy.apply_along_axis(self.apply_fit, 0, input)

    def transform(self, input):
        self.i = 0
        self.transf = []
        numpy.apply_along_axis(self.apply_transform, 0, input)
        return list(zip(*self.transf))

    def fit_transform(self, datalist):
        self.fit(datalist)
        return self.transform(datalist)

    def backtransform(self, input):
        self.i = 0
        self.transf = []
        numpy.apply_along_axis(self.apply_backtransform, 0, input)
        return list(zip(*self.transf))

    def apply_fit(self, input):
        minV = min(input)
        self.minVList.append(minV)
        self.rangeVList.append(max(input) - minV)

        
    def apply_transform(self, input):
        self.transf.append( [(x-self.minVList[self.i])/(self.rangeVList[self.i]) for x in input] )
        self.i += 1

    def apply_backtransform(self, input):
        self.transf.append( [x*self.rangeVList[self.i] + self.minVList[self.i] for x in input] )
        self.i += 1

class Model:
    def KNN(self, train, test, ktype, k):
        predict = []
        if k > len(train):
            raise OverflowError
        for test_row in test:
            # [5, 39, 1]
            neighwdis = [] # neighbors with distance
            for train_row in train:
                target = train_row[-1]
                if ktype == 'EUC': #euclidean distance
                    neighwdis.append([ self.euclidean_distance(test_row[:-1], train_row[:-1]) , target ])
                elif ktype == 'MAN':
                    neighwdis.append([ self.manhattan_distance(test_row[:-1], train_row[:-1]) , target ])
                else:
                    raise TypeError
            
            neighwdis.sort(key = lambda x: x[0])
            nClosestList = [e[1] for e in neighwdis[:k]]
            predict.append( max(set(nClosestList), key = nClosestList.count)    ) # take mode of neighbor list [:k]
            
        return predict
            
    # input1: Test, input2: Train
    def euclidean_distance(self, input1 , input2):
        temp = 0.0
        for i in range(len(input1)):
            temp += numpy.abs(input1[i]-input2[i])**2
        return numpy.sqrt(temp)
    def manhattan_distance(self, input1, input2):
        temp = 0.0
        for i in range(len(input1)):
            temp += numpy.abs(input1[i]-input2[i])
        return temp
    def accuracy_score(self, y, predict):
        true = 0
        total = 0
        for i in range(len(y)):
            if type(y[i]) != type(predict[i]):
                raise TypeError 
            if y[i] == predict[i]:
                true += 1
            total += 1
        return true / float(total)

if __name__ == "__main__":
    filepath = "covid.csv"

    train_set = numpy.genfromtxt(filepath, delimiter=',')[1:] # Skip feature name row
    test_set = [[5, 39.0, 1], [4, 35.0, 0], [3, 38.0, 0],
              [2, 39.0, 1], [1, 35.0, 0], [0, 36.2, 0],
              [5, 39.0, 1], [2, 35.0, 0], [3, 38.9, 1],
              [0, 35.6, 0], [4, 37.0, 0], [4, 36.0, 1],
              [3, 36.6, 0], [3, 36.6, 1], [4, 36.6, 1]]

    scaler = MinMaxScaler()
    train_set = scaler.fit_transform(train_set)
    test_set = scaler.transform(test_set)

    test_y = [e[2] for e in test_set]
    model = Model()

    kList = range(1, len(train_set))
    acc_euc = []
    acc_man = []
    

    for i in kList:
        print("For k =", i)
        predict = model.KNN(train_set, test_set, 'EUC', i)
        acc = model.accuracy_score(test_y, predict)
        acc_euc.append(acc)
        print("Accuracy | EUC:", acc)
        predict = model.KNN(train_set, test_set, 'MAN', i)
        acc = model.accuracy_score(test_y, predict)
        acc_man.append(model.accuracy_score(test_y, predict))
        print("Accuracy | MAN:", acc)
        print("\n")

    # EUC: Eucledian Distance | MAN: Manhattan Distance

    fig3, (ax3, ax4) = plt.subplots(1,2,sharey=True, sharex=True,figsize=(10,5))

    line3, = ax3.plot( kList, acc_euc)
    line4, = ax4.plot( kList, acc_man)
    
    ax3.set_ylabel('Accuracy Score')
    ax3.set_title("Euclidean Distance")
    ax3.set_xlabel("k")
    ax4.set_title("Manhattan Distance")
    ax4.set_xlabel("k")
    
    ax3.grid(True)
    ax4.grid(True)

    plt.subplots_adjust(bottom = 0.2)
    
    plt.show()


    # Best k value
    # For euclidean distance, k = 13 is the least number that gives the best accuracy score which is 0.8
    # For manhattan distance, k = 1 is the least number that gives the best accuracy score which is 0.8

    # Additional Part
    # You can see the distribution of the train set and test set
    # Blue: Train set positive, Red: Train set negative
    # Green: Test set positive, Yellow: Test set negative
    # Delete or comment sys.exit(0) to see this additional part

    import sys
    sys.exit(0)

    train_set = scaler.backtransform(train_set)
    test_set = scaler.backtransform(test_set)

    import matplotlib.pyplot as plt
    pos = []
    neg = []
    for e in train_set:
        if e[2] == 1:
            pos.append(e[:2])
        else:
            neg.append(e[:2])

    plt.plot([e[0] for e in pos], [e[1] for e in pos], 'bo', label='Train Set Positive')
    plt.plot([e[0] for e in neg], [e[1] for e in neg], 'ro', label='Train Set Negative')

    pos = []
    neg = []
    for e in test_set:
        if e[2] == 1:
            pos.append(e[:2])
        else:
            neg.append(e[:2])
    plt.plot([e[0] for e in pos], [e[1] for e in pos], 'go', label='Test Set Positive')
    plt.plot([e[0] for e in neg], [e[1] for e in neg], 'yo', label='Test Set Negative')

    plt.title("Dataset Distribution")
    plt.xlabel("Couch Level")
    plt.ylabel("Fever")
    plt.legend()
    plt.show()