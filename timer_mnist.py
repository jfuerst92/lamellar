import numpy as np
import pandas as pd
import pprint
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.datasets import load_digits
from IPython.display import display, HTML
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import math, time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import tensorflow
import graphviz
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation, BatchNormalization, ZeroPadding2D
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Binarizer, StandardScaler
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


def main():

    np.random.seed(0)
    overall_start_time = time.time()
    print("loading data...")
    df_train = pd.read_csv("../data/mnist/csv_format/train.csv")
    df_test = pd.read_csv("../data/mnist/csv_format/test.csv")
    X = []
    y = []
    for row in df_train.iterrows() :
        label = row[1][0] # label (the number visible in the image)
        image = list(row[1][1:]) # image information as list, without label
        image = np.array(image) / 255
        X.append(image)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    print("done loading.")

    print("y")
    rs=13

    hyperparameters = {
        'decision tree': {

            'max_depth': [2, 4, 8, 16, 32, 64, 128, None],
            'min_samples_split':  [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            'min_samples_leaf':  [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            'random_state': [13]

        },
        'SVM': {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf', 'linear', 'poly']
        },
        'KNN': {
            'k': np.arange(1, 30, 1)
        },
        'adaboost': {
            'n_estimators': 50
        },
        'nnet': {
            'loss': "categorical_crossentropy",
            'optimizer': 'adam',
            'epochs': 60,
            'batch_size':128
        }
    }

    #data_train_sizes = [0.1, 0.2, 0.3]
    #data_test_sizes = [0.025, 0.05, 0.075]
    data_train_sizes = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
    data_test_sizes = [0.025, 0.0275, 0.0325, 0.035, 0.0375]
    num_train_datapoints = []

    times = {
        'KNN':[],
        'decision tree':[],
        'adaboost':[],
        'SVM':[],
        #'nnet':[]
    }
    test_times = {
        'KNN':[],
        'decision tree':[],
        'adaboost':[],
        'SVM':[],
        #'nnet':[]
    }
    test_accs = {
        'KNN':[],
        'decision tree':[],
        'adaboost':[],
        'SVM':[],
        #'nnet':[]
    }
    train_accs = {
        'KNN':[],
        'decision tree':[],
        'adaboost':[],
        'SVM':[],
        #'nnet':[]
    }
    methods = ['KNN', 'decision tree', 'SVM', 'adaboost']#, 'adaboost', 'SVM', 'nnet']

    for i in range(5):
        print("Splitting data")
        sss = StratifiedShuffleSplit(n_splits=1, random_state=0, train_size=data_train_sizes[i], test_size=data_test_sizes[i])
        for train_index, test_index in sss.split(X, y):
            print("hm")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        print(len(X_train), len(y_train))
        print(len(X_test), len(y_test))
        num_train_datapoints.append(len(X_train))
        print("done splitting data")

        """
        KNN
        """
        print("starting knn timer")
        start_time = time.time()
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train,y_train)
        duration = time.time() - start_time
        print("knn train time for train size ", len(X_train), ":", duration)
        times['KNN'].append(duration)
        print("scoring")
        #accs['KNN'].append(model.score(X_test, y_test))
        start_time = time.time()
        y_pred = model.predict(X_test)
        duration = time.time() - start_time
        test_times['KNN'].append(duration)
        y_pred = model.predict(X_train)
        train_accs['KNN'].append(metrics.accuracy_score(y_train, y_pred))
        y_pred = model.predict(X_test)
        test_accs['KNN'].append(metrics.accuracy_score(y_test, y_pred))
        print("done scoring")

        """
        Decision Tree
        """
        print("starting dt timer")
        model = tree.DecisionTreeClassifier(max_depth=128, min_samples_leaf=4, random_state=13)

        start_time = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - start_time

        print("decision tree train time for train size ", len(X_train), ":", duration)
        times['decision tree'].append(duration)

        start_time = time.time()
        y_pred = model.predict(X_test)
        duration = time.time() - start_time

        test_times['decision tree'].append(duration)
        y_pred = model.predict(X_train)
        train_accs['decision tree'].append(metrics.accuracy_score(y_train, y_pred))
        y_pred = model.predict(X_test)
        test_accs['decision tree'].append(metrics.accuracy_score(y_test, y_pred))

        """
        SVM
        """
        model = SVC(kernel='rbf', C=10, gamma=0.01)

        start_time = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - start_time

        print("svm train time for train size ", len(X_train), ":", duration)
        times['SVM'].append(duration)

        start_time = time.time()
        y_pred = model.predict(X_test)
        duration = time.time() - start_time

        test_times['SVM'].append(duration)
        y_pred = model.predict(X_train)
        train_accs['SVM'].append(metrics.accuracy_score(y_train, y_pred))
        y_pred = model.predict(X_test)
        test_accs['SVM'].append(metrics.accuracy_score(y_test, y_pred))


        """
        adaboost
        """
        print("starting adaboost timer")
        d_tree = tree.DecisionTreeClassifier(
            max_leaf_nodes=172,
            max_depth=10,
            max_features=771,
            min_samples_leaf=4,
            random_state=13
        )
        model = AdaBoostClassifier(base_estimator=d_tree, n_estimators=50)
        start_time = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - start_time
        print("adaboost train time for train size ", len(X_train), ":", duration)
        times['adaboost'].append(duration)
        y_pred = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)

        start_time = time.time()
        y_pred = model.predict(X_test)
        duration = time.time() - start_time
        test_times['adaboost'].append(duration)
        y_pred = model.predict(X_train)
        train_accs['adaboost'].append(metrics.accuracy_score(y_train, y_pred))
        y_pred = model.predict(X_test)
        test_accs['adaboost'].append(metrics.accuracy_score(y_test, y_pred))


        """
        CNN
        """
        """
        num_category = 10
        y_train_oh = keras.utils.to_categorical(y_train, num_category)
        y_test_oh = keras.utils.to_categorical(y_test, num_category)
        model = keras.models.Sequential()
        model.add(Dense(units=32, activation='sigmoid', input_shape=(784,)))
        model.add(Dense(units=10, activation='softmax'))
        opt = keras.optimizers.SGD(lr=0.008)
        loss_fn = keras.losses.categorical_crossentropy
        model.compile(loss=loss_fn,
                      optimizer=opt,
                      metrics=['accuracy'])
        start_time = time.time()
        history = model.fit(X_train, y_train_oh, batch_size=128, epochs=5, verbose=False, validation_split=.1)
        duration = time.time() - start_time
        print("nnet train time for train size ", len(X_train), ":", duration)
        times['nnet'].append(duration)
        loss, accuracy  = model.evaluate(X_test, y_test_oh, verbose=False)
        loss, accuracy  = model.evaluate(X_train, y_train_oh, verbose=False)
        train_accs['KNN'].append(metrics.accuracy_score(y_train, y_pred))
        loss, accuracy  = model.evaluate(X_test, y_test_oh, verbose=False)
        test_accs['KNN'].append(metrics.accuracy_score(y_test, y_pred))
        """

    """
    plots
    """

    print(num_train_datapoints)
    #print(times)
    for method in methods:
        print(num_train_datapoints)
        print(test_accs[method])
        plt.plot(num_train_datapoints, times[method], label = "train")
        plt.plot(num_train_datapoints, test_times[method], label = "test")
        plt.xlabel("# training datapoints")
        plt.ylabel("Time (s)")
        title = "Train/Test time for " + method + " on MNIST"
        plt.title(title)
        #plt.show()
        filename = 'output/' + method + '_time_train.png'
        plt.legend()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.clf()

        plt.plot(num_train_datapoints, test_accs[method], label="test")
        plt.plot(num_train_datapoints, train_accs[method], label="train")
        plt.xlabel("# training datapoints")
        plt.ylabel("Accuracy on test set")
        title = "Accuracy for " + method + " on MNIST"
        plt.title(title)
        #plt.show()
        filename = 'output/' + method + '_acc.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.clf()

    overall_duration = time.time() - overall_start_time
    print("executed in ", overall_duration, " seconds.")



if __name__ == "__main__":
    main()















"""
class KNN:
    def __init__(n_neighbors):
        self.n_neighbors = n_neighbors

    def test_k_range(kVals, X_train, y_train, X_test, y_test):
        accs = []
        for k in kVals:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train,y_train)
            score = model.score(X_test, y_test)
            print("k=%d, accuracy=%.2f%%" % (k, score * 100))
            accs.append(score* 100)
        return accs

    def run
"""
