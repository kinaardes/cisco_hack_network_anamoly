import pandas as pd
import numpy as np
from sklearn import preprocessing

#df = pd.read_csv('/Users/kindesai/PycharmProjects/CISCOHACK/Kaggle/Train_data1.csv')
import sklearn as sk
from sklearn.linear_model import LogisticRegression
import os

names=[

    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted' ,'num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate'
]

import matplotlib.pyplot as plt

le = preprocessing.LabelEncoder()

df = pd.read_csv('/Users/kindesai/PycharmProjects/CISCOHACK/Kaggle/Train_data.csv').apply(le.fit_transform)

X = df.drop('class', axis = 1)

y = df['class']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB

import math


def predDescisionTree(X_train, y_train, X_test, y_test):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)


    print('------------*Descision Tree CONFUSION MATRIX*------------------')

    print(confusion_matrix(y_test, y_pred) )
    print('\n')
    print(classification_report(y_test, y_pred))

def predSVM(X_train, y_train, X_test, y_test):

    svclassifier = SVC()
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)

    print('------------*SVM CONFUSION MATRIX*-------------------')

    print(confusion_matrix(y_test,y_pred))
    print('\n')
    print(classification_report(y_test,y_pred))

def predNaiveBayes(X_train, y_train, X_test, y_test):

    gnb = GaussianNB()
    gnb = gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)

    print('------------*Naive Bayes CONFUSION MATRIX*-------------------')

    print(confusion_matrix(y_test, y_pred))
    print('\n')
    print(classification_report(y_test, y_pred))

def predKNN(X_train, y_train, X_test, y_test):

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)

    print('------------*KNN CONFUSION MATRIX*-------------------')

    print(confusion_matrix(y_test, y_pred))
    print('\n')
    print(classification_report(y_test, y_pred))


def predRandomForest(X_train, y_train, X_test, y_test):

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)



    print('------------*RandomForest CONFUSION MATRIX*-------------------')

    print(confusion_matrix(y_test, y_pred))
    print('\n')
    print(classification_report(y_test, y_pred))



import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential

import numpy as np

train_df = pd.read_csv('/Users/kindesai/PycharmProjects/CISCOHACK/Kaggle/Train_data.csv')
test_df = pd.read_csv('/Users/kindesai/PycharmProjects/CISCOHACK/Kaggle/Test_data.csv')


le = preprocessing.LabelEncoder()
le2 = preprocessing.StandardScaler()

labeled_train_df = train_df.apply(le.fit_transform)
labels = labeled_train_df['class']
ip = labeled_train_df.loc[:, labeled_train_df.columns!='class']
ip=ip.values

ip2=le2.fit_transform(ip)
train_ip, test_ip, train_op, test_op = train_test_split(ip2, labels, test_size=0.2)

train_op = train_op.values
test_op = test_op.values




model = Sequential()
model.add(Dense(10, activation=tf.nn.relu, input_shape=[41]))
model.add(Dropout(0.2))
model.add(Dense(15, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(1, activation=tf.nn.sigmoid))

model.compile(optimizer='rmsprop', loss ='binary_crossentropy', metrics=['accuracy'])

model.fit(train_ip, train_op, epochs = 10)


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

y_pred = list(model.predict(train_ip))
output=[]
for item in y_pred:
    output.append(math.ceil(item))

output = np.array(output)


print('------------*FeedFordward Neural Network CONFUSION MATRIX*-------------------')

print(confusion_matrix(train_op, output))
print('\n')

print(classification_report(train_op, output))



if __name__ == "__main__":


    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

    predDescisionTree(X_train, y_train, X_test, y_test)
    predRandomForest(X_train, y_train, X_test, y_test)
    predNaiveBayes(X_train, y_train, X_test, y_test)
    predKNN(X_train, y_train, X_test, y_test)
    predSVM(X_train, y_train, X_test, y_test)









