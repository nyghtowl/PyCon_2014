'''
Linear Regression Example

Working on this for PyCon2014 


'''

import pandas as pd
import numpy as np
import seaborn


import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split


def load_data():
    #diabetes = datasets.load_diabetes()
    return datasets.load_diabetes()

def create_datasets(data):
    diabetes = data

    diabetes_X = diabetes.data[:, np.newaxis] # Use only one feature

    X_train, X_test, y_train, y_test = train_test_split( diabetes_X[:, :, 2],diabetes.target, test_size=0.30)

    return X_train, X_test, y_train, y_test

def build_model(X_train, y_train):
    regr = linear_model.LinearRegression()# Create regression object
    return regr.fit(X_train, y_train) # Train the model

def predict_eval(model, X_test, y_test):
    regr = model
    #The coefficients
    print('Coefficients: %.2f' % regr.coef_)

    # The mean square error
    print("Residual sum of squares: %.2f" % np.mean((regr.predict(X_test) - y_test) ** 2))

    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(X_test, y_test))

def visualize(model, X_test, y_test):
    regr = model

    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, regr.predict(X_test), linewidth=3)
    plt.title('Sci-kit Learn Diabetes Linear Regression')
    plt.xlabel('?')
    plt.ylabel('?')
    plt.savefig('linear_example.png')

    plt.show()

def main():
    data = load_data()
    X_train, X_test, y_train, y_test = create_datasets(data)
    model = build_model(X_train, y_train)
    predict_eval(model, X_test, y_test)
    visualize(model, X_test, y_test)


if __name__ == '__main__':
    main()