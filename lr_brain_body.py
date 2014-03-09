import pandas as pd
import numpy as np
import seaborn


import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split

def load_data(filename):
    return pd.read_csv('./data/brain_body_weight.txt', sep=r"\s*", header=0, index_col=0)

def create_datasets(data):
    X, y = data['BodyWeight'], data['BrainWeight'] # Use only one feature
    return  train_test_split(X, y, test_size=0.30)

def build_model(X_train, y_train):
    ml = linear_model.LinearRegression()# Create regression object
    return ml.fit(X_train, y_train) # Train the model

def predict_eval(model, X_test, y_test):
    # The mean square error
    print('Residual sum of squares: %.2f' % np.mean((model.predict(X_test) - y_test) ** 2))

    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % model.score(X_test, y_test))

def visualize(model, X_test, y_test):
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, model.predict(X_test), color='blue', linewidth=3)
    plt.title('BSci-kit Learn Diabetes Linear Regression')
    plt.xlabel('Body Weight')
    plt.ylabel('Brain Weight')
    plt.savefig('body_brain_lr_ex.png')

    plt.show()

def main():
    data = load_data('./data/brain_body_weight.txt')
    X_train, X_test, y_train, y_test = create_datasets(data)
    model = build_model(X_train, y_train)
    predict_eval(model, X_test, y_test)
    visualize(model, X_test, y_test)


if __name__ == '__main__':
    main()