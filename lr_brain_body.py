import pandas as pd
import numpy as np
import seaborn


import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split

def load_data(filename):
    return pd.read_csv(filename, sep=r"\s*", header=0)

def create_datasets(data):
    X, y = np.array(data['Head_Size']), np.array(data['Brain_Weight'])
    return  train_test_split(X[:,np.newaxis], y, test_size=0.30)

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
    plt.title('Linear Regression Predict Brain Weight by Head Size')
    plt.xlabel('Head Size (cm^3)')
    plt.ylabel('Brain Weight (grams)')
    plt.savefig('body_brain_lr_ex.png')

    plt.show()

def main():
    data = load_data('./data/female_brain_body.txt')
    X_train, X_test, y_train, y_test = create_datasets(data)
    model = build_model(X_train, y_train)
    predict_eval(model, X_test, y_test)
    visualize(model, X_test, y_test)


if __name__ == '__main__':
    main()