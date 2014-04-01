'''
PyCon Presentation Example
'''

import pandas as pd
import numpy as np
import seaborn

import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split

def cm_inch(cm):
    return float(cm) * 0.0610237441

def grm_pd(grm):
    return float(grm) * 0.00220462

def load_data(filename):
    return pd.read_csv(filename, sep=r"\s*",  converters={'Head_Size': cm_inch, 'Brain_Weight': grm_pd}, header=0)

def create_datasets(data):
    X, y = np.array(data['Head_Size']), np.array(data['Brain_Weight'])
    return  train_test_split(X[:,np.newaxis], y, test_size=0.30)

def build_model(X_train, y_train):
    ml = linear_model.LinearRegression()# Create regression object
    return ml.fit(X_train, y_train) # Train the model

def predict_eval(model, X_test, y_test):
    # The mean square error - how much the values can fluctuate
    print('Root Mean Squared Error: %.2f' % ((np.mean((model.predict(X_test) - y_test) ** 2))**(0.5)))

    # R squared / proportion of variance explained by model: 1 is perfect prediction
    print('Accuracy: %.2f' % model.score(X_test, y_test))

def visualize(model, X_test, y_test):
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, model.predict(X_test), linewidth=3)
    plt.title('Linear Regression - Predict Brain Weight by Head Size', fontsize=14)
    plt.xlabel('Head Size (in^3)')
    plt.ylabel('Brain Weight (lb)')
    plt.xlim(180,280)
    plt.ylim(2.4,3.8)

    plt.savefig('body_brain_lr_ex.png')

    plt.plot([240,240],[2.4,3], color='r')
    plt.plot([180,240],[3,3], color='r')

    plt.savefig('body_brain_lr_red_ex.png')

    # plt.xlabel('Head Size (cm^3)')
    # plt.ylabel('Brain Weight (grams)')

    # font = {'family' : 'normal',
    #     'weight' : 'bold',
    #     'size'   : 22}

    # plt.rc('font', **font)


    plt.show()

def main():
    data = load_data('./data/both_brain_body.txt')
    X_train, X_test, y_train, y_test = create_datasets(data)
    model = build_model(X_train, y_train)
    predict_eval(model, X_test, y_test)
    visualize(model, X_test, y_test)


if __name__ == '__main__':
    main()