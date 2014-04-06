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

def create_datasets(X, y, random):
    return  train_test_split(X, y, test_size=0.30, random_state=random)

def build_model(X_train, y_train):
    model = linear_model.LinearRegression()# Create regression object
    return model.fit(X_train, y_train) # Train the model

def predict_eval(model, X_test, y_test):
    # R squared / proportion of variance explained by model: 1 is perfect prediction
    print('Accuracy: %.2f' % model.score(X_test, y_test))

    # The mean square error - how much the values can fluctuate
    print('Root Mean Squared Error: %.2f' % ((np.mean((model.predict(X_test) - y_test) ** 2))**(0.5)))


def scatter_plot(model, X, y, title, fn):
    plt.scatter(X, y,  color='black')
    plt.title(title, fontsize=16)
    plt.xlabel('Head Size (in^3)', fontsize=13)
    plt.ylabel('Brain Weight (lb)', fontsize=13)
    plt.xlim(180,280)
    plt.ylim(2.4,3.8)

    plt.savefig(fn)
    plt.show()

def model_plot(model, X, y, title, fn1, fn2):
    plt.scatter(X, y,  color='black')

    plt.plot(X, model.predict(X), linewidth=3)

    plt.title(title, fontsize=16)
    plt.xlabel('Head Size (in^3)', fontsize=13)
    plt.ylabel('Brain Weight (lb)', fontsize=13)
    plt.xlim(180,280)
    plt.ylim(2.4,3.8)

    plt.savefig(fn1)

    plt.plot([240,240],[2.4,3], color='r')
    plt.plot([180,240],[3,3], color='r')

    plt.savefig(fn2)


    plt.show()

def main():
    data = load_data('./data/both_brain_body.txt')    
    X, y = np.array(data['Head_Size']), np.array(data['Brain_Weight'])
    X_train, X_test, y_train, y_test = create_datasets(X[:,np.newaxis], y, 6)

    model = build_model(X_train, y_train)
    predict_eval(model, X_test, y_test)

    scatter_plot(model, X, y, 'Linear Regression - Predict Brain Weight by Head Size', 'full_scatter_ex.png')
    
    scatter_plot(model, X_train, y_train, 'Training Data - Brain Weight & Head Size', 'train_scatter_ex.png')

    model_plot(model, X_train, y_train,'Model with Training Data', 'train_model_ex.png', 'train_model_red_ex.png')

    scatter_plot(model, X_test, y_test, 'Test Data - Brain Weight & Head Size', 'test_scatter_ex.png')


    model_plot(model, X_test, y_test, 'Model Evaluation - Brain Weight & Head Size Training Data', 'body_brain_lr_ex.png','body_brain_lr_red_ex.png')

    return model

if __name__ == '__main__':
    main()