import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

def init():
    df = pd.read_csv("./data.csv")
    
    # Dropping the 'id' column from the dataset
    df = df.drop("id", axis=1)
    df = df.drop("Unnamed: 32", axis=1)
    
    # Mapping M to 1 and B to 0 in the output Label DataFrame
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Split Data into training and test (70% and 30%)
    train, test = train_test_split(df, test_size=0.3, random_state=1)
    
    # Training Data
    train_x = train.loc[:, 'radius_mean':'fractal_dimension_worst']
    train_y = train.loc[:, ['diagnosis']]
    
    # Testing Data
    test_x = test.loc[:, 'radius_mean':'fractal_dimension_worst']
    test_y = test.loc[:, ['diagnosis']]
    
    # Converting Training and Test Data to numpy array
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    test_x = np.asarray(test_x)
    test_y = np.asarray(test_y)
    
    # Calling the model function to train a Logistic Regression Model on Training Data
    d_logistic = model(train_x.T, train_y.T, num_of_iterations=10000, alpha=0.000001, logistic=True)
    d_linear = model(train_x.T, train_y.T, num_of_iterations=10000, alpha=0.000001, logistic=False)

    costs_logistic = d_logistic["costs"]
    w = d_logistic["w"]
    b = d_logistic["b"]
    
    # Drawing the plot between cost and number of iterations
    plt.plot(costs_logistic)
    plt.title("Cost vs #Iterations")
    plt.xlabel("Number of Iterations (* 1000)")
    plt.ylabel("Cost")
    plt.show()

    costs = d_linear["costs"]
    w = d_linear["w"]
    b = d_linear["b"]
    
    # Drawing the plot between cost and number of iterations
    plt.plot(costs)
    plt.title("Cost vs #Iterations")
    plt.xlabel("Number of Iterations (* 1000)")
    plt.ylabel("Cost")
    plt.show()
    
    # Logistic Regression Results
    print("Logistic Regression:")
    print("Train accuracy: {}%".format(evaluate_model(train_x.T, train_y.T, d_logistic)))
    print("Test accuracy: {}%".format(evaluate_model(test_x.T, test_y.T, d_logistic)))
    print()
    
    # Linear Regression Results
    print("Linear Regression:")
    print("Train accuracy: {}%".format(evaluate_model(train_x.T, train_y.T, d_linear, logistic=False)))
    print("Test accuracy: {}%".format(evaluate_model(test_x.T, test_y.T, d_linear, logistic=False)))
    print()

# Rest of the code remains the same

def initialize(m):
    w = np.zeros((m, 1))
    b = 0
    return w, b

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def propagate(X, Y, w, b, logistic=True):
    m = X.shape[1]
    Z = np.dot(w.T, X) + b
    
    if logistic:
        A = sigmoid(Z)
        cost = -(1/m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
        dw = (1/m) * np.dot(X, (A-Y).T)
        db = (1/m) * np.sum(A-Y)
    else:
        A = Z
        cost = (1/(2*m)) * np.sum((A-Y)**2)
        dw = (1/m) * np.dot(X, (A-Y).T)
        db = (1/m) * np.sum(A-Y)
    
    grads = {"dw": dw, "db": db}
    return grads, cost

def optimize(X, Y, w, b, num_of_iterations, alpha, logistic=True):
    costs = []
    
    for i in range(num_of_iterations):
        grads, cost = propagate(X, Y, w, b, logistic)
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - alpha * dw
        b = b - alpha * db
        
        if i % 10 == 0:
            costs.append(cost)
            print("Cost after %i iteration: %f" % (i, cost))
    
    parameters = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    
    return parameters, grads, costs

def predict(X, w, b, logistic=True):
    m = X.shape[1]
    y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    if logistic:
        A = sigmoid(np.dot(w.T, X) + b)
        y_prediction = np.round(A)
    else:
        y_prediction = np.dot(w.T, X) + b
    
    return y_prediction

def evaluate_model(X, Y, parameters, logistic=True):
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction = predict(X, w, b, logistic)
    accuracy = 100 - np.mean(np.abs(Y_prediction - Y)) * 100
    return accuracy

def model(Xtrain, Ytrain, num_of_iterations, alpha, logistic=True):
    dim = Xtrain.shape[0]
    w, b = initialize(dim)
    parameters, grads, costs = optimize(Xtrain, Ytrain, w, b, num_of_iterations, alpha, logistic)
    w = parameters["w"]
    b = parameters["b"]
    d = {"w": w, "b": b, "costs": costs}
    return d

init()