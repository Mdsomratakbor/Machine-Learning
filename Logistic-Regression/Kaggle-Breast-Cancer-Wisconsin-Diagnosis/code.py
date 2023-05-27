import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def initialize(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    
    # Forward propagation
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    
    # Backward propagation
    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.sum(A - Y)
    
    grads = {"dw": dw, "db": db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        if i % 1000 == 0:
            costs.append(cost)
        
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
    
    parameters = {"w": w, "b": b}
    return parameters, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    
    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    
    return Y_prediction

def model(X_train, Y_train, num_iterations, learning_rate, print_cost=False):
    dim = X_train.shape[0]
    w, b = initialize(dim)
    
    parameters, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    w = parameters["w"]
    b = parameters["b"]
    
    d = {"w": w, "b": b, "costs": costs}
    return d

def init():
    df = pd.read_csv("./data.csv")
    
    # Dropping unnecessary columns
    df = df.drop(["id", "Unnamed: 32"], axis=1)
    
    # Mapping M to 1 and B to 0 in the diagnosis column
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Splitting the data into training and test sets
    train, test = train_test_split(df, test_size=0.3, random_state=1)
    
    # Training data
    train_x = train.loc[:, 'radius_mean':'fractal_dimension_worst']
    train_y = train.loc[:, 'diagnosis']
    
    # Testing data
    test_x = test.loc[:, 'radius_mean':'fractal_dimension_worst']
    test_y = test.loc[:, 'diagnosis']
    
    # Converting training and test data to numpy arrays
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    test_x = np.asarray(test_x)
    test_y = np.asarray(test_y)
    
    # Calling the model function to train a Logistic Regression Model on Training Data
    d = model(train_x.T, train_y.reshape(1, -1), num_iterations=10000, learning_rate=0.000001, print_cost=True)
    
    costs = d["costs"]
    w = d["w"]
    b = d["b"]
    
    # Drawing the plot between cost and number of iterations
    plt.plot(costs)
    plt.title("Cost vs #Iterations")
    plt.xlabel("Number of Iterations (* 1000)")
    plt.ylabel("Cost")
    plt.show()
    
    # Now, calculating the accuracy on Training and Test Data
    Y_prediction_train = predict(w, b, train_x.T)
    Y_prediction_test = predict(w, b, test_x.T)
    
    train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - train_y.reshape(1, -1))) * 100
    test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - test_y.reshape(1, -1))) * 100
    
    print("\nTrain accuracy: {}%".format(train_accuracy))
    print("\nTest accuracy: {}%".format(test_accuracy))

# Calling the init function to start the program
init()