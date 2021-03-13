import numpy as np

def predict(Features, Weights):
    return Features @ Weights

def cost_function(X, Y, Weights):
    N = len(Y)
    Ypred = predict(X, Weights)
    sq_error = (Y - Ypred)**2
    return 1/(2*N) * sq_error.sum()

def update_weights(X, Y, Weights, learning_rate):
    N = len(Y)
    Ypred = predict(X, Weights)
    error = Ypred - Y
    grad = (error.T @ X).T
    newWeights = Weights - learning_rate *(1/N) * grad
    return newWeights
    
def train(X, Y, learning_rate, n_iterations):
    # Step to log the intermediant result
    step = n_iterations/10
    
    # History of cost function progress
    mse_history = []
    
    # History on weights change
    w_history = []
    
    # Adding bias term (w0)
    bias = np.ones(shape=(len(X),1))
    Features = np.append(bias, X, axis=1)
    
    # Initial setup of Weights to 1
#     Weights = 0.2*np.ones((Features.shape[1],1))
    Weights = np.random.ranf([Features.shape[1],1])
    
    print("iter\t Cost \t\t Weights")
    for i in range(n_iterations):
        Weights = update_weights(Features, Y, Weights, learning_rate)
        cost = cost_function(Features, Y, Weights)
        mse_history.append(cost)
        w_history.append(Weights.flatten())
        if i%step==0:
            print("{}\t{:0.8}\t{}".format(i,cost,Weights.T))
    return Weights, mse_history, w_history