import numpy as np

class LogisticRegression(object):

    def __init__(self, variables, learn_rate = 0.01, error = 1e-2):
        """
            Create a new instance of a LigisticRegressor
        """
        #weights, also add bias
        self._weights = np.zeros(variables + 1)
        self._learn_rate = learn_rate
        self._error = error
        

    def train(self, data, labels):
        """
            Train the current logistic regresion model
        """

        # Current error
        error = 1e4

        # Max iterations
        max_iter = 1e4
        # Current iteration
        curr_iter = 0

        # Cost of the previous train iteration
        old_cost = 1e4

        # While we haven't accuired the required error
        # or haven't reahced hte maximum number of iterations
        # keep training.

        while error > self._error and curr_iter < max_iter:

            # Cost and gradient
            cost = 0 
            gradient = np.zeros(self._weights.shape[0])
        
            # Calculate the cost and the gradient
            for idx in range(0, data.shape[0]): 
                y_est = self._sigmoid(np.dot(self._weights, data[idx]))
                y = labels[idx]
                term = -(y*np.log(y_est) + (1-y)*np.log(1-y_est))

                cost += term
                gradient += term*data[idx]
        
            cost = cost/data.shape[0]
            gradient = gradient/data.shape[0]

            # Update the weights
            self._weights += self._learn_rate*gradient

            # Update the error
            error = abs(cost - old_cost)
            old_cost = cost
            curr_iter += 1


    def predict(self, data):
        """
            Predict the labels for the given data.
        In:
            param: -- A matrix where each row represent a row in the data
                      set for which a label needs to be predicted. The last
                      value of each row must be a 1.
        Out:
            return: An array with predicted labels.
        """
        labels = list()
        for row in data:
            labels.append(self._sigmoid(np.dot(self._weights, row))> 0.5)

        return np.array(labels)


    def _sigmoid(self, x):
        """
            Evaluate the sigmoid activation function.
        In:
            param: x -- The point(s) to evaluate it for.
        Out:
            return: The sigmoid activation function evaluated at the
                    given point(s).
        """
        # 1e-7 is to correct for exponents that are zero.
        # This was needed as some values of columns where taken to the exponent.

        return 1/(1+np.exp(-x) + 1e-7) 