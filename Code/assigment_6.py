import numpy as np

class LogisticRegression(object):

    def __init__(self, variables):

        #weights, also add bias
        self._weights = np.zeros(variables + 1)
        

    def train(self):
        pass

    def predict(self):
        pass