'''

Unit test to make sure that the dimensions of the inputs (X, y) are compatible.  That is, there have to be the same number
of entries in X as in y.

'''

import unittest
import numpy as np
from sklearn.datasets import make_classification
from random_forest_parallel import train_random_forest

class TestRandomForestInvalidInput(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(n_samples = 1000, n_features=  20, n_informative = 15, n_redundant = 5, random_state = 13)
        # 1000 samples, 20 columns (15 of which are helpful for classification, remaining 5 are linear combos. of other features)

    def test_invalid_X_shape(self):
        # Create invalid input with incorrect shape
        X_invalid = np.array([None] * 1000).reshape(-1, 1)  # Incorrect shape
        y_valid = self.y
        
        with self.assertRaises(ValueError):
            train_random_forest(X_invalid, y_valid)
    
    def test_invalid_y_shape(self):
        # Create invalid target with incorrect shape
        X_valid = self.X
        y_invalid = np.array([None] * 1000).reshape(-1, 1)  # Incorrect shape
        
        with self.assertRaises(ValueError):
            train_random_forest(X_valid, y_invalid)
    
    def test_invalid_X_values(self):
        # Create invalid input with None values
        X_invalid = np.array([None] * 1000).reshape(1000, 1)  # Invalid values
        y_valid = self.y
        
        with self.assertRaises(ValueError):
            train_random_forest(X_invalid, y_valid)
    
    def test_invalid_y_values(self):
        # Create invalid target with None values
        X_valid = self.X
        y_invalid = np.array([None] * 1000)  # Invalid values
        
        with self.assertRaises(ValueError):
            train_random_forest(X_valid, y_invalid)

if __name__ == '__main__':
    unittest.main()
