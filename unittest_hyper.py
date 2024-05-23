'''
    Unit test to ensure that all hyperparameters are non-negative.
'''



import unittest
from random_forest_parallel import train_random_forest

class TestHyperparameters(unittest.TestCase):
    def setUp(self):
        # Set hyperparameters
        self.n_estimators = 10
        self.max_depth = 10
        self.min_samples_split = -3
        self.n_partitions = 7
    
    def test_hyperparameters_positive(self):
        # Check if n_estimators is greater than 0
        self.assertGreater(self.n_estimators, 0, "n_estimators should be greater than 0")
        
        # Check if max_depth is greater than 0 or None
        if self.max_depth is not None:
            self.assertGreater(self.max_depth, 0, "max_depth should be greater than 0 or None")
        
        # Check if min_samples_split is greater than 0
        self.assertGreater(self.min_samples_split, 0, "min_samples_split should be greater than 0")
        
        # Check if n_partitions is greater than 0
        self.assertGreater(self.n_partitions, 0, "n_partitions should be greater than 0")

if __name__ == '__main__':
    unittest.main()
