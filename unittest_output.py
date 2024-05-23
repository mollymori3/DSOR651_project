'''
    Unit test to ensure that all reported accuracies are between 0 and 1.
'''

import unittest
from sklearn.datasets import make_classification
from random_forest_parallel import train_random_forest

class TestRandomForestTraining(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(n_samples = 1000, n_features = 20, n_informative = 15, n_redundant = 5, random_state = 13)
    
    def test_train_random_forest(self):
        # Call the train_random_forest function
        n_estimators = 10
        max_depth = 5
        min_samples_split = 3
        n_partitions = 4
        results = train_random_forest(self.X, self.y, n_estimators, max_depth, min_samples_split, n_partitions)

        # Check if the results are as expected
        self.assertEqual(len(results), n_partitions, "Number of results should match number of partitions")
        for accuracy in results:
            self.assertGreaterEqual(accuracy, 0, "Accuracy should be non-negative")
            self.assertLessEqual(accuracy, 1, "Accuracy should not exceed 1")

if __name__ == '__main__':
    unittest.main()
