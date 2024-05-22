# DSOR651 Final Project
The following content outlines the context of the final project.  

## 1. Algorithm Purpose
The purpose of the algorithm is to generate a user-specified number of random forest models in parallel on a given dataset. 

## 2. Hyperparameters
There are four user-specified hyperparameters:

1.    *n_estimators* --   the number of trees in each random forest
2.    *max_depth* --  the maximum vertical levels of the tree
3.    *min_samples_split* --  the minimum number of samples required to split a node
4.    *n_partitions* --  the number of random trees to be ran in parallel

## 3. Background

## 4. Pseudocode

## 5. Example Code

    from random_forest_parallel import train_random_forest
    import numpy as np

    if __name__ == '__main__':

        X = np.random.rand(1000, 20)
        y = np.random.randint(0, 2, size=1000)
    
        n_estimators = 10 
        max_depth = 10
        min_samples_split = 3
        n_partitions = 7
    
        # Calling function from random_forest_parallel.py
        accuracies = train_random_forest(X, y, n_estimators = n_estimators, 
                                         max_depth = max_depth, 
                                         min_samples_split = min_samples_split, 
                                         n_partitions = n_partitions)
    
        for i, accuracy in enumerate(accuracies):
            print(f"Accuracy for partition {i}: {accuracy}")
            
        print(f"The highest accuracy among the parallel processed random forests is", max(accuracies),
              "from partition", accuracies.index(max(accuracies)), ".")

## 6. Visualization of Algorithm

## 7. Benchmark Results

## 8. Lessons Learned

## 9. Unit-Testing
There are 3 unit tests to check the capability of the code to handle errors and logic.  
- The first is a unit test to ensure that the accuracy of each random forest model is between 0 and 1.

- The next is a unit test to check that the dimensions of the data entered (X, y) are compatible.

- The last is a unit test to enforce that all user-defined hyperparameters are non-negative. 
