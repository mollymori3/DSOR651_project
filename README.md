# DSOR651 Final Project
The following content outlines the context of the final project.  

## 1. Algorithm Purpose
The purpose of the algorithm is to use parallel processing for generating random forests on a user-specified number of partitions.  The algorithm allows the user to define four different hyperparameters.  After all models are generated on the partitions, the accuracies are reported and compared for best result.  

## 2. Hyperparameters
There are four user-specified hyperparameters:

1.    *n_estimators* --   the number of trees in each random forest
2.    *max_depth* --  the maximum vertical levels of the tree
3.    *min_samples_split* --  the minimum number of samples required to split a node
4.    *n_partitions* --  the number of random trees to run in parallel

## 3. Background
To build this algorithm, I first learned how to use `RandomForestClassifier()` from Python's `sklearn.ensemble` package given an arbitrary numerical dataset. I explored the various hyperparameters, as defined above, and decided on *accuracy* as my metric of choice for the model.

With parallel processing in mind, I coded partitioning of the arbitrary data based on a globally defined variable, *n_partitions*.  With each partition made, I was able to implement `pool.starmap` from the `multiprocessing` module, which essentially mapped tuples of information to respective models.  That is, the pooled parallel processing ensured that each partition trained a unique tree and yielded its own accuracy.  

For my first three datasets -- randomly generated numbers, pizza data, and wine data -- the algorithm worked as intended.  That's because all of these datasets had strictly *numerical* data.  It wasn't until I tried to pass my challenge dataset through that I realized there were no precautinos for *categorical* data.  With this, I coded a `LabelEncoder()` from the `scikitlearn.preprocessing` package which converted all categorical variables to labeled numerical data.  This addition allowed the algorithm to run on the heart data.  

## 4. Pseudocode

   **Training provided data**

     import necessary packages
     
       1. `train_partition()` function to train data
               a. Split data into training and testing
               b. Fit models based on hyperparameters
       2. `train_random_forest()` function to feed partitions back to `train_partition `
               a. Label encode any categorical data
               b. Split data into partitions
       3. Create pool with processes equal to number of partitions
       4. Return results.

   **Results of provided data**
   
    import necessary packages
    import random forest code
    *Initiate parallel processing*
        1. Define X, y
        2. Specify hyperparameters
        3. Call random forest function in random forest module
        4. For all models,
            print accuracies
        5. Print highest accuracy and which partition it came from.

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

<img width="407" alt="image" src="https://github.com/mollymori3/DSOR651_project/assets/144690206/655ec083-e605-492f-bb0e-7b518c024479">


## 7. Benchmark Results

## 8. Lessons Learned

## 9. Unit-Testing
There are 3 unit tests to check the capability of the code to handle errors and logic.  
- The first is a unit test to ensure that the accuracy of each random forest model is between 0 and 1.

- The next is a unit test to check that the dimensions of the data entered (X, y) are compatible.

- The last is a unit test to enforce that all user-defined hyperparameters are non-negative. 
