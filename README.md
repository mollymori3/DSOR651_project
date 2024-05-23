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
To build this algorithm, I first learned how to use `RandomForestClassifier()` from Python's `sklearn.ensemble` package given an arbitrary numerical dataset. I explored the various hyperparameters, as defined above, and decided on *accuracy* as my metric of choice for evaluating the effectiveness of the model.

With parallel processing in mind, I coded partitioning of the arbitrary data based on a globally defined variable, *n_partitions*.  With each partition made, I was able to implement `pool.starmap()` from the `multiprocessing` module, which essentially maps tuples of information to respective models.  That is, the pooled parallel processing ensures that each partition trained a unique tree and yields its own accuracy.  

For my first three datasets -- randomly generated numbers, pizza data, and wine data -- the algorithm worked as intended.  That's because all of these datasets had strictly *numerical* data.  It wasn't until I tried to pass my challenge dataset, car data, through that I realized there were no precautions for *categorical* data.  With this, I coded a `LabelEncoder()` from the `scikitlearn.preprocessing` package which converted all categorical variables to labeled numerical data.  This addition allowed the algorithm to run on the car data.  

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

<img width="418" alt="image" src="https://github.com/mollymori3/DSOR651_project/assets/144690206/24df10f9-68c6-4ffa-a75b-802097fd790e">

To compare the benchmark results of efficiency and effectiveness, I used the metrics of time and accuracy, respectively.  In the table above are best results for each dataset given a specified *n_estimators* value.  The other hyperparameters were set to default because after some initial trials, only *n_estimators* proved make noticeable change to total times.  

It is clear that as the number of estimators in the models increases, the time it takes to process the algorithm increases.  This doesn't necessarily imply that the efficiency decreased, though.  Notice that the value of *n_estimators* increases each time by one order of magnitude.  Surprisingly, the times reflect the same trend -- that is, the times are directly proportional to *n_estimators*.  

As for accuracy, there is no noticeable trend.  Logically, as the number of partitions approaches the number of observations in the dataset, the accuracies will begin to near 0 or 1 (i.e. the random forest either did or did not predict the correct categorization).  This could lead to the false idea that the models are getting more effective, when in reality the models lack adequate data.

## 8. Lessons Learned

I learned three main lessons while designing this algorithm:

1. 

 if isinstance(X, pd.DataFrame):
        label_encoder = LabelEncoder()
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = label_encoder.fit_transform(X[col])

## 9. Unit-Testing
There are 3 unit tests to check the capability of the code to handle errors and logic.  
- The first is a unit test to ensure that the accuracy of each random forest model is between 0 and 1.

- The next is a unit test to check that the dimensions of the data entered (X, y) are compatible.

- The last is a unit test to enforce that all user-defined hyperparameters are non-negative. 
