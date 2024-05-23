
from random_forest_parallel import train_random_forest
import numpy as np

# Example with randomly generated data
if __name__ == '__main__':

    X = np.random.rand(1000, 20)  # 1000 data points, 20 features
    y = np.random.randint(0, 2, size=1000)  # 1000 responses, either 0 or 1

    # Hyperparameters tunable by user
    n_estimators = 10 # default 100
    max_depth = None # default None
    min_samples_split = 2 # default 2
    n_partitions = 4 # default 4 ; NOTE that for whatever reason, cannot exceed 61.

    # Calling function from random_forest_parallel.py
    accuracies, times = train_random_forest(X, y, n_estimators = n_estimators, 
                                     max_depth = max_depth, 
                                     min_samples_split = min_samples_split, 
                                     n_partitions = n_partitions)

    # Accuracies and times from each partition
    for i, (accuracy, time) in enumerate(zip(accuracies, times)):
        print(f"Accuracy for partition {i}: {accuracy:.4f}, Time taken: {time:.2f} seconds")
        
    # Report highest accuracy and partition from which it was generated
    max_accuracy = max(accuracies)
    best_partition = accuracies.index(max_accuracy)
    print(f"The highest accuracy among the parallel processed random forests is {max_accuracy:.4f} from partition {best_partition}.")

    # Report the partition with the shortest time taken
    min_time = min(times)
    min_time_partition = times.index(min_time)
    print(f"The shortest time taken among the parallel processed random forests is {min_time:.2f} seconds from partition {min_time_partition}.")