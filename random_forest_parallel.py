
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import multiprocessing as mp
from sklearn.preprocessing import LabelEncoder
import time
import pandas as pd


# Function to train RandomForest on a given dataset partition
def train_partition(data_partition, n_estimators, max_depth, min_samples_split):
    start_time = time.time()

    X_train, y_train, X_test, y_test = data_partition

    clf = RandomForestClassifier(n_estimators = n_estimators, 
                                 max_depth = max_depth, 
                                 min_samples_split = min_samples_split)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    end_time = time.time()
    duration = end_time - start_time  # Duration of the process

    return accuracy, duration  # Effectiveness and efficiency benchmarks



def train_random_forest(X, y, n_estimators = 100, max_depth = None, min_samples_split = 2, n_partitions = 4):

    if isinstance(X, pd.DataFrame):
        label_encoder = LabelEncoder()
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = label_encoder.fit_transform(X[col])


    # Function to partition the data into n_partitions subsets
    def partition_data(X, y, n_partitions):
        partitions = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 13)
        partition_size = len(X_train) // n_partitions 

        for i in range(n_partitions):
            start = i * partition_size
            end = (i + 1) * partition_size if i != n_partitions - 1 else len(X_train) # takes care of "leftover" data from partitions
            X_train_part = X_train[start:end]
            y_train_part = y_train[start:end]
            partitions.append((X_train_part, y_train_part, X_test, y_test))
        
        return partitions
    
    # Partition the data
    data_partitions = partition_data(X, y, n_partitions)

    # Create a pool of processes and train the model on each partition in parallel
    with mp.Pool(processes=n_partitions) as pool:
        results = pool.starmap(train_partition, [(partition, n_estimators, max_depth, min_samples_split) for partition in data_partitions]) # for each partition, apply random forest model.
                    # starmap is like map for tuples (i.e. partitions to hyperparameters)
    
    # Extract accuracies and durations from results
    accuracies = [result[0] for result in results]
    durations = [result[1] for result in results]
    
    return accuracies, durations  # Return accuracies and durations