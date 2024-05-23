
'''
                                        CHALLENGE SET!

    Testing the algorithm on the "cars" dataset consisting of 302 observations, 8 features, and 1
    reponse variable.  There are 3 continuous, 4 categorical features, and 1 binary feature. 
    The response is categorical.  The goal is to classify by transmission type.
'''


from random_forest_parallel import train_random_forest
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv(r"C:\Users\molly\Downloads\cars.csv")
    X = data.drop(columns='Transmission')
    y = data['Transmission']

    n_estimators = 1000
    max_depth = None
    min_samples_split = 2
    n_partitions = 50

    accuracies, times = train_random_forest(X, y, n_estimators = n_estimators, 
                                     max_depth = max_depth, 
                                     min_samples_split = min_samples_split, 
                                     n_partitions = n_partitions)

    for i, (accuracy, time) in enumerate(zip(accuracies, times)):
        print(f"Accuracy for partition {i}: {accuracy:.4f}, Time taken: {time:.2f} seconds")
        
    max_accuracy = max(accuracies)
    best_partition = accuracies.index(max_accuracy)
    print(f"The highest accuracy among the parallel processed random forests is {max_accuracy:.4f} from partition {best_partition}.")

    min_time = min(times)
    min_time_partition = times.index(min_time)
    print(f"The shortest time taken among the parallel processed random forests is {min_time:.2f} seconds from partition {min_time_partition}.")
