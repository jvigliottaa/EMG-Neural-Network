import numpy as np

"""
Load data from npy file and return training_data, validation_data, and test_data

training_data: is a list containing 20,000 tuples (x,y)
    x: 112 dimensional array containing features extracted from EMG
    y: 5 dimensional array cotaining unit vector for correct gesture
"""

def load_data_from_npy_keras(fileName):
    data_set = np.load(fileName, allow_pickle=True)
    # Randomize data set for each trial
    np.random.shuffle(data_set)

    input_training_data = [x[0] for x in data_set]
    output_training_data = [list(vectorized_result(int(y[1]))) for y in data_set]

    return np.array(input_training_data), np.array(output_training_data)


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((5))
    e[j-1] = 1.0
    return e