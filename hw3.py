import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from random import *

datafile = "MLdata.csv"
testing_amount = 0.2
validation_amount = 0.25

def random_num_list(length, max):
    count = 0;
    random_list = []
    while(count < length):
        x = randint(0, max)
        random_list.append(x)
        count += 1
    return random_list


def split_data(data, index_list):
    new_list = []
    for i in index_list:
        new_list.append(data[i])
    for i in sorted(index_list, reverse=True):
        data = np.delete(data, i, 0)
    return new_list, data


def main():
    # Reads in data file without header
    df = pd.read_csv(datafile, header=0)
    # Stores a list of header names
    headers = list(df.columns.values)

    numpy_array = df.as_matrix()

    # Split the dataset into a tuple of a data set and a test set
    num_test_set = int(testing_amount*len(numpy_array))
    test_index_list = random_num_list(num_test_set, len(numpy_array)-1)
    testing_split = split_data(numpy_array, test_index_list)
    test_set = testing_split[0]
    numpy_array = testing_split[1]
    # Split the dataset into training and validation
    num_validation_set = int(validation_amount * len(numpy_array))
    validation_index_list = random_num_list(num_validation_set, len(numpy_array)-1)
    training_split = split_data(numpy_array, validation_index_list)
    validation_set = training_split[0]
    training_set = training_split[1]
    print(len(validation_set))
    print(len(test_set))
    print(len(training_set))



if __name__ == '__main__':
    main()
