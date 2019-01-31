#!python

'''

prepare data for network

    load all datasets
    for each dataset, immedietly as it's loaded:
            break off last 30% for testing
            make sure dataset is divisible by feature/label, cut remainder
            break into features and labels
            normalize features
            rewrite labels as ratio of value from previous day
            add to list of feature - label tuples
    shuffle list
    split into set of features and set of labels
    make both sets into arrays
    return them

'''

import os
import random
import pickle
import numpy as np
import pandas as pd


def get_data(data_dir):
    data_set = []
    
    for file in os.listdir(data_dir):
        print(file[:-4])
        
        load_dir = data_dir + '/' + file
        df = pd.read_csv(load_dir)

        df = clean_data(df)
        df = normalize_data(df)
        df = size_data(df)
        for batch in df:
            data_set.append(batch)

    random.shuffle(data_set)
    # is this causing the error?
    # shouldn't be unless it's shuffling the inside contents??
    data_set = np.array(data_set)
    print(len(data_set))
    
    return data_set


def clean_data(df, max_drop=10):
    df = df[['Open', 'Close', 'Low', 'High', 'Volume', 'Adj Close']]
    # is above line necessary at all?
    new_df = []
    for i, row in df.iterrows():
        new_row = []
        for value in row:
            new_row.append(value)
        new_df.append(new_row)

    old_len = len(new_df)
    
    new_df = [x for x in new_df if 0 not in x]

    if old_len - len(new_df) > max_drop:
        print('dropped, too many 0 values')
        return []
    
    return new_df


def normalize_data(df):
    new_data = []
    i = 0
    
    for row in df:
        new_row = []
        if i >= 1:
            for index in range(len(row)):
                value = (row[index] / prev[index]) - 1
                # change this to a log scale?
##                if value < 0:
##                    value = -1
##                if value > 0:
##                    value = 1
                value *= 10
                # tune multiplication value to work with tanh
                new_row.append(value)
                # as a final, run this through a stepper?
                # if negative -1, if positive 1
                # shows if stock increases or decreases that day
                
                
            new_row = np.array(new_row)
            new_data.append(new_row)
        
        prev = row
        i += 1
    new_data = np.array(new_data)
    return new_data


def size_data(df, train_period=7):
    batches = []
    batch = []
    for day in df:
        if len(batch) <= train_period:
            batch.append(day)
        else:
            batch = np.array(batch)
            batches.append(batch)
            batch = []
            batch.append(day)

    batches = np.array(batches)
    return batches


if __name__ == '__main__':

    load_data_dir = 'SP500 Data'
    save_data_dir = 'RNN Data.pickle'

    data = get_data(load_data_dir)
    # save features and labels separately?

    save_file = open(save_data_dir, 'wb')
    pickle.dump(data, save_file)
    save_file.close()
    print('saved')






