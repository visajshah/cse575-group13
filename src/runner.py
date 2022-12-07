# This is the main file which needs to be run

# Importing the required libraries and functions
import os
import pandas as pd

from create_datafile import download_files, datafile
from custom_gridsearch_cv import custom_gridsearch_cv
from model import model
from plots import gridsearch_plots
from utils import sort_data, split_data, tensorDataset

# Generate data files
download_files()
datafile()

# Making data useable for the model
print("Reading datafile...")
if not os.path.isfile("data/sorted_data.csv"):
        data = pd.read_csv("data/data.csv", header = None)
        data.columns = ["movieID", "userID", "rating", "date"]
        data = data.iloc[:,[1, 0, 2, 3]]
        data['year'] = data.apply(lambda row: row.date[:4], axis = 1)
        data = sort_data(data)
        data.to_csv("data/sorted_data.csv")
sortedData = pd.read_csv("data/sorted_data.csv").sample(frac = 0.01)
print("Data is ready for use")

#Finding out the best paramaters for our model using our custom GridSearchCV function
parameters = {
        'batch size': [1024, 2048, 4096, 8192],
        'epoch': [10, 50, 100],
        'lr': [0.01, 0.1]
}
best_params, params = custom_gridsearch_cv(
        df = sortedData,
        parameters = parameters
)
print("Best parameters are: ", best_params)

# Plotting the variation of RMSE with hyperparameters
gridsearch_plots(parameters = params)

# Preparing data for the final model
sortedData = pd.read_csv("data/sorted_data.csv").sample(frac = 0.1)
trainSet, valSet, testSet, movieCount, userCount = split_data(sortedData)
trainSet = tensorDataset(trainSet)
valSet = tensorDataset(valSet)
testSet = tensorDataset(testSet)
print("RMSE = ", model(
        trainSet = trainSet,
        valSet = valSet,
        testSet = testSet,
        movieCount = movieCount,
        userCount = userCount,
        parameters = best_params,
        plot = True
))

print("Success")
