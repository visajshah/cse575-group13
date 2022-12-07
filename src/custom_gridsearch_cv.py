# Importing the required libraries and functions
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import LatentEmbeddingModel, model
from utils import split_data_for_cv, tensorDataset, Trainer

def cv(df, trainingYears, testingYear, params):
    trainSet, valSet, testSet, movieCount, userCount = split_data_for_cv(df = df, trainingYears = trainingYears, testingYear = testingYear)
    trainSet = tensorDataset(trainSet)
    valSet = tensorDataset(valSet)
    testSet = tensorDataset(testSet)
    return model(
        trainSet = trainSet,
        valSet = valSet,
        testSet = testSet,
        movieCount = movieCount,
        userCount = userCount,
        parameters = params,
        plot = False
    )

def custom_gridsearch_cv(df, parameters):
    params = {}

    #These for loops perform the GridSearch
    for i in range(len(parameters['batch size'])):
        for j in range(len(parameters['lr'])):
            for k in range(len(parameters['epoch'])):
                rmse = 0
                model_parameters = {
                    'batch size': parameters['batch size'][i],
                    'lr': parameters['lr'][j],
                    'epoch': parameters['epoch'][k]
                }

                #We use 4-fold Cross Validation based on year
                rmse += cv(df = df,  trainingYears = [1999, 2000], testingYear = 2001, params = model_parameters)
                rmse += cv(df = df,  trainingYears = [2000, 2001], testingYear = 2002, params = model_parameters)
                rmse += cv(df = df,  trainingYears = [2001, 2002], testingYear = 2003, params = model_parameters)
                rmse += cv(df = df,  trainingYears = [2002, 2003], testingYear = 2004, params = model_parameters)
                params[rmse / 4] = model_parameters
                #params[rmse / 2] = model_parameters
                        
    for rmse in sorted(params.keys()):
        return params[rmse], params
        