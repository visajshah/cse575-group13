# Importing the required libraries
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def sort_data(df):
    print("Sorting datafile...")
    return df.sort_values(by = ['date'])

def split_data(df):
    # Splitting the dataset based on the date column
    # The training data will have the first half
    # The val and test data will consist of the second half
    dateSplit = df['date'].tolist()[int(len(df) * 0.5)]
    trainDF = df[df['date'] <= dateSplit]
    remainingDF = df[(df['date'] > dateSplit)
                   & df['userID'].isin(trainDF['userID'].unique())
                   & df['movieID'].isin(trainDF['movieID'].unique())].sample(frac=1)
    testDF = remainingDF.iloc[:int(0.5 * len(remainingDF))]
    valDF = remainingDF.iloc[int(0.5 * len(remainingDF)):]

    # Resetting the indices
    movieIDs = set().union(*[trainDF['movieID'].unique(), valDF['movieID'].unique(), testDF['movieID'].unique()])
    userIDs = set().union(*[trainDF['userID'].unique(), valDF['userID'].unique(), testDF['userID'].unique()])
    movieReset = {id: idx for idx, id in enumerate(movieIDs)}
    userReset = {id: idx for idx, id in enumerate(userIDs)}

    trainDF['movieIdx'] = trainDF['movieID'].map(lambda x: movieReset[x])
    trainDF['userIdx'] = trainDF['userID'].map(lambda x: userReset[x])
    valDF['movieIdx'] = valDF['movieID'].map(lambda x: movieReset[x])
    valDF['userIdx'] = valDF['userID'].map(lambda x: userReset[x])
    testDF['movieIdx'] = testDF['movieID'].map(lambda x: movieReset[x])
    testDF['userIdx'] = testDF['userID'].map(lambda x: userReset[x])
    
    return trainDF, valDF, testDF, len(movieIDs), len(userIDs)

def split_data_for_cv(df, trainingYears, testingYear):
    trainDF = df[(df['year'] == trainingYears[0]) | (df['year'] == trainingYears[1])]
    remainingDF = df[(df['year'] == testingYear)
                   & df['userID'].isin(trainDF['userID'].unique())
                   & df['movieID'].isin(trainDF['movieID'].unique())].sample(frac=1)
    testDF = remainingDF.iloc[:int(0.5 * len(remainingDF))]
    valDF = remainingDF.iloc[int(0.5 * len(remainingDF)):]

    movieIDs = set().union(*[trainDF['movieID'].unique(), valDF['movieID'].unique(), testDF['movieID'].unique()])
    userIDs = set().union(*[trainDF['userID'].unique(), valDF['userID'].unique(), testDF['userID'].unique()])
    movieReset = {id: idx for idx, id in enumerate(movieIDs)}
    userReset = {id: idx for idx, id in enumerate(userIDs)}

    trainDF['movieIdx'] = trainDF['movieID'].map(lambda x: movieReset[x])
    trainDF['userIdx'] = trainDF['userID'].map(lambda x: userReset[x])
    valDF['movieIdx'] = valDF['movieID'].map(lambda x: movieReset[x])
    valDF['userIdx'] = valDF['userID'].map(lambda x: userReset[x])
    testDF['movieIdx'] = testDF['movieID'].map(lambda x: movieReset[x])
    testDF['userIdx'] = testDF['userID'].map(lambda x: userReset[x])

    return trainDF, valDF, testDF, len(movieIDs), len(userIDs)

class tensorDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.length = len(df)
    
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        row = self.df.iloc[i]
        movie_idx = row['movieIdx']
        user_idx = row['userIdx']
        rating_val = row['rating']

        return {
            'x': torch.tensor([user_idx, movie_idx]),
            'r': torch.tensor(rating_val)
        }

class Evaluator:
    def __init__(self):
        self.predicts = []
        self.labels = []

    def append(self, label, predict):
        if type(label) == torch.Tensor:
            if label.dim() == 0:
                self.predicts.append(predict)
                self.labels.append(label)
            elif label.dim() == 1:
                self.predicts += predict.tolist()
                self.labels += label.tolist()
        else:
            self.predicts.append(predict)
            self.labels.append(label)

    def calulate(self, kind):
        self.predicts = np.array(self.predicts)
        self.labels = np.array(self.labels)

        if kind == 'rmse':
            value = np.sqrt(np.mean((self.labels - self.predicts)**2))
        else:
            raise KeyError(f'invalid kind: {kind}')

        self.clear()
        return value

    def clear(self):
        self.predicts = []
        self.labels = []

class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.evaluator = Evaluator()
        self.writer = SummaryWriter()

    def single_epoch(self, dataloader, tag, epoch):
        running_loss = 0

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for it, data in pbar:
            # inference
            x = data['x']
            r = data['r']
            r_pred = self.model(x)

            # evaluate
            self.evaluator.append(r, r_pred)

            # calculate loss
            loss = self.criterion(r.float(), r_pred.float())
            running_loss += loss.item()
            rmse = self.evaluator.calulate('rmse')

            # update
            if tag == 'train':
                self.optimizer.zero_grad()      # clear out the gradients
                loss.backward()                 # calculate gradients
                self.optimizer.step()           # update parameters based on gradients

            # print current performance of the net on console
            pbar.set_description(f'epoch {epoch} iter {it}: {tag} loss {running_loss/len(dataloader):.5f} rmse {rmse:.5f}')

        running_loss *= 1 / len(dataloader)
        self.writer.add_scalar(f'Loss/{tag}', running_loss, epoch)
        self.writer.add_scalar(f'rmse/{tag}', rmse, epoch)
        return rmse, running_loss
            