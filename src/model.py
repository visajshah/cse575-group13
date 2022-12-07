# Importing the required libraries and functions
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from plots import final_model_plots
from utils import Trainer

# LatentEmbeddingModel defined
class LatentEmbeddingModel(nn.Module):
    def __init__(self, movieCount, userCount, n_embed):
        super(LatentEmbeddingModel, self).__init__()

        # embedding layer (B, 2) -> (B, 2, N_EMBED)
        self.embedding_movie = nn.Embedding(movieCount, n_embed)
        self.embedding_user = nn.Embedding(userCount, n_embed)
        
        # global avg, bias 
        self.mu = nn.Embedding(1, 1)
        self.b_movie = nn.Embedding(movieCount, 1)
        self.b_user = nn.Embedding(userCount, 1)

    def forward(self, x):
        # x (B, 2)
        x_movie = self.embedding_movie(x[:, 1].long())                # (B, N_EMBED)
        x_user = self.embedding_user(x[:, 0].long())                # (B, N_EMBED)

        mu = self.mu(torch.tensor([0] * x.shape[0])).view(-1)         # (B)
        b_movie = self.b_movie(x[:, 1].long()).view(-1)               # (B)
        b_user = self.b_user(x[:, 0].long()).view(-1)               # (B)
        
        # x = torch.sum(x_user * x_item, dim=1)  # (B)
        x = mu + b_movie + b_user + torch.sum(x_movie * x_user, dim=1)  # (B)
        return x

# Single function to call the model
def model(trainSet, valSet, testSet, movieCount, userCount, parameters, plot):
    trainLoader = DataLoader(dataset = trainSet, batch_size = parameters['batch size'], shuffle = True)
    valLoader = DataLoader(dataset = valSet, batch_size = parameters['batch size'], shuffle = True)
    testLoader = DataLoader(dataset = testSet, batch_size = parameters['batch size'], shuffle = True)

    model = LatentEmbeddingModel(movieCount, userCount, 3)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr = parameters['lr'])

    trainer = Trainer(model, criterion, optimizer)
    training_rmse = []
    training_loss = []
    val_rmse = []
    val_loss = []
    testing_rmse = []
    testing_loss = []
    for epoch in range(parameters['epoch']):
            rmse, loss = trainer.single_epoch(trainLoader, tag='train', epoch = epoch)
            training_rmse.append(rmse)
            training_loss.append(loss)
            rmse, loss = trainer.single_epoch(valLoader, tag='val', epoch = epoch)
            val_rmse.append(rmse)
            val_loss.append(loss)
            rmse, loss = trainer.single_epoch(testLoader, tag='test', epoch = epoch)
            testing_rmse.append(rmse)
            testing_loss.append(loss)
    
    if plot:
        final_model_plots(
            training_rmse = training_rmse,
            training_loss = training_loss,
            val_rmse = val_rmse,
            val_loss = val_loss,
            testing_rmse = testing_rmse,
            testing_loss = testing_loss,
            epochs = parameters['epoch']
        )

    return testing_rmse[-1]
