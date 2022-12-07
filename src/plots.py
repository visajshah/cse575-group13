# Importing required libraries and functions

import matplotlib.pyplot as plt

# Function to generate plots for GridSearchCV
def gridsearch_plots(parameters):
    batch_size_rmse = []
    epoch_rmse = []
    lr_rmse = []
    rmse = []
    for key in parameters.keys():
        rmse.append(key)
        batch_size_rmse.append(parameters[key]['batch size'])
        epoch_rmse.append(parameters[key]['epoch'])
        lr_rmse.append(parameters[key]['lr'])
    
    # Batch size vs RMSE
    plt.scatter(batch_size_rmse, rmse)
    plt.xlabel("Batch Size")
    plt.ylabel("RMSE")
    plt.title("Batch Size vs RMSE")
    plt.show()

    # Epoch vs RMSE
    plt.scatter(epoch_rmse, rmse)
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Epoch vs RMSE")
    plt.show()

    # LR vs RMSE
    plt.scatter(lr_rmse, rmse)
    plt.xlabel("LR")
    plt.ylabel("RMSE")
    plt.title("LR vs RMSE")
    plt.show()

# Function to generate plots for final model
def final_model_plots(training_rmse, training_loss, val_rmse, val_loss, testing_rmse, testing_loss, epochs):
    x_axis = list(range(epochs))

    # Training RMSE vs Epoch
    plt.plot(x_axis, training_rmse)
    plt.xlabel("Epochs")
    plt.ylabel("Training RMSE")
    plt.title("Training RMSE vs Epochs")
    plt.show()

    # Training Loss vs Epoch
    plt.plot(x_axis, training_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Epochs")
    plt.show()

    # Val RMSE vs Epoch
    plt.plot(x_axis, val_rmse)
    plt.xlabel("Epochs")
    plt.ylabel("Val RMSE")
    plt.title("Val RMSE vs Epochs")
    plt.show()

    # Val Loss vs Epoch
    plt.plot(x_axis, val_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Val Loss")
    plt.title("Val Loss vs Epochs")
    plt.show()

    # Testing RMSE vs Epoch
    plt.plot(x_axis, testing_rmse)
    plt.xlabel("Epochs")
    plt.ylabel("Testing RMSE")
    plt.title("Testing RMSE vs Epochs")
    plt.show()

    # Testing Loss vs Epoch
    plt.plot(x_axis, testing_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Testing Loss")
    plt.title("Testing Loss vs Epochs")
    plt.show()
