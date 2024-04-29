import preprocessing as pr
import tensorflow as tf
import keras
from keras import layers
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt 

def custom_RMSE(y_true, y_pred):
    """
    Calculate the root mean square error between predictions and true values.

    Parameters:
    y_true (nparray or tensor): The target values for the training.
    
    y_pred (nparry or tensor): predicted values of the neural network.

    Returns:
    float: The Root Mean Square Error (RMSE) between the inputs.
        

    Example:
    >>> custom_RMSE(np.array([3, -0.5, 2, 7])  , np.array([2.5, 0.0, 2, 8]))
    0.6123
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def set_model(num_of_nodes_per_layer, learning_rate):
    """
    Construct and compile a custom neural network that performs a linear regression task of fixed input and output nodes, based on the number of nodes per hidden level and the number of hidden levels.

    Parameters:
    num_of_nodes_per_layer (int or list): Builds the hidden layers of the model, 1 for int, # values in the list
    learning_rate (float): Specifies the learning rate of the Adam optimizer.

    Returns:
    Model (Instance of Sequencial class): Returns the constructed model of the neural network.
    
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=(8000,)))
    if isinstance(num_of_nodes_per_layer, int):
        model.add(layers.Dense(num_of_nodes_per_layer, activation="relu"))
    elif isinstance(num_of_nodes_per_layer, list):
        for nodes in num_of_nodes_per_layer:
            if nodes != 0:
                model.add(layers.Dense(nodes, activation="relu"))
            else: break
    model.add(layers.Dense(1, activation="linear"))
    
    model.compile(
    optimizer=keras.optimizers.Adam(learning_rate), 
    
    loss=custom_RMSE,
    )
    return model       
    

def fit_model(model, x_train, y_train, x_test, y_test, batch, epoch):
    """
    Train the given model for a particular given dataset(x_train, y_train), over specified epochs and batches .

    Parameters:
    model (Model Object): The specified model 
    x_train (array): The training data.
    y_train (array): The training labels.
    x_test (array): The testing data.
    y_test (array): The testing labels.
    batch (int): The size of the batch.
    epoch (int): The number of epochs.

    Returns:
    History: Contains the training loss and validation values per epoch. 
    
    """
    history = model.fit(
    x = x_train,
    y = y_train,
    batch_size=batch,
    epochs=epoch,
    validation_data = (x_test,y_test)
    )
    return history

def evaluate_model(model,x_test,y_test):
    """
    Evaluate the given model for a particular given dataset(x_test, y_test).

    Parameters:
    model (Model Object): The specified model. 
    x_test (array): The testing data.
    y_test (array): The testing labels.

    Returns:
    A testing loss value as well as metrics values if specified. 
    
    """
    loss = model.evaluate(x_test,y_test)
    return loss


def train_model(nodes_per_layer, learning_rate, dataset, batch_sizes, epochs):
    """
    Builds, trains and evaluates a model on a dataset based on the parameters given.

    Parameters:
    nodes_per_layer (int or list): Builds the hidden layers of the model, 1 for int, # values in the list. 
    learning_rate (float): Specifies learning rate.
    batch_sizes (int): The size of the batches
    epochs (int): The number of epochs
    
    Returns:
    tuple: A list of the mean training losses over the folds, for every epoch and the mean testing losses over the folds for every epoch. 
    
    """
    test_losses = []
    training_losses = []
        
    for data in dataset:
        x_training = data["X_train"]
        y_training = data["y_train"]
        x_testing = data["X_test"]
        y_testing = data["y_test"]
        indx = data["fold_index"]
        print(f"Fold {indx}:") 
        model = set_model(nodes_per_layer,learning_rate)
        history = fit_model(model, x_training, y_training, x_testing, y_testing, batch=batch_sizes, epoch=epochs)
        training_losses.append(history.history['loss'])
        test_losses.append(history.history['val_loss'])
    # Mean value of loss for every epoch
    mean_training_losses = np.mean(training_losses, axis=0) 
    mean_testing_losses = np.mean(test_losses, axis=0)
    
    return mean_training_losses, mean_testing_losses  


# new train_model_early stopping that outputs the history for each fold individually to plot  

mean_train_loss, mean_test_loss = train_model([500,300,100], 0.001, pr.fold_dataset, 32, 50)

plt.figure(figsize=(12, 4))

for i, (train_loss, test_loss) in enumerate(zip(mean_train_loss, mean_test_loss)):
    plt.plot(train_loss, label=f'Mean Train Loss Network {i + 1}')
    plt.plot(test_loss, linestyle='--', label=f'Mean Test Loss Network {i + 1}')

plt.legend(fontsize = 6)
plt.title('Mean Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')