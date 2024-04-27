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
    

def fit_model(model, x_train, y_train, batch, epoch):
    """
    Train the given model for a particular given dataset(x_train, y_train), over specified epochs and batches .

    Parameters:
    model (Model Object): The specified model 
    x_train (array): The training data.
    y_train (array): The training labels.
    batch (int): The size of the batch.
    epoch (int): The number of epochs.

    Returns:
    History: Contains the training loss values per epoch. 
    
    """
    history = model.fit(
    x_train,
    y_train,
    batch_size=batch,
    epochs=epoch,
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


def train_model(nodes_per_layer, learning_rate, print_summary, dataset, batch_sizes, epochs):
    """
    Builds, trains and evaluates a model on a dataset based on the parameters given.

    Parameters:
    nodes_per_layer (int or list): Builds the hidden layers of the model, 1 for int, # values in the list. 
    learning_rate (float): Specifies learning rate.
    print_summary (boolean): Set to True to view the architecture of the model.
    batch_sizes (int): The size of the batches
    epochs (int): The number of epochs
    
    Returns:
    tuple: A list of the training losses per epoch per fold, a list of testing losses per fold and the mean training losses over the folds, for every epoch . 
    
    """
    test_losses = []
    training_losses = []
    
    model = set_model(nodes_per_layer,learning_rate)
    
    if print_summary:
        model.summary()
        
    for data in dataset:
        x_training = data["X_train"]
        y_training = data["y_train"]
        x_testing = data["X_test"]
        y_testing = data["y_test"]
        indx = data["fold_index"]
        print(f"Fold {indx}:") 
        history = fit_model(model, x_training, y_training, batch=batch_sizes, epoch=epochs)
        training_losses.append(history.history['loss'])
        result = evaluate_model(model, x_testing, y_testing)
        test_losses.append(result)
    # Mean value of loss for every epoch
    mean_training_losses = np.mean(training_losses, axis=0) 
    
    return training_losses, test_losses, mean_training_losses   

# Test the network
# tr_loss, ts_loss, mean_tr_loss = train_model(125, 0.001, True, pr.fold_dataset, 32, 10) 

def plot_loss_over_epoch(mean_tr_loss):
    """
    Plots the mean loss from all the folds for each of the training epochs.

    Parameters:
    mean_train_loss (list): the mean training loss per epoch of the particular model. 
    
    Returns:
    The diagram.
    
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)

    # Plot the training loss over epochs, with the label 'Train Loss'
    plt.plot(mean_tr_loss, label='Train Loss')

    # Create a legend for the plot
    plt.legend()

    # Set the title of the plot to 'Loss Over Epochs'
    plt.title('Loss Over Epochs')

mean_train_losses = []
tr_loss1, ts_loss1, mean_tr_loss1 = train_model(1000, 0.001, True, pr.fold_dataset, 32, 50)
mean_train_losses.append(mean_tr_loss1)
tr_loss2, ts_loss2, mean_tr_loss2 = train_model(750, 0.001, True, pr.fold_dataset, 32, 50)
mean_train_losses.append(mean_tr_loss2)
tr_loss3, ts_loss3, mean_tr_loss3 = train_model(500, 0.001, True, pr.fold_dataset, 32, 50)
mean_train_losses.append(mean_tr_loss3)
tr_loss4, ts_loss4, mean_tr_loss4 = train_model(250, 0.001, True, pr.fold_dataset, 32, 50)
mean_train_losses.append(mean_tr_loss4)

plt.figure(figsize=(12, 4))
 
for i,loss in enumerate(mean_train_losses):
    plt.plot(loss, label='Mean Train Loss Network {}'.format(i+1))

plt.legend()

plt.title('Mean Loss Over Epochs')

mean_test_loss1 = np.mean(ts_loss1)
mean_test_loss2 = np.mean(ts_loss2)
mean_test_loss3 = np.mean(ts_loss3)
mean_test_loss4 = np.mean(ts_loss4)

plt.figure(figsize=(12, 4))
plt.plot(ts_loss1, label='Test Loss Network 1')
plt.plot(ts_loss2, label='Test Loss Network 2')
plt.plot(ts_loss3, label='Test Loss Network 3')
plt.plot(ts_loss4, label='Test Loss Network 4')
plt.legend()
plt.title('Train Loss Over Folds per Model')

mean_loss_ml = []

tr_loss_ml_1, ts_loss_ml_1, mean_tr_loss_ml_1 = train_model([500,100], 0.001, True, pr.fold_dataset, 32, 50)
mean_loss_ml.append(mean_tr_loss_ml_1)
tr_loss_ml_2, ts_loss_ml_2, mean_tr_loss_ml_2 = train_model([500,200], 0.001, True, pr.fold_dataset, 32, 50)
mean_loss_ml.append(mean_tr_loss_ml_2)
tr_loss_ml_3, ts_loss_ml_3, mean_tr_loss_ml_3 = train_model([500,300], 0.001, True, pr.fold_dataset, 32, 50)
mean_loss_ml.append(mean_tr_loss_ml_3)
tr_loss_ml_4, ts_loss_ml_4, mean_tr_loss_ml_4 = train_model([1000,100], 0.001, True, pr.fold_dataset, 32, 50)
mean_loss_ml.append(mean_tr_loss_ml_4)
tr_loss_ml_5, ts_loss_ml_5, mean_tr_loss_ml_5 = train_model([1000,200], 0.001, True, pr.fold_dataset, 32, 50)
mean_loss_ml.append(mean_tr_loss_ml_5)
tr_loss_ml_6, ts_loss_ml_6, mean_tr_loss_ml_6 = train_model([1000,300], 0.001, True, pr.fold_dataset, 32, 50)
mean_loss_ml.append(mean_tr_loss_ml_6)

plt.figure(figsize=(12, 4))
 
for i,loss in enumerate(mean_loss_ml):
    plt.plot(loss, label='Mean Train Loss Multilayered Network {}'.format(i+1))

plt.legend()

plt.title('Mean Loss Over Epochs')