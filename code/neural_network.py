import preprocessing as pr
import tensorflow as tf
import keras
from keras import layers
import keras.backend as K
import numpy as np

def custom_RMSE(y_true, y_pred):
    """
    Custom loss function for a tensorflow keras model, that computes the root of the Mean Square Error between the predicted values from the neural network and the target values.

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
    num_of_nodes_per_layer (int or list): Builds a single hidden layer if type of integer, or len(num_of_nodes_per_layer) if type of list, with num_of_nodes_per_layer[i] = #nodes in hidden layer i.
    learning_rate (float): Adjusts the learning rate of the Adam optimizer.

    Returns:
    model (Instance of Sequencial class): Returns the constructed and compiled model of the neural network, tuned by the hyperparameters defined in the input.
    
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=(1000,)))
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
    fitting = model.fit(
    x_train,
    y_train,
    batch_size=batch,
    epochs=epoch,
    )
    return fitting

def evaluate_model(model,x_test,y_test):
    loss = model.evaluate(x_test,y_test)
    return loss





# Testing the network

model = set_model(125,0.001)

for fold_data in pr.fold_dataset:
    x_training = fold_data["X_train"]
    y_training = fold_data["y_train_mean"]
    x_testing = fold_data["X_test"]
    y_testing = fold_data["y_test_mean"]
    indx= fold_data["fold_index"]
    print(f"Fold {indx}:")
    
    history = fit_model(model, x_training, y_training, batch=32, epoch=50)
    results = evaluate_model(model, x_testing, y_testing)
    print("test loss:", results)
    
    