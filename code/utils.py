import tensorflow as tf
import keras
from keras import layers
import keras.backend as K
import keras.callbacks as cb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold 
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

def set_model(num_of_nodes_input_layer, num_of_nodes_per_layer, learning_rate, momentum):
    """
    Construct and compile a custom neural network that performs a linear regression task of fixed input and output nodes, based on the number of nodes per hidden level and the number of hidden levels.

    Parameters:
    num_of_nodes_input_layer (int): The number of nodes in the input layer of the model.
    num_of_nodes_per_layer (int or list): Builds the hidden layers of the model, 1 hidden layer for an int, or as many as the size of the list.
    learning_rate (float): Specifies the learning rate of the Adam optimizer.
    momentum (float): Specifies the momentum of the Adam optimizer.

    Returns:
    Model (Instance of Sequencial class): Returns the constructed model of the neural network.
    
    """
    model = keras.Sequential()
    
    model.add(keras.Input(shape=(num_of_nodes_input_layer,)))
    
    if isinstance(num_of_nodes_per_layer, int):
        model.add(layers.Dense(num_of_nodes_per_layer, activation="relu"))
    elif isinstance(num_of_nodes_per_layer, list):
        for nodes in num_of_nodes_per_layer:
            if nodes != 0:
                model.add(layers.Dense(nodes, activation="relu"))
            else: break
    model.add(layers.Dense(1, activation="linear"))
    
    model.compile(
    optimizer=keras.optimizers.Adam(learning_rate, beta_1 =momentum), 
    
    loss=custom_RMSE,
    )
    return model       

def set_model_with_dropout(num_of_nodes_input_layer, num_of_nodes_per_layer, learning_rate, momentum,dropout_rate_in, dropout_rate_h):
    """
    Construct and compile a custom neural network that performs a linear regression task of fixed input and output nodes, based on the number of nodes per hidden level and the number of hidden levels, using the technique of dropout.

    Parameters:
    num_of_nodes_input_layer (int): The number of nodes in the hidden layer of the model.
    num_of_nodes_per_layer (int or list): Builds the hidden layers of the model, 1 for int, # values in the list
    learning_rate (float): Specifies the learning rate of the Adam optimizer.
    momentum (float): Specifies the momentum of the Adam optimizer.
    dropout_rate_in (float): The dropout rate for the input layer
    dropout_rate_h (float): The dropout rate for the hidden layer/s

    Returns:
    Model (Instance of Sequencial class): Returns the constructed model of the neural network.
    
    """
    model = keras.Sequential()
    
    model.add(keras.Input(shape=(num_of_nodes_input_layer,)))
    model.add(layers.Dropout(dropout_rate_in))
    
    if isinstance(num_of_nodes_per_layer, int):
        model.add(layers.Dense(num_of_nodes_per_layer, activation="relu"))
        model.add(layers.Dropout(dropout_rate_h))
    elif isinstance(num_of_nodes_per_layer, list):
        for nodes in num_of_nodes_per_layer:
            if nodes != 0:
                model.add(layers.Dense(nodes, activation="relu"))
                model.add(layers.Dropout(dropout_rate_h))
            else: break
    model.add(layers.Dense(1, activation="linear"))
    
    model.compile(
    optimizer=keras.optimizers.Adam(learning_rate, beta_1 =momentum), 
    
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

def fit_model_with_stopping(model, x_train, y_train, x_test, y_test, batch, epoch, callback):
    """
    Train the given model for a particular given dataset(x_train, y_train), over specified epochs and batches using an early stopping callback.

    Parameters:
    model (Model Object): The specified model 
    x_train (array): The training data.
    y_train (array): The training labels.
    x_test (array): The testing data.
    y_test (array): The testing labels.
    batch (int): The size of the batch.
    epoch (int): The number of epochs.
    callback (EarlyStopping object): Specified callback

    Returns:
    History: Contains the training loss and validation values per epoch. 
    
    """
    history = model.fit(
    x = x_train,
    y = y_train,
    batch_size=batch,
    epochs=epoch,
    validation_data = (x_test,y_test),
    callbacks=[callback]
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


def train_model( num_of_nodes_input_layer,nodes_per_layer, learning_rate, momentum, dataset, batch_sizes, epochs, have_callback , dropout, dropout_in, dropout_h):
    """
    Builds, trains and evaluates a model on a dataset based on the parameters given.

    Parameters:
    num_of_nodes_input_layer (int): the number of nodes in the input layer of the model.
    nodes_per_layer (int or list): Builds the hidden layers of the model, 1 for int, # values in the list. 
    learning_rate (float): Specifies learning rate of the optimizer.
    momentum (float): Specifies the momentum of the optimizer.
    batch_sizes (int): The size of the batches
    epochs (int): The number of epochs
    have_callback (boolean): Defines if the model is trained using early stopping or not
    dropout (boolean): Defines if the dropout teqnique will apply.
    dropout_in (float): The dropout rate for the input layer.
    dropout_h (float): The dropout rate for the hidden layers.
    
    Returns:
    tuple: A list of the mean training losses over the folds, for every epoch and the mean testing losses over the folds for every epoch. 
    
    """
    test_losses = []
    training_losses = []
    
    callback = cb.EarlyStopping(monitor = 'loss', patience= 10)
        
    for data in dataset:
        x_training = data["X_train"]
        y_training = data["y_train"]
        x_testing = data["X_test"]
        y_testing = data["y_test"]
        indx = data["fold_index"]
        print(f"Fold {indx}:") 
        
        if dropout:
            model = set_model_with_dropout(num_of_nodes_input_layer,nodes_per_layer, learning_rate, momentum, dropout_in, dropout_h)
        else:
            model = set_model(num_of_nodes_input_layer,nodes_per_layer,learning_rate, momentum)
        if have_callback:
            history = fit_model_with_stopping(model, x_training, y_training, x_testing, y_testing, batch=batch_sizes, epoch=epochs, callback = callback)
        else:
            history = fit_model(model, x_training, y_training, x_testing, y_testing, batch=batch_sizes, epoch=epochs)
        
        training_losses.append(history.history['loss'])
        test_losses.append(history.history['val_loss'])
        
    if have_callback:
        return training_losses, test_losses 
    else:
        # Mean value of loss for every epoch
        mean_training_losses = np.mean(training_losses, axis=0) 
        mean_testing_losses = np.mean(test_losses, axis=0)
        return mean_training_losses, mean_testing_losses 

def mean_of_all(data):
    means = []  
    for row in data:
        # Calculate the mean by summing the row elements and dividing by the number of elements
        mean = sum(row) / len(row) if len(row) > 0 else None  # Handle empty rows
        means.append(mean)  
    overall_mean = sum(means) / len(means)
    return overall_mean

def print_info_df(df):
    """
    Prints various useful information about a dataframe.

    Parameters:
    df (pandas Dataframe): The dataframe.
    
    """
    print(f"Original shape of Dataset: {df.shape}\n")
    print(f"\nNumber of NULL values per column:{df.isnull().sum()}\n")
    print(f"\nNumber of unique values per column:{df.nunique()}\n")
 
def vectorize_inscriptions(df, stopwords, max_features):
    """
    Uses the Tf-Idf Vectorizer, in order to convert a particular column of a dataframe to vectors, using the BoW model.

    Parameters:
    df (pandas Dataframe): The Dataframe. 
    stopwords (list): A list conatinig ancient greek stopwords.
    max_features (int): The max number of features to be selected by the vectorizer, based on their idf values.

    Returns:
    A numpy array containing the vectorized inscriptions. 
    
    """
    vectorizer = TfidfVectorizer(stop_words=stopwords, max_features=max_features)
    
    if 'text' in df.columns:
        index_matrix = vectorizer.fit_transform(df['text'].to_list())
        print(f"The shape of the matrix is: {index_matrix.shape} \n")
        # print(f"The vocabulary created from the vectorizer is the following \n {sorted(vectorizer.vocabulary_)}")
        return index_matrix.toarray()    
    else: print("The dataframe provided does not have the specific column needed.")

def split_to_folds(num_of_folds, X, y):
    """
    Splits the dataset into k different folds of training and testing data.

    Parameters:
    num_of_folds (int): The number of different folds. 
    X ():
    y ():
    

    Returns:
    A a list of dictionaries, each one being one fold. 
    
    """
    fold_dataset = []

    kf = KFold(n_splits= num_of_folds, shuffle=True, random_state=42)

    for fold_index, (train_index, test_index) in enumerate(kf.split(X), 1):
    
        # split data to train/test sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        # Store the training and test datasets along with fold index
        fold_data = {
            "fold_index": fold_index,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
        fold_dataset.append(fold_data)
    
    return fold_dataset