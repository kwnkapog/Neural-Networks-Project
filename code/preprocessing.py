import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import utils as ul


# Load the dataset into a pandas dataframe.
df = pd.read_csv('iphi2802.csv', delimiter='\t')

ul.print_info_df(df)

# Create a new column 'mean_date' in the dataframe, which is the mean of the two dates  
df['mean_date'] = df[['date_min', 'date_max']].mean(axis=1)


# Iitializing the tf-idf vectorizer, using a stopword list from the nltk library and and transform the 'text' column into a TF-IDF matrix of 1000 columns
stopwords = nltk.corpus.stopwords.words('greek') 
vectorizer = TfidfVectorizer(stop_words=stopwords, max_features=8000)
index_matrix = vectorizer.fit_transform(df['text'].to_list())

# Visualize the output of the vectorizer(words and their idf values)
shape = index_matrix.shape
idf_values = vectorizer.idf_
vocab = sorted(vectorizer.vocabulary_)

# Convert the input/target martices for normalization
texts = index_matrix.toarray()
dates = df['mean_date'].values.reshape(-1,1)

# Initialize a MinMaxScaler and scale both the TF-IDF matrix (input) and the mean_dates (output) column
scaler = MinMaxScaler()
X = scaler.fit_transform(texts)
y = scaler.fit_transform(dates)

# Initialize 5-Fold Cross Validation, create dictionary of each fold, store all dictionaries to fold_dataset list
fold_dataset = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

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
