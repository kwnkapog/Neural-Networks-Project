import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold


# load the dataset into a pandas dataframe, using tab as a delimiter.
df = pd.read_csv('iphi2802.csv', delimiter='\t')
df1 = df

# check for any missing or undefined values from the first row
null_values = df.iloc[:,0].isnull()
print(df[null_values])

missing_values = df.iloc[:,0].isna()
print(df[missing_values])

df.describe()

#create a list of the different inscriptions and import list of stopwords from nltk
inscriptions = df['text'].to_list()
stopwords = nltk.corpus.stopwords.words('greek')

# build the tf-idf vectorizer, using a stopword list from the nltk library and 1000 words  
vectorizer = TfidfVectorizer(stop_words=stopwords, max_features=1000)
index_matrix = vectorizer.fit_transform(inscriptions)

#visualize the output of the vectorizer(words and their idf values)
shape = index_matrix.shape
idf_values = vectorizer.idf_
vocab = sorted(vectorizer.vocabulary_)

# transform the input/output martices for normalization
texts = index_matrix.toarray()
dates = df1[['date_min', 'date_max']].values

# put the transformed text into the dataframe
df1['text'] = [row for row in texts]

# normalize the input and output sets:
#   text(tf-idf values)->[0,1]
#   date_min,date_max ->[0,1] 
input_scaler = MinMaxScaler()
X = input_scaler.fit_transform(texts)

target_scaler = MinMaxScaler()
y = target_scaler.fit_transform(dates)


# initialize 5-Fold Cross Validation, store to a list all the folds.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_dataset = []
# take fold, train and test indexes starting the fold index from 1
for fold_index, (train_index, test_index) in enumerate(kf.split(X), 1):
    
    # split data to train/test sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #calculate mean values for both target sets
    y_train_mean = np.mean(y_train, axis=1)
    y_test_mean = np.mean(y_test, axis=1)
    
    # Store the training and test datasets along with fold index
    fold_data = {
        "fold_index": fold_index,
        "X_train": X_train,
        "y_train": y_train,
        "y_train_mean": y_train_mean,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_mean": y_test_mean
    }
    fold_dataset.append(fold_data)


