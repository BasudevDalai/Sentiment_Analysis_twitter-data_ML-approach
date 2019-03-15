# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:28:48 2019

@author: dalai
"""



# Prerocessing the Data



# Step 1: Importing the libraries
import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import nltk
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Step 2: Importing the datasets
train_data  = pd.read_csv('train_E6oV3lV.csv')
test_data = pd.read_csv('test_tweets_anuFYb8.csv')

# Step 3: Removing all the at_the_rates from the datasets
def remove_attherate(txt, pattern):
    r = re.findall(pattern, txt)
    for i in r:
        txt = re.sub(i, '', txt)
    return txt

train_data['clean_tweet'] = np.vectorize(remove_attherate)(train_data['tweet'], "@[\w]*")
test_data['clean_tweet'] = np.vectorize(remove_attherate)(test_data['tweet'], "@[\w]*")

# Step 4: Removing special entities using replace function
train_data['clean_tweet'] = train_data['clean_tweet'].str.replace("[^a-zA-Z#]", " ") # [^a-zA-z#] means everything not a-z or A-Z or #
test_data['clean_tweet'] = test_data['clean_tweet'].str.replace("[^a-zA-Z#]", " ")

## Not necessary step, just making a copy of the original data to keep a copy
clean_train_data = train_data
clean_test_data = test_data

# Step 5: Removing all the words having less than 3 letters in it
clean_train_data['clean_tweet'] = clean_train_data['clean_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3])) # here variable w iterates through all the words in the tweet column of dataset and accepts only the one with more than 3 letters
clean_test_data['clean_tweet'] = clean_test_data['clean_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

# Step 6: Lower casing all the letters
clean_train_data['clean_tweet'] = clean_train_data['clean_tweet'].str.lower()
clean_test_data['clean_tweet'] = clean_test_data['clean_tweet'].str.lower()

# Step 7: Tokenization (here breaking the string into words)
tokenized_tweet_train = clean_train_data['clean_tweet'].apply(lambda x: x.split())
tokenized_tweet_test = clean_test_data['clean_tweet'].apply(lambda x: x.split())

# Step 8: Lemmatizing the tweet (that is converting each word to its root word)
import textblob
from textblob import Word
clean_train_data['clean_tweet'] = clean_train_data['clean_tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# Step 9: Reversing tokenization
for i in range(len(tokenized_tweet_train)):
    tokenized_tweet_train[i] = ' '.join(tokenized_tweet_train[i])
for i in range(len(tokenized_tweet_test)):
    tokenized_tweet_test[i] = ' '.join(tokenized_tweet_test[i])

# Step 10: Adding tokenized tweet to the dataset under the name clean_tweet 
clean_train_data['clean_tweet'] = tokenized_tweet_train
clean_test_data['clean_tweet'] = tokenized_tweet_test



# Data Visualiztion



# Step 11: Getting better understanding of data using data visualization techniques
def hash_extract(x):
    hashtags = []
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags

hash_positive = hash_extract(clean_train_data['clean_tweet'][clean_train_data['label'] == 0])
hash_negative = hash_extract(clean_train_data['clean_tweet'][clean_train_data['label'] == 1])

hash_positive = sum(hash_positive,[])
hash_negative = sum(hash_negative,[])

a = nltk.FreqDist(hash_positive)
b = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
b = b.nlargest(columns = "Count", n = 10) 
plt.figure(figsize=(16,5))
axes = sns.barplot(data = b, x = "Hashtag", y = "Count")
axes.set(ylabel = 'Count')
plt.show()

c = nltk.FreqDist(hash_negative)
d = pd.DataFrame({'Hashtag': list(c.keys()),
                  'Count': list(c.values())})
d = d.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
axis = sns.barplot(data=d, x= "Hashtag", y = "Count")
axis.set(ylabel = 'Count')
plt.show()



# Training of Dataset



# Step 12: Text processing using TF-IDF features (alternatively bag of words model can be also used)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf_train = tfidf_vectorizer.fit_transform(clean_train_data['clean_tweet'])
tfidf_test = tfidf_vectorizer.fit_transform(clean_test_data['clean_tweet'])

# Step 13: Spltting the Training_data to train and validation set
from sklearn.model_selection import train_test_split
X_tfidf_train, X_tfidf_validn, y_tfidf_train, y_tfidf_validn = train_test_split(tfidf_train, clean_train_data['label'], random_state = 42, test_size=0.30) # 30% of training data is converted to validation data

# Step 14: Training the model using SVM
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_tfidf_train,y_tfidf_train)

# Step 15: Predicting the value and confirming it using validation set
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_tfidf_train,y_tfidf_train)

grid_predictions = grid.predict(X_tfidf_validn)

# Checking the performance using Classification report and Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_tfidf_validn,grid_predictions))
print(classification_report(y_tfidf_validn,grid_predictions))

