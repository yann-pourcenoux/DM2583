#!/usr/bin/env python
# coding: utf-8

# # Project Notebook

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import nltk
import tensorflow as tf
import gzip
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import gc
from time import time
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# In[2]:


# Machine learning librairies and tools
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from tensorflow.keras.layers import Bidirectional, LSTM, Embedding, SpatialDropout1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall, AUC

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def FCN(x_train, y_train):
    model = Sequential()
    model.add(InputLayer(input_shape=(x_train.shape[1])))
    model.add(Dense(units=32,
                    activation='elu',
                    kernel_initializer='lecun_normal'))
    model.add(Dropout(rate=0.4))

    model.add(Dense(units=16,
                    activation='elu',
                    kernel_initializer='lecun_normal'))
    model.add(Dropout(rate=0.4))

    model.add(Dense(units=y_train.shape[1],
                    activation='softmax',
                    kernel_initializer='lecun_normal'))

    METRICS = [CategoricalAccuracy(name='accuracy'),
               Precision(name='precision'),
               Recall(name='recall'),
               AUC(name='auc')]

    optimizer = Adam(lr=1e-5)
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=METRICS)
    model.summary()

    earlystop = EarlyStopping(monitor='val_loss',
                             patience=15,
                             restore_best_weights=True)
    reduceLR = ReduceLROnPlateau(monitor='val_loss',
                                factor=np.sqrt(1e-1),
                                verbose=1,
                                patience=5)
    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=32,
                        epochs=10000,
                        verbose=0,
                        callbacks=[earlystop, reduceLR],
                        validation_split=0.3,
                        shuffle=True,
                        workers=4)

    return model

def RNN(x_train, y_train):
    model = Sequential()
    model.add(Embedding(max_features, 256, embeddings_initializer='lecun_normal'))
    model.add(SpatialDropout1D(rate=0.4))
    model.add(Bidirectional(LSTM(units=64, dropout=0.4, recurrent_dropout=0.4)))

    model.add(Dense(units=y_train.shape[1],
                    activation='softmax',
                    kernel_initializer='lecun_normal'))

    METRICS = [CategoricalAccuracy(name='accuracy'),
               Precision(name='precision'),
               Recall(name='recall'),
               AUC(name='auc')]

    optimizer = Adam(lr=1e-5)
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=METRICS)
    model.summary()

    earlystop = EarlyStopping(monitor='val_loss',
                             patience=15,
                             restore_best_weights=True)
    reduceLR = ReduceLROnPlateau(monitor='val_loss',
                                factor=np.sqrt(1e-1),
                                verbose=1,
                                patience=5)
    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=32,
                        epochs=10000,
                        verbose=0,
                        callbacks=[earlystop, reduceLR],
                        validation_split=0.3,
                        shuffle=True,
                        workers=4)

    return model

def evaluate(dataframe, classifier, preprocessing_x, preprocessing_y=None, NN=False):
    size = len(dataframe)
    scores = []
    n = size//80000
    for k in range(n):
        gc.collect()
        y_true = dataframe.classification.values[k*80000:(k+1)*80000]
        if preprocessing_y:
            y_true = preprocessing_y(y_true)

        if NN:
            scores.append(model.evaluate(preprocessing_x(dataframe.text.values[k*80000:(k+1)*80000]), y_true, verbose=0)[1:])

        else:
            y_pred = classifier.predict(preprocessing_x(dataframe.text.values[k*80000:(k+1)*80000]))
            scores.append([accuracy_score(y_true, y_pred),
                           precision_score(y_true, y_pred, average='weighted'),
                           recall_score(y_true, y_pred, average='weighted', zero_division=0)])



    # last values
    y_true = dataframe.classification.values[(k+1)*80000:]
    if preprocessing_y:
        y_true = preprocessing_y(y_true)

    if NN:
        scores.append(model.evaluate(preprocessing_x(dataframe.text.values[(k+1)*80000:]), y_true, verbose=0)[1:])

    else:
        y_pred = classifier.predict(preprocessing_x(dataframe.text.values[(k+1)*80000:]))
        scores.append([accuracy_score(y_true, y_pred),
                       precision_score(y_true, y_pred, average='weighted'),
                       recall_score(y_true, y_pred, average='weighted', zero_division=0)])

    return np.mean(np.asarray(scores), axis=0)

def get_scores(y_true, y_pred):
    return [accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, average='weighted'),recall_score(y_true, y_pred, average='weighted')]

def one_hot(x):
    classes = np.asarray([-1, 0, 1])
    array = np.zeros((*x.shape, classes.shape[0]), dtype=np.int)
    for i, classe in enumerate(classes):
        vector = np.zeros((1,classes.shape[0]), dtype=np.int)
        vector[:,i]=1
        array[x==classe] = vector
    return array


############## MAIN#######################

df_airlines = pd.read_csv('data/airlines_cleaned.csv').dropna()
df_sentiment = pd.read_csv('data/sentiment_cleaned.csv').dropna()
df_amazon = pd.read_csv('data/amazon_movies.csv').dropna()

vectorizer = TfidfVectorizer(min_df=10,
                             norm='l2',
                             ngram_range=(1,1))
vectorizer_bow = TfidfVectorizer(min_df=10,
                                 norm='l2',
                                 ngram_range=(1,3))

max_features = 2048 # around number of unigrams in the data
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df_airlines.text.values)

x_train = vectorizer.fit_transform(df_airlines.text.values).toarray()
x_train_bow = vectorizer_bow.fit_transform(df_airlines.text.values).toarray()
x_train_seq = pad_sequences(tokenizer.texts_to_sequences(df_airlines.text.values))
y_train = df_airlines.classification.values
y_train_oh = one_hot(y_train)


amazon_nb = []
amazon_svm = []
amazon_adaboost = []
amazon_ann = []
amazon_rnn = []

sentiment_nb = []
sentiment_svm = []
sentiment_adaboost = []
sentiment_ann = []
sentiment_rnn = []

airlines_nb = []
airlines_svm = []
airlines_adaboost = []
airlines_ann = []
airlines_rnn = []

kf = KFold(n_splits=4, shuffle=True)
start=time()
for k in range(3): #needs to average the results
    for train_index, test_index in tqdm(kf.split(x_train), total=4):
        indexes_amazon = np.random.choice(a=len(df_amazon), size=len(df_amazon), replace=False)
        indexes_sentiment = np.random.choice(a=len(df_sentiment), size=len(df_sentiment), replace=False)

        print('RNN')
        # RNN
        model = RNN(x_train_seq[train_index], y_train_oh[train_index])

        print('evaluation sentiment')
        sentiment_rnn.append(evaluate(df_sentiment.iloc[indexes_sentiment], model, lambda x:pad_sequences(tokenizer.texts_to_sequences(x), maxlen=21), one_hot, NN=True))
        print('evaluation amazon')
        amazon_rnn.append(evaluate(df_amazon.iloc[indexes_amazon], model, lambda x:pad_sequences(tokenizer.texts_to_sequences(x), maxlen=21), one_hot, NN=True))
        print('evaluation airlines')
        airlines_rnn.append(model.evaluate(x_train_seq[test_index], y_train_oh[test_index])[1:])

        print('ANN')
        # ANN
        model = FCN(x_train[train_index], y_train_oh[train_index])

        print('evaluation sentiment')
        sentiment_ann.append(evaluate(df_sentiment.iloc[indexes_sentiment], model, lambda x:vectorizer.transform(x).toarray(), one_hot, NN=True))
        print('evaluation amazon')
        amazon_ann.append(evaluate(df_amazon.iloc[indexes_amazon], model, lambda x:vectorizer.transform(x).toarray(), one_hot, NN=True))
        print('evaluation airlines')
        airlines_ann.append(model.evaluate(x_train[test_index], y_train_oh[test_index])[1:])

        print('Naive Bayes')
        # Naive Bayes
        classifier = MultinomialNB()
        classifier.fit(x_train[train_index], y_train[train_index])

        print('evaluation')
        amazon_nb.append(evaluate(df_amazon.iloc[indexes_amazon], classifier, lambda x:vectorizer.transform(x).toarray()))
        sentiment_nb.append(evaluate(df_sentiment.iloc[indexes_sentiment], classifier, lambda x:vectorizer.transform(x).toarray()))
        airlines_nb.append(get_scores(y_train[test_index], classifier.predict(x_train[test_index])))

#         print('SVM')
#         # SVM
#         classifier = SVC(kernel='rbf')
#         classifier.fit(x_train[train_index], y_train[train_index])

#         print('evaluation sentiment')
#         sentiment_svm.append(evaluate(df_sentiment.iloc[indexes_sentiment], classifier, lambda x:vectorizer.transform(x).toarray()))
#         print('evaluation amazon')
#         amazon_svm.append(evaluate(df_amazon.iloc[indexes_amazon], classifier, lambda x:vectorizer.transform(x).toarray()))
#         print('evaluation airlines')
#         airlines_svm.append(get_scores(y_train[test_index], classifier.predict(x_train[test_index])))

        print('AdaBoost')
        # AdaBoost
        classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1.)
        classifier.fit(x_train[train_index], y_train[train_index])

        print('evaluation sentiment')
        sentiment_adaboost.append(evaluate(df_sentiment.iloc[indexes_sentiment], classifier, lambda x:vectorizer.transform(x).toarray()))
        print('evaluation amazon')
        amazon_adaboost.append(evaluate(df_amazon.iloc[indexes_amazon], classifier, lambda x:vectorizer.transform(x).toarray()))
        print('evaluation airlines')
        airlines_adaboost.append(get_scores(y_train[test_index], classifier.predict(x_train[test_index])))


print(time()-start)

print(amazon_nb)
print(amazon_adaboost)
print(amazon_ann)
print(amazon_rnn)

print(sentiment_nb)
print(sentiment_adaboost)
print(sentiment_ann)
print(sentiment_rnn)

print(airlines_nb)
print(airlines_adaboost)
print(airlines_ann)
print(airlines_rnn)

amazon = pd.DataFrame({'NB':amazon_nb, 'AdaBoost':amazon_adaboost, 'ANN':amazon_ann, 'RNN':amazon_rnn})
amazon.to_csv('amazon_scores.csv')

sentiment = pd.DataFrame({'NB':sentiment_nb, 'AdaBoost':sentiment_adaboost, 'ANN':sentiment_ann, 'RNN':sentiment_rnn})
sentiment.to_csv('sentiment_scores.csv')

airlines = pd.DataFrame({'NB':airlines_nb, 'AdaBoost':airlines_adaboost, 'ANN':airlines_ann, 'RNN':airlines_rnn})
airlines.to_csv('airlines_scores.csv')


# In[ ]:


x_train = x_train_bow

amazon_nb = []
amazon_svm = []
amazon_adaboost = []
amazon_ann = []
amazon_rnn = []

sentiment_nb = []
sentiment_svm = []
sentiment_adaboost = []
sentiment_ann = []
sentiment_rnn = []

airlines_nb = []
airlines_svm = []
airlines_adaboost = []
airlines_ann = []
airlines_rnn = []

kf = KFold(n_splits=4, shuffle=True)
for k in range(5): #needs to average the results
    for train_index, test_index in tqdm(kf.split(x_train), total=4):
        indexes_amazon = np.random.choice(a=len(df_amazon), size=len(df_amazon), replace=False)
        indexes_sentiment = np.random.choice(a=len(df_sentiment), size=len(df_sentiment), replace=False)

        # Naive Bayes
        classifier = MultinomialNB()
        classifier.fit(x_train[train_index], y_train[train_index])

        sentiment_nb.append(evaluate(df_sentiment[indexes_sentiment], classifier, vectorizer_bow.transform))
        amazon_nb.append(evaluate(df_amazon[indexes_amazon], classifier, vectorizer_bow.transform))
        airlines_nb.append(get_scores(y_train[test_index], classifier.predict(x_train[test_index])))
        # SVM
        classifier = SVC(kernel='rbf')
        classifier.fit(x_train[train_index], y_train[train_index])

        sentiment_svm.append(evaluate(df_sentiment[indexes_sentiment], classifier, vectorizer_bow.transform))
        amazon_svm.append(evaluate(df_amazon[indexes_amazon], classifier, vectorizer_bow.transform))
        airlines_svm.append(get_scores(y_train[test_index], classifier.predict(x_train[test_index])))
        # AdaBoost
        classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1.)
        classifier.fit(x_train[train_index], y_train[train_index])

        sentiment_adaboost.append(evaluate(df_sentiment[indexes_sentiment], classifier, vectorizer_bow.transform))
        amazon_adaboost.append(evaluate(df_amazon[indexes_amazon], classifier, vectorizer_bow.transform))
        airlines_adaboost.append(get_scores(y_train[test_index], classifier.predict(x_train[test_index])))

        # ANN
        model = FCN(x_train[train_index], y_train_oh[train_index])
        sentiment_ann.append(evaluate(df_sentiment, model, vectorizer_bow.transform, one_hot))
        amazon_ann.append(evaluate(df_amazon, model, vectorizer_bow.transform, one_hot))
        airlines_ann.append(model.evaluate(x_train[test_index], y_train_oh[test_index])[1:])

amazon = pd.DataFrame({'NB':amazon_nb, 'SVM':amazon_svm, 'AdaBoost':amazon_adaboost, 'ANN':amazon_ann, 'RNN':amazon_rnn})
amazon.to_csv('amazon_scores_bow.csv')

sentiment = pd.DataFrame({'NB':sentiment_nb, 'SVM':sentiment_svm, 'AdaBoost':sentiment_adaboost, 'ANN':sentiment_ann, 'RNN':sentiment_rnn})
sentiment.to_csv('sentiment_scores_bow.csv')

airlines = pd.DataFrame({'NB':airlines_nb, 'SVM':airlines_svm, 'AdaBoost':airlines_adaboost, 'ANN':airlines_ann, 'RNN':airlines_rnn})
airlines.to_csv('airlines_scores_bow.csv')
