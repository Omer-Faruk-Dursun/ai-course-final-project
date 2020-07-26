# -*- coding: utf-8 -*-
"""
@author: Omer Faruk Dursun
"""
from textblob import TextBlob
from textblob import Word
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice

class MainProgram:
    # Utilit
    # Utility Function for calculating accuracy
    def accuracy(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
    
    # Text pre-process steps 
    def text_pre_process(self, df):
        df['summary'] = df['summary'].apply(lambda x: " ".join(x.lower() for x in x.split()))
        df['summary'] = df['summary'].str.replace('[^\w\s]','')
        stop = set(stopwords.words('english'))
        df['summary'] = df['summary'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
        df['summary'] = df['summary'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        general_words = ['to','synopsis','look','must','make','however','day','yet','dont','want','come','know','back']
        most_frequent = pd.Series(' '.join(df['summary']).split()).value_counts()[0:20]
        least_frequent = pd.Series(' '.join(df['summary']).split()).value_counts()[300:-1]
        df['summary'] = df['summary'].apply(lambda x: " ".join(x for x in x.split() if x not in general_words))
        df['summary'] = df['summary'].apply(lambda x: " ".join(x for x in x.split() if x not in most_frequent))
        df['summary'] = df['summary'].apply(lambda x: " ".join(x for x in x.split() if x not in least_frequent))
        return df
    
    
    # Function that prepares know-word-dictionary
    def dictionary(self, df):
        word_dictionary = {}
        for row in df['summary']:
            blob = TextBlob(row)
            for word in blob.words:
                if word in word_dictionary:
                    word_dictionary[word] = word_dictionary[word] + 1
                else:
                    word_dictionary[word] = 1
        return word_dictionary    
    
    
    # Function that vectorizes raw text inputs based on know-word-dictionary
    def vectorize(self, df, word_dictionary):                
        X = []
        y = []
        vector = []            
        lenght = df.shape[0]
        
        for i in range(0,lenght):
            blob = TextBlob(df.iloc[i][2])
            for key in word_dictionary:
                if key in blob.words:
                    vector.append(1)
                else:
                    vector.append(0)
            y.append(df.iloc[i][1])
            X.append(vector)
            vector = []   
        return np.array(X), np.array(y)
    
    
    # Utility Function that reads the csv file and stores it on a pandas dataframe
    def read_csv(self, path):
        df = pd.read_csv(path)
        return df
    
    
    # Utility function that splits data into test, train parts
    def test_train_split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
        return X_train, X_test, y_train, y_test
    

    # Function that plots the most popular 20 words from selected genre 
    def plot_data(self,df, selected_genre):
        df = self.text_pre_process(df)
        groups = df.groupby(['genre'])
        for i in groups:
            if selected_genre == i[0]:
                plotted_genre = selected_genre
                genre_dic = self.dictionary(i[1])
                a = {k: v for k, v in sorted(genre_dic.items(), key=lambda item: item[1], reverse = True)}
                n_items = self.take(20, a.items())
                self.bar_chart(n_items, plotted_genre)
    
    
    # Utility function that takes the first n items of a dictionary
    def take(self, n, iterable):
        "Return first n items of the iterable as a list"
        return list(islice(iterable, n))
    
    
    # Function that returns the most popular 20 words from selected genre
    def popular_words(self, df, selected_genre):
        df = self.text_pre_process(df)
        groups = df.groupby(['genre'])
        for i in groups:
            if selected_genre == i[0]:
                genre_dic = self.dictionary(i[1])
                a = {k: v for k, v in sorted(genre_dic.items(), key=lambda item: item[1], reverse = True)}
                n_items = self.take(20, a.items())
        x = []
        y = []
        for i in range(0,len(n_items)):
            x.append(n_items[i][0]) 
            y.append(n_items[i][1])
        return x, y        


    # Utility function that draws the bar chart for given values 
    def bar_chart(self, n_items, plotted_genre):
        x = []
        y = []
        for i in range(0,len(n_items)):
            x.append(n_items[i][0]) 
            y.append(n_items[i][1])
        
        x_pos = [i for i, _ in enumerate(x)]
        plt.bar(x_pos, y, color='green',width = 0.8)
        plt.xlabel("Popular Words")
        plt.ylabel("Count")
        title = "Popular Words in "+ str(plotted_genre) +' Genre'
        plt.title(title)
        plt.xticks(x_pos, x)
        plt.xticks(rotation=90)
        plt.show()
    
    # Function that vectorizes custom (user given summary)
    def custom_input(self, text, dic):
        vector = []
        blob = TextBlob(text)
        for key in dic:
            if key in blob.words:
                vector.append(1)
            else:
                vector.append(0)
        return vector
    