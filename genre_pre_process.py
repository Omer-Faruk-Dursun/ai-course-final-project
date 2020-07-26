# -*- coding: utf-8 -*-
"""
@author: Omer Faruk Dursun
"""
import pandas as pd

df = pd.read_csv('movie_data_raw.csv')

lenght = df.shape[0]

for i in range(0,lenght):
    genre = df.iloc[i][1]
    genre = " ".join(genre.split())
    if 'Comedy' in genre:
        genre = 'Comedy'
    elif 'Romance' in genre:
        genre = 'Romance'
    elif 'Horror' in genre or 'Thriller' in genre:
        genre = 'Horror'
    elif 'Fantasy' in genre:
        genre = 'Fantasy'
    elif 'Sci-Fi' in genre:
        genre = 'Sci-Fi'  
    elif 'Crime' in genre:
        genre = 'Crime'
    elif 'History' in genre or 'War' in genre:
        genre = 'History' 
    elif 'Action' in genre or 'Adventure' in genre:
        genre = 'Action'
    elif 'Sport' in genre:
        genre = 'Sport'
    elif 'Music' in genre:
        genre = 'Music' 
    else:
        genre = 'Drama'
    df.iloc[i][1] = genre

df = df[df['genre'] != 'Sport']
df = df[df['genre'] != 'Music']
df = df[df['genre'] != 'History']

df.to_csv('movie_data_v3.csv', index = False)