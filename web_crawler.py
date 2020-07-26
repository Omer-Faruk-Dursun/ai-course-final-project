# -*- coding: utf-8 -*-
"""
@author: Omer Faruk Dursun
"""

from bs4 import BeautifulSoup
import requests
import pandas as pd


BASE_URL = 'https://www.imdb.com'

categories = ['https://www.imdb.com/list/ls009668711/', 'https://www.imdb.com/list/ls050784999/', 'https://www.imdb.com/search/title/?genres=crime&title_type=feature&explore=genres',
              'https://www.imdb.com/search/title/?title_type=feature&genres=romance', 'https://www.imdb.com/list/ls057433882/', 'https://www.imdb.com/search/title/?genres=action&title_type=feature&explore=genres',
              'https://www.imdb.com/search/title/?&genres=horror&explore=title_type,genres', 'https://www.imdb.com/search/title/?title_type=feature&genres=sci-fi&sort=user_rating,desc&explore=genres'
              'https://www.imdb.com/list/ls072723203/?sort=user_rating,desc&st_dt=&mode=detail&page=1','https://www.imdb.com/list/ls072723203/?sort=user_rating,desc&st_dt=&mode=detail&page=2',
              'https://www.imdb.com/list/ls072723203/?sort=user_rating,desc&st_dt=&mode=detail&page=3', 'https://www.imdb.com/search/title/?groups=top_250&sort=user_rating,desc&start=51&ref_=adv_nxt',
              'https://www.imdb.com/search/title/?groups=top_250&sort=user_rating,desc&start=101&ref_=adv_nxt','https://www.imdb.com/search/title/?groups=top_250&sort=user_rating,desc&start=151&ref_=adv_nxt',
              'https://www.imdb.com/search/title/?groups=top_250&sort=user_rating,desc&start=201&ref_=adv_nxt','https://www.imdb.com/list/ls020810230/','https://www.imdb.com/list/ls027636165/']

df = pd.DataFrame()
movie_names = []
summary = []
movie_genre = []
all_the_summaries = ''
step = 0
for category in categories:
    step += 1
    print(step)
    page_links = []
    source = requests.get(category).text
            
    with open('web_page.html', 'w', encoding='utf-8') as f:
        f.write(source)
    
    
    with open('web_page.html', 'r', encoding='utf-8') as html_file:
        soup = BeautifulSoup(html_file,'lxml')
      
    header = soup.findAll('h3', class_="lister-item-header")
    text_muted = soup.findAll('p', class_='text-muted')
    
    for element in header:
        link = element.find('a', href=True)
        movie_title = link.text
        href = link['href']
        movie_names.append(movie_title)
        page_links.append(href)
    
    for i in page_links:
        TARGET_URL = BASE_URL+i+'plotsummary'
        source = requests.get(TARGET_URL).text
        soup = BeautifulSoup(source,'lxml')
        summary_list = soup.findAll('li', class_='ipl-zebra-list__item')
        for k in summary_list:
            if k.find('p') is not None:
                all_the_summaries = all_the_summaries + k.find('p').text
        summary.append(all_the_summaries)
        all_the_summaries = ''       
        
    for element in text_muted:
        if element.find('span', class_='genre') is not None:
            genre = element.find('span', class_='genre').text
            movie_genre.append(genre)    
    
          
df['movie_titles'] = movie_names
df['genre'] = movie_genre
df['summary'] = summary

df.to_csv('movie_data_raw.csv', index =False)
            
    
    