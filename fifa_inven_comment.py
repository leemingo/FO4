#!/usr/bin/env python
# coding: utf-8

# In[1]:


# module set
import numpy as np
import pandas as pd

import requests
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import lxml

from collections import OrderedDict
from itertools import repeat

import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


#피파인벤 연결
driver = webdriver.Chrome('./chromedriver')


# In[3]:


positions = ['st', 'mid', 'def', 'gl']
player_url = 'http://fifaonline4.inven.co.kr/dataninfo/player/?code='


# In[4]:


def position_db(pos):
    # 선수 DB 페이지로 이동

    driver.get('http://fifaonline4.inven.co.kr/dataninfo/player/')
    driver.implicitly_wait(5)
    # 포지션 체크// checkbox의 xpath가 아닌 span 태그의 xpath를 긁어와야 함.
    if pos == 'st':
        checkbox = driver.find_element_by_xpath('//*[@id="fifaonline4Db"]/div[2]/ul[1]/li[3]/div[2]/div[1]/div[2]/label[1]/span')
        checkbox.click()
    if pos == 'mid':
        checkbox = driver.find_element_by_xpath('//*[@id="fifaonline4Db"]/div[2]/ul[1]/li[3]/div[2]/div[2]/div[2]/label[1]/span')
        checkbox.click()
    if pos == 'def':
        checkbox = driver.find_element_by_xpath('//*[@id="fifaonline4Db"]/div[2]/ul[1]/li[3]/div[2]/div[3]/div[2]/label[1]/span')
        checkbox.click()
    if pos =='gl':
        checkbox = driver.find_element_by_xpath('//*[@id="fifaonline4Db"]/div[2]/ul[1]/li[3]/div[2]/div[4]/div[2]/label[1]/span')
        checkbox.click()
    search = driver.find_element_by_xpath('//*[@id="fifaonline4Db"]/div[2]/ul[2]/li[17]/button[1]/img')
    search.click()


# In[5]:


def players_info(html):
    soup = bs(html, 'html.parser')
    search_url_list = []
    
    for link in soup.find_all('a'):
        if 'href' in link.attrs:
            li = link.attrs['href'].split('\n')
            if li[0].startswith(player_url):
                search_url_list.append(li[0])
            else:
                continue
    names = [name.text for name in soup.select('span > b')]
    search_url_list = list(OrderedDict(zip(search_url_list, repeat(None))))
    print("num of url : ", len(search_url_list))
    print(search_url_list[:5])
#     print(search_url_list)
    return search_url_list


# In[11]:


position_db('st')


# In[24]:


urls = players_info(driver.page_source)
print("len st players : ", len(urls))
print(urls[5])


# In[25]:


name = []
for search_url in urls: # 아까 검색해야할 url 리스트에서 하나씩 가져온다
    one_player_page = requests.get(search_url)
    one_soup=bs(one_player_page.content,'lxml')
        # one_player_db 는 각 선수마다 데이터베이스. 루프 한번 돌 때마다 각 선수의 DB를 일단 긁어옴
    one_player_db=one_soup.find_all("div", {"class":"fifa4 db_tooltip"})
    
    for db in one_player_db:
            for i,p in enumerate(db.find_all("p")):
                if i==0:
                    name.append(p.text)
name


# In[39]:


total_df = pd.DataFrame()
for i in range(len(urls)):
    driver.get(urls[i])
    driver.implicitly_wait(2)

    html = driver.page_source
    one_soup = bs(html, 'html.parser')
    one_player_db=one_soup.find_all('tr', class_=['item', 'reply'])
    

    cmt_list = []
    for cmt in one_player_db:
        a = cmt.find_all('span')
        cmt_list.append(a[3].text)
    one_df = pd.DataFrame({'name': name[i], 'comment': cmt_list})
    total_df = pd.concat([total_df, one_df], axis = 0)
# total_df = total_df.T
total_df


# In[80]:


total_df = pd.DataFrame()
for i in range(2):
    driver.get(urls[i])
    driver.implicitly_wait(2)

    html = driver.page_source
    one_soup = bs(html, 'html.parser')
    one_player_db=one_soup.find_all('tr', class_=['item', 'reply'])
    

    cmt_list = []
    for cmt in one_player_db:
        a = cmt.find_all('span')
        cmt_list.append(a[3].text)
    one_df = pd.DataFrame({'name': name[i], 'comment': cmt_list})
    total_df = pd.concat([total_df, one_df], axis = 0)
# total_df = total_df.T
total_df


# In[81]:


total_df.reset_index(drop = True)


# 위에는 axis = 1 로 해줘서 잘못 나왔고, 밑에 것처럼 나올 것!

# In[45]:


from konlpy.tag import Okt


# In[48]:


okt = Okt()
print(okt.pos(u'이 밤 그날의 반딧불을 당신의 창 가까이 보낼게요'))


# In[91]:


okt.pos(total_df.loc[2, 'comment'])


# In[93]:


okt.nouns(total_df.loc[0, 'comment'])


# In[98]:


nouns = []
for i in range(len(total_df)):
    nouns += okt.nouns(total_df.loc[i, 'comment'])
    
nouns    

