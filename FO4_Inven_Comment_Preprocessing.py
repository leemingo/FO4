#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd, numpy as np
import os
from konlpy.tag import Okt
import re
from collections import Counter


# In[2]:


okt = Okt()


# In[17]:


df = pd.read_csv('/Users/leeminho/Desktop/leemingo/sogang/parrot/fifa_recommendation_system/TT_comment.csv')
del df['Unnamed: 0']
del df['Unnamed: 0.1']


# In[81]:


df = pd.read_csv('total comment.csv')
del df['Unnamed: 0']
df.head()


# In[83]:


df['tagged comment']=df['comment'].apply(lambda row: okt.nouns(row))


# In[84]:


df.head()


# In[101]:


df.to_csv('original.csv')


# In[85]:


name_list = np.unique(df['name'].values)


# In[76]:


name_list


# In[86]:


def extract_nouns(name):
    player=df[df['name']==name]['tagged comment']
    temp=[]
    for word in player:
        temp.extend(word)
    sorted_list=sorted(Counter(temp).items(), key= lambda x: x[1], reverse=True)
    sorted_list=list(map(lambda x: x[0],sorted_list))
    return sorted_list


# In[189]:


#자주 나오는 잘못된 선수 이름 제거
exception = ['티티', '크로스', '바로', '자기', '레알', '스타', '머리', '피지', '밀란', '코너', '시세', '카라', '신세계', '이용',
            '카이', '알리']
change = {'다비드' : '다비드 루이스', '앙리' : '티에리 앙리', '워커' : '카일 워커', '멘디': '페를랑 멘디', '토레스' : '페르난도 토레스',
         '로시' : 'D. 데로시', '카일': '카일 워커', '보아텡': '제롬 보아텡', '덕배' : 'K. 더브라위너'}


# In[164]:


def extract_players(sorted_list):
    temp=[]
    count = 0
    for i in sorted_list:
        for j in name_list:#unique players' name list
            if (i in j.split() or i == j or i == re.sub('\s+', '', j))and len(i)>1:
                if i in exception: continue
                elif i in change:
                    j = change[i]
                original=[i,j]
                temp.append(original)
                count += 1
                break
        if count == 5: break
                
    return temp
 


# In[165]:


def make_dict():
    co_dict={}
    for name in name_list:
        co_dict[str(name)]=[]
        temp=extract_players(extract_nouns(str(name)))
        co_dict[str(name)].extend(temp)
    return co_dict


# In[190]:


temp = make_dict()


# In[191]:


mh = pd.DataFrame.from_dict(temp.items())
mh


# In[192]:


mh.columns = ['name', 'comparison']
mh


# In[193]:


import ast

def preprocess(name,text):
    #convert sting -> list
    #text=ast.literal_eval(text)
    
    # drop_duplicates    
    temp=[]
    for i in range(len(text)):
        if name == text[i][1]:
            continue
        temp.append(text[i][1])
    temp=list(set(temp))
    return temp


# In[194]:


mh['preprocessed']=mh.apply(lambda row: preprocess( row['name'],row['comparison']), axis=1)
mh


# In[195]:


mh['length']=mh['preprocessed'].apply(lambda row: len(row))
mh


# In[196]:


mh = mh[mh['length'] != 0][['name', 'preprocessed']]
mh


# In[197]:


mh.index = np.arange(0, len(mh))


# In[186]:


mh


# In[198]:


mh.to_csv('full_player_comparison_final.csv')


# In[188]:


mh


# In[ ]:




