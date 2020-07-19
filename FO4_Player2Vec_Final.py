#!/usr/bin/env python
# coding: utf-8

# In[190]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


import os
os.chdir('/content/drive/Shared drives/parrot_AAAAB/FIFA ONLINE 4/comment_data/codes')


# In[3]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Dot, Reshape, Dense
from tensorflow.keras.models import Model

import pathlib
import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# In[92]:


#데이터 불러오기
raw_data=pd.read_csv('full_player_comparison_final.csv')


# In[93]:


raw_data


# In[94]:


def to_list(x):
    return [name.strip()[1:-1] for name in x[1:-1].split(',')]


# In[95]:


raw_data['list'] = raw_data['preprocessed'].apply(lambda x: to_list(x))


# In[96]:


raw_data = raw_data.iloc[:, 1:]


# In[97]:


raw_data


# In[98]:


for i in range(len(raw_data)):
  raw_data.list[i].append(raw_data.name[i])
raw_data


# # 단어에 숫자 부여하기

# In[99]:


raw_data['string'] = [','.join(map(str, l)) for l in raw_data['list']]
raw_data


# In[100]:


my_str= raw_data['string'].str.cat(sep=', ')
words = my_str.split(",")
#빈칸지우기
words = [x.strip(' ') for x in words]


# In[101]:


len(words)


# In[102]:


#빈칸지우고 유일한것만 남기기
words= list(filter(None, words))
words= list(set(words))


# In[103]:


#단어에 숫자 부여하기
word2int = {}
for i,word in enumerate(words):
    word2int[word] = i


# # N그램 만들기

# In[104]:


def make_ngram(x,WINDOW_SIZE):
    data = []
    for sentence in x:
        for idx,word in enumerate(sentence):
            for neighbor in sentence[max(idx - WINDOW_SIZE ,0) : min( idx+ WINDOW_SIZE, len(sentence))]:
                if neighbor != word:
                    data.append([word,neighbor])
    return data


# In[107]:


data=make_ngram(raw_data['list'],6)
df=pd.DataFrame(data, columns = ['input','label'])


# In[108]:


df[:50]


# In[109]:


df['input'] = df['input'].map(word2int) 
df['label'] = df['label'].map(word2int) 


# In[110]:


df


# In[111]:


subset = df[['input', 'label']]
pairs = [tuple(x) for x in subset.to_numpy()]


# In[112]:


pairs[1000]


# In[113]:


print([k for k,v in word2int.items() if v == pairs[1][0]],
      [k for k,v in word2int.items() if v == pairs[1][1]])


# In[114]:


pairs_set = set(pairs)


# ---
# # 모델 설계하기

# In[115]:


import numpy as np
import random
random.seed(100)

def generate_batch(pairs, n_positive = 50, negative_ratio = 1.0, classification = False):
    """Generate batches of samples for training"""
    #배치사이즈
    batch_size = n_positive * (1 + negative_ratio)
    #배치사이즈 x 3 의 batch만들기
    batch = np.zeros((batch_size, 3))
    
    # Adjust label based on task
    if classification:
        neg_label = 0
    else:
        neg_label = -1
    
    # This creates a generator
    while True:
        # randomly choose positive examples 긍정라벨갯수만큼 뽑음
        for idx, (book_id, link_id) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (book_id, link_id, 1)

        # Increment idx by 1
        idx += 1
        
        # Add negative examples until reach batch size 부정라벨은 총 배치사이즈까지 뽑음
        while idx < batch_size:
            
            # random selection 임의로 뽑아서
            random_book = random.randrange(len(words))
            random_link = random.randrange(len(words))
            
            # Check to make sure this is not a positive example 페어셋에 있는지 확인하고
            if (random_book, random_link) not in pairs_set:
                
                # Add to batch and increment index 배치에 추가함
                batch[idx, :] = (random_book, random_link, neg_label)
                idx += 1
                
        # Make sure to shuffle order 랜덤 셔플합니다
        np.random.shuffle(batch)
        yield {'book': batch[:, 0], 'link': batch[:, 1]}, batch[:, 2]


# In[220]:


def book_embedding_model(embedding_size = 50, classification = False):
    """Model to embed books and wikilinks using the functional API.
       Trained to discern if a link is present in a article"""
    
    # Both inputs are 1-dimensional
    book = Input(name = 'book', shape = [1])
    link = Input(name = 'link', shape = [1])
    
    # Embedding the book (shape will be (None, 1, 50))
    book_embedding =Embedding(name = 'book_embedding',
                                           input_dim = len(words),
                                           output_dim = embedding_size)(book)
    
    # Embedding the link (shape will be (None, 1, 50))
    link_embedding =Embedding(name = 'link_embedding',
                                                    input_dim = len(words),
                                                    output_dim = embedding_size)(link)
    
    # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
    merged = Dot(name = 'dot_product', normalize = True, axes = 2)([book_embedding, link_embedding])
    
    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape = [1])(merged)
    
    # If classifcation, add extra layer and loss function is binary cross entropy
    if classification:
        merged = Dense(1, activation = 'sigmoid')(merged)
        model = Model(inputs = [book, link], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Otherwise loss function is mean squared error
    else:
        model = Model(inputs = [book, link], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'mse')
    
    return model

# Instantiate model and show parameters
model = book_embedding_model()

model.summary()


# In[221]:


n_positive =256

gen = generate_batch(pairs, n_positive, negative_ratio = 2)

# Train
h = model.fit_generator(gen, epochs = 50, 
                        steps_per_epoch = len(pairs) // n_positive,
                        verbose = 2 ,)


# In[177]:


model.save('/content/drive/Shared drives/parrot_AAAAB/FIFA ONLINE 4/comment_data/codes/embedding_space_2nd.h5')


# In[223]:


# Extract embeddings
book_layer = model.get_layer('book_embedding')
book_weights = book_layer.get_weights()[0]
#book_weights = book_weights / np.linalg.norm(book_weights, axis = 1).reshape((-1, 1))


# In[224]:


book_weights.shape


# In[231]:


dists = np.dot(book_weights, book_weights[word2int.get('폴 포그바')])
dists


# In[232]:


sorted_dists = np.argsort(dists)


# In[233]:


n=10
closest = sorted_dists[-n-1: len(dists) - 1]


# In[234]:


closest


# In[235]:


items=[]
for c in closest:
    a= [ k for k,v in word2int.items() if v == c ]
    items.append(a)


# In[236]:


#손흥민
items


# In[184]:


distances = [dists[c] for c in closest]


# In[185]:


distances


# In[ ]:


import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for key,value in word2int.items():
  vec = book_weights[value] # skip 0, it's padding.
  out_m.write(key + "\n")
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()


# In[ ]:


try:
  from google.colab import files
except ImportError:
   pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')


# In[ ]:


for key,value in word2int.items():
  vec = book_weights[value] # skip 0, it's padding.
  #print(key + "\n")
  print('\t'.join([str(x) for x in vec]) + "\n")


# In[187]:


with open('data.json', 'w') as fp:
    json.dump(word2int, fp,  indent=4)


# In[188]:


from tensorboard.plugins import projector


# In[222]:


# Set up a logs directory, so Tensorboard knows where to look for files
log_dir='/logs/fo4/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
'''
# Save Labels separately on a line-by-line manner.
with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
  for subwords in encoder.subwords:
    f.write("{}\n".format(subwords))
  # Fill in the rest of the labels with "unknown"
  for unknown in range(1, encoder.vocab_size - len(encoder.subwords)):
    f.write("unknown #{}\n".format(unknown))
'''

checkpoint = tf.train.Checkpoint(embedding=book_weights)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# Set up config
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
# The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(log_dir, config)


# In[ ]:




