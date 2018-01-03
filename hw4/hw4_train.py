

from CRNN_config import *

import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from collections import defaultdict


save_dict = False
word_freq_path =  'word_freq.pk'

load_word2vec = False
word2vec_path = 'word2vec.bin'

load_model_bool = False
model_path = 'CRNN_1st.h5'
"""
nolabel = np.load("/home/derricksu/ML_data/hw4/train_nolabel.npy")
train_path = "/home/derricksu/ML_data/hw4/train.npy"
train = np.load(train_path)
test_path = "/home/derricksu/ML_data/hw4/test.npy"
test = np.load(test_path)
predict_path = "/home/derricksu/pred/best.csv"

"""
## for homework
train_path = sys.argv[1]
train=load_data(train_path,train=True)

nolabel_path = sys.argv[2]
nolabel = load_nolabel(nolabel_path)


# Record frequency of each word
word_freq=defaultdict(int)

# nolabel
sentence = []
for row in nolabel:
    x = text_to_wordlist(row).split()
    sentence.append(x)
    for s in x:
        word_freq[s]+=1

print("After nolabel: %d" % len(word_freq))

# label
X_data = train[:,1]
y_data = train[:,0].astype('int')


train_sentence = []

for i,row in enumerate(X_data):
    x = text_to_wordlist(row).split()
    train_sentence.append(x)
    for s in x:
        word_freq[s]+=1
print("After label: %d" % len(word_freq))




# produce word_freq
if save_dict:
    with open(word_freq_path,"wb") as pk:
        pickle.dump( word_freq,pk )
        pk.close()
else:
    with open(word_freq_path,"rb") as pk:
        word_freq = pickle.load(pk)
        pk.close()
print("word_freq is already")


# produce word2vec
if not load_word2vec:
    print("train word2vec...") 
    # sg defines the training algorithm. By default (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed. window=12
    my_word2vec = Word2Vec(sentence + train_sentence + test_sentence, sg = 0,#window=15,
                           iter=30, min_count=min_c,size=128,workers=16)
    # summarize the loaded model
    print(my_word2vec)
    # summarize vocabulary
    words = list(my_word2vec.wv.vocab)
    #print(words)
    # access vector for one word
    print(my_word2vec['sentence'].shape)
    # save model
    
    #my_word2vec.save(word2vec_path)
else:
    print("load model...") 
    my_word2vec = Word2Vec.load(word2vec_path)
    #print(my_word2vec)

vocab = dict([(k, v.index) for k, v in my_word2vec.wv.vocab.items()])
weight_matrix = my_word2vec.wv.syn0 #word_to_vec
print("Weight_matrix shape : " , weight_matrix.shape)
del my_word2vec
print("word2vec is already.")


# word to idx of train
train_word2idx=[]
for row in train_sentence:
    idx = []
    for word in row:
        if word_freq[word]>=min_c:
            idx.append(vocab[word])
    train_word2idx.append(idx)

print("Length of train_sentence : ",len(train_sentence))
print("Length of train_word2idx : ",len(train_word2idx))

max_len = 0
for row in train_word2idx:
    if max_len <len(row):
        max_len = len(row)
print("Max length of train sentence : ",max_len)


#import keras
from keras import utils
from keras.models import load_model


Y_train = utils.to_categorical(y_data ,2)
X_train = train_word2idx
X_test = test_word2idx


max_review_length = 100

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

n_batch = 256
n_epoch = 10

if load_model_bool:
    model = load_model(model_path)
else:
    from keras.callbacks import EarlyStopping , History
    model = build(weight_matrix,vocab,max_review_length)

    earlystopping=EarlyStopping(monitor='val_categorical_accuracy', patience=3, verbose=0, mode='auto')
    history = History()

    hist_lstm = model.fit(X_train , Y_train ,
                          batch_size = n_batch,epochs=n_epoch,
                          callbacks=[earlystopping,history],
                          validation_split=0.1)


