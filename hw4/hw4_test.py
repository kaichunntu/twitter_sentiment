

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

load_word2vec = True
word2vec_path = 'word2vec.bin'

load_model_bool = True
model_path = 'CRNN_1st.h5'
"""
nolabel = np.load("/home/derricksu/ML_data/hw4/train_nolabel.npy")
train_path = "/home/derricksu/ML_data/hw4/train.npy"
train = np.load(train_path)
test_path = "/home/derricksu/ML_data/hw4/test.npy"
test = np.load(test_path)
predict_path = "/home/derricksu/pred/best.csv"

"""

test_path = sys.argv[1]
test = load_data(test_path,train=False)

predict_path = sys.argv[2]



test_sentence = []
for i,row in enumerate(test[1::,1]): #第一列不為name
    x = text_to_wordlist(row).split()
    test_sentence.append(x)



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
    my_word2vec.save(word2vec_path)
else:
    print("load model...") 
    my_word2vec = Word2Vec.load(word2vec_path)
    #print(my_word2vec)

vocab = dict([(k, v.index) for k, v in my_word2vec.wv.vocab.items()])
weight_matrix = my_word2vec.wv.syn0 #word_to_vec
print("Weight_matrix shape : " , weight_matrix.shape)
del my_word2vec
print("word2vec is already.")



# word to idx of test
test_word2idx=[]
for row in test_sentence:
    idx = []
    for word in row:
        if word_freq[word]>=min_c:
            idx.append(vocab[word])
    test_word2idx.append(idx)
print("Length of test_sentence : ",len(test_sentence))
print("Length of test_word2idx : ",len(test_word2idx))




#import keras
from keras import utils
from keras.models import load_model


X_test = test_word2idx

max_review_length = 100

X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

n_batch = 256
n_epoch = 10

if load_model_bool:
    model = load_model(model_path)
else:
    from keras.callbacks import EarlyStopping , History
    model = build(weight_matrix,vocab,max_review_lengt)

    earlystopping=EarlyStopping(monitor='val_categorical_accuracy', patience=3, verbose=0, mode='auto')
    history = History()

    hist_lstm = model.fit(X_train , Y_train ,
                          batch_size = n_batch,epochs=n_epoch,
                          callbacks=[earlystopping,history],
                          validation_split=0.1)

pred_y = model.predict(X_test,batch_size=n_batch)
print("Shape of predict of test : ",pred_y.shape)

with open(predict_path , "w" , encoding = "utf-8") as f :
    f.write("id,label\n")
    for i ,pre in enumerate(pred_y):
        f.write( "{0},{1}\n".format( i , np.argmax(pre) ) )
    f.close()
