

# acc : 0.821
# CRNN /home/derricksu/model_para/RCNN

from keras.models import Sequential ,Model
from keras.layers import Input , Dense,Dropout,Flatten , BatchNormalization , Concatenate
from keras.layers import LSTM , Conv1D , ZeroPadding1D , MaxPooling1D ,AveragePooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence,text
from keras import regularizers


def load_data(path,train=True):
    import numpy as np
    
    sentence = np.genfromtxt(path,dtype='str')
    _label = []
    _sentence = []
    sentence=res.text.split("\n")
    sentence.remove("")
    if train:
        sp=" +++$+++ "
    else:
        sp=","
    
    for row in sentence:
        sp_index = row.find(sp)
        _label.append(row[0:sp_index])
        _sentence.append(row[sp_index+1::])

    x = np.array(_sentence)
    y = np.array(_label)
    return np.c_[y,x]
    
    
def load_nolabel(path):
    import numppy as np
    
    sentence=np.genfromtxt(path,dtype='str')
    none_index = []
    for i,row in enumerate(sentence):
        if row =="":
            none_index.append(i)

    s_index = np.arange(0,len(sentence))
    end = len(sentence)
    l=0
    TF=[]
    while(l<end):
        if l in none_index:
            TF.append(False)
        else:
            TF.append(True)
        l+=1

    return np.array(sentence)[TF]
    
    
    
    

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    import re
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9,+^!.\/'-=?]", " ", text) # origin is [^A-Za-z0-9^,!.\/'+-=?]
    text = re.sub(r"what ' s", "what is ", text)
    # my turn
    text = re.sub(r"he ' s", "he is", text)
    text = re.sub(r"she ' s", "she is", text)
    text = re.sub(r"it ' s", "it is", text)
    text = re.sub(r"let ' s", "let us", text)
    text = re.sub(r"can ' t", "can not ", text)
    text = re.sub(r"4get", "forget ", text)
    text = re.sub(r"coo[o]+", "cooo", text)
    text = re.sub(r"so[o]+", "sooo ", text)
    text = re.sub(r" [0-9]+ : [0-9]+", " aa:bb ", text) #時間
    text = re.sub(r" [0-9]+", " ", text) #去除純數字
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\.\.[\.]+", " a... ", text)
    text = re.sub(r"uh[h]+", " uh ", text)
    text = re.sub(r"zz[z]+", " zzz ", text)
    text = re.sub(r"loo[o]+l", " lool ", text)
    text = re.sub(r" \.\.", " ", text)
    
    #end my turn
    text = re.sub(r"\' s", " ", text)
    text = re.sub(r"\' ve", " have ", text)
    text = re.sub(r"can ' t", "can not ", text)
    text = re.sub(r"n ' t", " not ", text)
    text = re.sub(r"i ' m", "i am ", text)
    text = re.sub(r"\' re", " are ", text)
    text = re.sub(r"\' d", " would ", text)
    text = re.sub(r"\' ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r" \.", " ", text)
    
    text = re.sub(r"!![!]+", " !!! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r" \=", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r" : ", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    #text = re.sub(r" u s ", " american ", text)
    #text = re.sub(r"\0s", "0", text)
    #text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return text



min_c = 3

def build(weight_matrix , vocab,max_review_length):
    embedding_vecor_length = weight_matrix.shape[1]
    word_total_index = len(vocab)

    forget_bias_bool=True #default is False
    recu_drop_rate=0.2
    drop_rate=0.2
    back_bool=False #default False
    unroll_bool=True #default is False. This can speed up but consume more memory
    multi_layer_bool=True #default False

    model = Sequential()
    model.add(Embedding(word_total_index, embedding_vecor_length, input_length=max_review_length,
                       weights = [weight_matrix]))
    
    model.add(LSTM(128,
                  unit_forget_bias=forget_bias_bool,
                  recurrent_dropout=recu_drop_rate,
                  dropout=drop_rate,
                  go_backwards=back_bool,
                  unroll=unroll_bool,
                  return_sequences=multi_layer_bool )) 
                   # Control output of this layer , return a decoded sequence whose dimension is same as input
    model.add(Dropout(0.2))
    
    model.add(LSTM(256,
                  unit_forget_bias=forget_bias_bool,
                  recurrent_dropout=recu_drop_rate,
                  dropout=drop_rate,
                  go_backwards=False,
                  unroll=unroll_bool))
    model.add(Dropout(0.2))

    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    print(model.summary())

    return model