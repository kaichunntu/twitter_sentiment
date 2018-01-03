

import numpy as np


def load_test(path):
    return np.load(path)

def load_word_freq(path):
    import pickle
    with open(path,"rb") as pk:
        word_freq = pickle.load(pk)
        pk.close()
    return word_freq

def load_word2vec(path):
    from gensim.models import Word2Vec
    word2vec = Word2Vec.load(path)
    return word2vec.wv.syn0 , dict([(k, v.index) for k, v in word2vec.wv.vocab.items()])

def prod_sentence(x):
    test_sentence = []
    for i,row in enumerate(x[1::,1]): #第一列不為name
        x = text_to_wordlist(row).split()
        test_sentence.append(x)
    return test_sentence

def word2idx(x,min_c,vocab,word_freq):
    test_word2idx=[]
    for row in x:
        idx = []
        for word in row:
            if word_freq[word]>=min_c:
                idx.append(vocab[word])
        test_word2idx.append(idx)
    return test_word2idx


# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
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
    text = re.sub(r"[^A-Za-z0-9,!+^.\/'-=?]", " ", text) # origin is [^A-Za-z0-9^,!.\/'+-=?]
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
    text = re.sub(r" u s ", " american ", text)
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


    
    