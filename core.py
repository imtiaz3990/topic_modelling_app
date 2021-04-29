import sys
import re, numpy as np, pandas as pd
from tqdm import tqdm
tqdm.pandas()
from pprint import pprint
from langdetect import detect
# Gensim
import gensim
import spacy
import logging, warnings
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
import matplotlib.patches as mpatch
# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import pyLDAvis
import pyLDAvis.gensim_models

def sent_to_words(sentences):
    for sent in sentences:
        sent = re.sub('\S*@\S*\s?', '', str(sent))  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent) 
        
def getPhrasers(data_words):
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return bigram_mod,trigram_mod 

def process_words(texts, bigram_mod,trigram_mod,stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load('en_core_web_sm')
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    return texts_out
    
def process_data(topics_data):
    topics_data = topics_data.values.tolist()
    data_words = list(sent_to_words(topics_data))
    print(data_words[:1])
    bigram_mod,trigram_mod=getPhrasers(data_words)
    data_ready = process_words(data_words,bigram_mod,trigram_mod)
    return data_ready

def get_lda_model(data_ready,num_topics):
    #Create Dictionary
    id2word = corpora.Dictionary(data_ready)
    #Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_ready]
    #Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_topics, 
                                               random_state=100,
                                               update_every=1,
                                               chunksize=10,
                                               passes=10,
                                               alpha='symmetric',
                                               iterations=100,
                                               per_word_topics=True)
    pprint(lda_model.print_topics())
    return lda_model
    
def pipeline(topics_data,num_topics=5):
    data_ready=process_data(topics_data)
    model=get_lda_model(data_ready,num_topics)
    return model,data_ready