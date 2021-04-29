import streamlit as st
import pandas as pd
import numpy as np
import codecs
from streamlit import components
from core import *


uploaded_file = st.file_uploader("Choose a XLSX file", type="xlsx")
if uploaded_file:
    df = pd.read_excel(uploaded_file,sheet_name='Sheet1',engine='openpyxl')
    user_input = st.text_input("Input the variable which contains text")
    if user_input:
        st.dataframe(df[[user_input]].head())
        num_topics = st.text_input("Enter the number of topics:")
        st.write("model training started")
        model,data_ready=pipeline(df[[user_input]],num_topics=num_topics)
        st.write("model training completed")
        id2word = corpora.Dictionary(data_ready)
        corpus = [id2word.doc2bow(text) for text in data_ready]
        vis = pyLDAvis.gensim_models.prepare(model, corpus, dictionary=id2word,mds='mmds')
        html = pyLDAvis.prepared_data_to_html(vis)
        components.v1.html(html, width=1300, height=800, scrolling=True)
