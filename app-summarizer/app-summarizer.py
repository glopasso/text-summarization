import streamlit as st
import model
import requests

st.sidebar.header('Text Summarization App')

summarizer_approach = st.sidebar.radio(
    "Select summarization approach",
    ('Extractive', 'Abstractive'))
# Side bar options for extractive approach
if summarizer_approach == 'Extractive':
    st.sidebar.text('Options for extractive summarization')
    n_gram = st.sidebar.selectbox(
        'n-gram',
        ('1-gram', '2-gram', '3-gram'))
    tokenizer = st.sidebar.selectbox(
        'token type',
        ('stem', 'lemma'))
    threshold = st.sidebar.text_input('threshold factor', 1.2)
    threshold = float(threshold)

# Submit button
button = False
if st.sidebar.button('Submit'):
    button = True
else:
    button = False
st.write('Enter text to summarize:\n')

source_txt = st.text_area('Source text')

if button == True:
    if summarizer_approach == 'Extractive':
        param_summarizer = {'tokenizer': tokenizer,
                            'n_gram': n_gram,
                            'threshold_factor': threshold}
        prediction = model.run_article_summary(source_txt, **param_summarizer)
        st.write('Summary:\n')
        st.write(prediction)
    else:
        res = requests.post(url='https://us-central1-data-engineering-gcp.cloudfunctions.net/summarizer',
                            data = {source_txt : 0})
        prediction = res.json()['prediction']
        st.write('Summary:\n')
        st.write(prediction)
