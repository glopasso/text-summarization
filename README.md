# Capstone Project - Text Summarization
### Capstone project at Springboard (work in progress repository)

## Contents
1. Project description
2. Process raw data
3. Exploratory data analysis (EDA)
4. Sentence scoring algorithm
5. Flask API on a web server

## 1. Project description

The objective of this project is to develop a text summarization tool able to create a short version of a given document retaining it most important information. This task is relevant for to access textual information and produce digests of news, social media and reviews. It can also be applied as part of other AI tasks such as answering questions and providing recommendations.

Dataset: The CNN news highlights dataset, which contains news articles and associated highlights, i.e., a few bullet points giving a brief overview of the article, with 92,579 documents.

The CNN dataset was downloaded from New York University, in the version made available by Kyunghyun Cho.

A description of this project development can be found on my portfolio website

## 2. Data cleaning

Basic processing of the original dataset file separting article from summaries.

01-process-raw-data.ipynb [launch notebook on Codelab](https://colab.research.google.com/github/glopasso/capstone/blob/master/notebooks/01-process-raw-data.ipynb)

## 3. Exploratory Data Analysis (EDA)
Analysis of number of characteres, words and sentences on both articles and summaries. Identification of malformed articles and cleaning the dataset from them.

02-exploratory-data-analysis.ipynb [launch notebook on Codelab]

## 4. Sentence scoring algorithm

The sentence scoring algorithm was mostly based on Alfrick Opidi's article on Floydhub, named "A Gentle Introduction to Text Summarization in Machine Learning".

03-sentence-scoring-algorithm.ipynb [launch notebook on Codelab]

## 5. Flask API on a web server

#### HTTP POST calls to the API
Format: curl --data-binary @ -d 'tokenizer=<stem | lemma>&n_gram=<1-gram |2-gram | 3-gram>&threshold_factor=' https://summarizer-lopasso.herokuapp.com/predict

The response is a JSON in the following format:

{"prediction" : "The generated summary"}

#### Web interface
Access the app on Heroku using the [link](https://summarizer-lopasso.herokuapp.com/)
The app has a self explanatory page, where one input the text to be summarized and the algorithm parameters. The generate summary appears in the field on the bottom of the page, when the button "Submit" is pressed.
