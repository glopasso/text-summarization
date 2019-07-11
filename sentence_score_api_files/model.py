# Import dependencies
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
import re
import string

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

PUNCTUATION = ['"', "'", '.', '...', '!', '?', '(', ')', '[', ']', '{', '}', '\\', '/', ':', ',', '...',
               '$', '#', '%', '*', '%', '$', '#', '@', '--', '-', '_', '+', '=', '^', "''", '""', '']


# utility functions()

def tokens_without_punctuation(text):
    '''
     Extract words from a piece of text, removing any punctuation
    :param text: text to be tokenized into words. Could be a text with several sentences or an unique sentence
    :return: list of words from text, without punctuation
    '''
    tokens = word_tokenize(text)
    no_punctuation = [x for x in tokens if not re.fullmatch('[' + string.punctuation + ']+', x)]
    return no_punctuation


def sentence_tokenize(text):
     '''
     Correctly separates in unique sentence, sentences that are terminated by newline "/n"  but without punctuation
     :param text: a string with the text to have sentences separated
     :return: a list of strings, each one a sentence from text
     '''
     sentence_tokenized = list()
     for txt in text.split('\n'):
          sentence_tokenized += sent_tokenize(txt)
     return sentence_tokenized


def _create_list_of_tokens(words_lst, tokenizer):
     '''
     Given a list of words, returns a list of tokens that might be lemmas or stems, cleaned from stop words
     :param words_lst: list of words to be tokenized
     :param tokenizer: (string) can be 'lemma' or 'stem'
     :return: list of tokens (strings), cleaned from stop words
     '''
     stop_words = set(stopwords.words('english'))
     stop_words.update(PUNCTUATION)

     if tokenizer == 'lemma':
          token_maker = lambda word: WordNetLemmatizer().lemmatize(word).lower()
     else:
          token_maker = lambda word: PorterStemmer().stem(word).lower()
     token_lst = list()
     for word in words_lst:
          token = token_maker(word)
          if token not in stop_words:
               token_lst.append(token)
     return token_lst


def _create_list_of_ngrams(token_lst, n_gram):
     '''
     Given a list of tokens retuns a unique list consolidating from unigram up to the required n-grams
     :param token_lst: list of tokens that could be stems or lemmas
     :param n_gram:  (string) the highest n-gram type ordered (e.g.: '1-gram', '2-gram', '3-gram', etc)
     :return: consolidated cummulative list of all ranges n-gram, from the unigrams up to the required n-grams
     '''
     n_gram_lst = []
     n = int(n_gram[0])
     for i in range(1, n + 1):
          i_grams = ngrams(token_lst, i)
          n_gram_lst += [' '.join(grams) for grams in i_grams]
     return n_gram_lst


def _create_dictionary_table(text, tokenizer='stem', n_gram='1-gram'):

     # words tokenized
     words_lst = tokens_without_punctuation(text)
     # list of tokens (stems or lemmas)
     token_lst = _create_list_of_tokens(words_lst, tokenizer)
     # list on n-grams according to the input, ranging from 1-gram up to the n_gram
     n_gram_lst = _create_list_of_ngrams(token_lst, n_gram)
     # create dictionary to count the frequency of n-grams
     frequency_table = dict()
     for n_gram_item in n_gram_lst:
           if n_gram_item in frequency_table:
               frequency_table[n_gram_item] += 1
           else:
               frequency_table[n_gram_item] = 1

     return frequency_table


def _calculate_sentence_scores(sentences, frequency_table, tokenizer='stem', n_gram='1-gram'):

     # algorithm for scoring a sentence by its n-grams
     sentence_weight = dict()

     for sentence in sentences:
          words_lst = tokens_without_punctuation(sentence)
          # list of tokens (stems or lemmas)
          token_lst = _create_list_of_tokens(words_lst, tokenizer)
          # list on n-grams according to the input, ranging from 1-gram up to the n_gram
          n_gram_lst = _create_list_of_ngrams(token_lst, n_gram)
          sentence_n_gram_count_without_stop_words = 0
          for n_gram_item in n_gram_lst:

               if n_gram_item in frequency_table:
                    sentence_n_gram_count_without_stop_words += 1
                    if sentence in sentence_weight:
                         sentence_weight[sentence] += frequency_table[n_gram_item]
                    else:
                         sentence_weight[sentence] = frequency_table[n_gram_item]

          # take the average score of the sentence
          # make sentences with only stop words to have zero score
          # also make sentences of single-character sentences of punctuation or special characters to have zero score
          if sentence in sentence_weight and sentence_weight[sentence] > 0:
               sentence_weight[sentence] = sentence_weight[sentence] / sentence_n_gram_count_without_stop_words
          else:
               sentence_weight[sentence] = 0

     return sentence_weight


def _calculate_average_score(sentence_weight):
     # calculating the average score for the sentences
     sum_values = 0
     for entry in sentence_weight:
          sum_values += sentence_weight[entry]

     # getting sentence average value from source text
     if len(sentence_weight) != 0:
          average_score = (sum_values / len(sentence_weight))
          return average_score
     else:
          return 0


def _get_article_summary(sentences, sentence_weight, threshold):
     sentence_counter = 0
     article_summary = ''

     for sentence in sentences:
          if sentence in sentence_weight and sentence_weight[sentence] >= (threshold):
               article_summary += " " + sentence
               sentence_counter += 1

     return article_summary


def run_article_summary(article, tokenizer='stem', n_gram='1-gram', threshold_factor=1):
     # creating a dictionary for the word frequency table

     frequency_table = _create_dictionary_table(article, tokenizer, n_gram)

     # tokenizing the sentences
     sentences = sentence_tokenize(article)

     # algorithm for scoring a sentence by its words
     sentence_scores = _calculate_sentence_scores(sentences, frequency_table, tokenizer, n_gram)

     # getting the threshold
     average_score = _calculate_average_score(sentence_scores)

     # producing the summary
     article_summary = _get_article_summary(sentences, sentence_scores, threshold_factor * average_score)

     article_summary = article_summary.lstrip()

     return article_summary



