from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from flask import request, escape
import numpy as np
import re
import time
from fuzzywuzzy import fuzz
import itertools
import pickle
from google.cloud import storage
import google.cloud.logging as cloud_logging

cloud_client = cloud_logging.Client()
log_name = 'summarizer-log'
cloud_logger = cloud_client.logger(log_name)

############################################
### Preprocessing for text and tokenization
############################################

### Preprocess data for seq2seq model
#### Including:
####- Convert everything to lowercase
####- Contraction mapping
####- Remove (‘s)
####- Remove parenthesis ( ), but keep text inside
####- Insert space before punctuation
####- Insert period to sentences missing punctuation
####- Eliminate special characters


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}

def preprocessing(text):
    '''
    prepare text input for the model accepted formats
    :param text: string that will treated to enter correctly into the model
    :return: string treated and sanitized
    '''

    s = text.lower()
    s = re.sub(r'[()]', '', s) # remove parenthesis
    s = re.sub(r'[\[\]]', '', s) # remove brackets
    s = re.sub('"','', s)     # remove double quotes  
    s = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in s.split(" ")])
    s = re.sub(r"'s\b","", s) # remove "'s"
    s = re.sub("'",'', s)     # remove single quotes  
    s = re.sub(r'([!?])\B', r' \1', s) # insert space before punctuation
    s = re.sub(r'([:,;]\B)', r' \1', s) # insert space before ending colon, semi-colon, semi-period
    s = re.sub(r'([^.::?!]\n)', r'\1.', s)# include period to mark end of sentence if missing punctuation
    s = ' '.join(treatment_of_period(i) for i in s.split()) # include space before periods, but cancel this action if the period is into an acronym or web address
    s = re.sub(r'\s+', r' ', s) # multiple spaces into 1 space
    

    return s

def treatment_of_period(word):
    '''
    locate substrings with periods and give it adequate treatment, dependening on if it is an acronym,
    the final word of sentence ended with a period, or an URL (web address)
    :param word: string containing single word
    :return: word with a space before period, but do nothing in cases of acronyms and URLs
    '''
    result = word
    # check for acronyms
    if re.search('\w[.]+\w[.]+', word) != None:
      result = re.sub('[.]+', '', word)
      return result

    # check for URLs, web site address
    if re.match(r'(https?:\/\/)?([\w\d]+\.)?[\w\d]+\.\w+\/?.+', word) != None:
      if word[-1] == '.':
        result = word[:-1] + ' .'
        return result
      return word

    # check for single period or periods after word
    if word[-1] == '.':
      result = re.sub(r'[.]+', '', word)
      return result + ' .'

    return word



"""### Tokenize inputs and outputs"""

def convert_to_ints(text_list, max_token, start_token = False, stop_token = False):
    '''

    :param text_list: list of words in the input source
    :param max_token: maximum number fo tokens allowed to enter the seq-2-seq model
    :param start_token: special token to start prediction
    :param stop_token: token used for training to indicate that the target summary has finished
    :return: list of numerical tokens, and the number of words in the list and
            number of unknow words (out-of-vocabulary)
    '''
    word_count = 0
    unk_count = 0
    ints = []
   
    for text in text_list:
        text_int = []
        word_text = 0
        if start_token:
            text_int.append(vocab_to_int[START_DECODING])
            word_count += 1
            word_text += 1
        for word in text.split():
            word_count += 1
            word_text += 1
            if word_text >= max_token - 1*stop_token:
              break
            if word in vocab_to_int:
                text_int.append(vocab_to_int[word])
            else:
                text_int.append(vocab_to_int[OOV_DECODING])
                unk_count += 1
        
        if stop_token:
            text_int.append(vocab_to_int[STOP_DECODING])
        ints.append(text_int)
    return ints, word_count, unk_count

def padding(ints, max_token, stop_token = False):
    '''

    :param ints: list of lists of integers
    :param max_token: expected lenght of input for the seq-2-seq model
    :param stop_token: if true includes, a stop token in the end (for training purposes)
    :return: list of integers with padding to complete the list with length equals to max_token
    '''
    new_ints =[]
    assert len(ints) <= max_token
    for int in ints:
          len_int = len(int)
          if len_int < max_token:
                  padding = [vocab_to_int[PAD_DECODING]]*(max_token-len_int)
                  if stop_token:
                      int = int[:-1]
                      int.extend(padding)
                      int.append(vocab_to_int[STOP_DECODING])
                  else:
                      int.extend(padding)
          new_ints.append(int)
    return new_ints

###############################################################
### Model Building Seq-2-seq unidirectional
################################################################
## Encoder subclass

class Encoder(tf.keras.Model):

  def __init__(self, 
               vocab_size, 
               embedding_size, 
               embedding_matrix, 
               lstm_size, 
               pre_trained_embeddings = False,
               trainable_embeddings = True,
               dropout_rate = 0.0):

      super(Encoder, self).__init__()

      #lstm_size contain the number of dimensions of the output
      self.lstm_size = lstm_size
      self.dropout_rate = dropout_rate
      #embedding with vocabulary size as input and dimension of embedding
      self.pre_trained_embeddings = pre_trained_embeddings
      if pre_trained_embeddings:
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                 embedding_size, 
                                                 weights=[embedding_matrix], 
                                                 trainable=trainable_embeddings) 
      else:
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                 embedding_size, 
                                                 trainable=trainable_embeddings) 

      self.backward_layer = tf.keras.layers.LSTM(lstm_size // 2, return_sequences=True, 
                                                return_state=True, go_backwards=True)
      self.forward_layer = tf.keras.layers.LSTM(lstm_size // 2, return_sequences=True, 
                                                return_state=True)
      
      self.bidirectional = tf.keras.layers.Bidirectional(layer = self.forward_layer, 
                                                         backward_layer= self.backward_layer, 
                                                         merge_mode = 'concat')
      self.dropout = tf.keras.layers.Dropout(rate = dropout_rate)
     

  def call(self, sequence, states, training_flag = False):
      embed = self.embedding(sequence)
      embed_dropout = self.dropout(embed, training = training_flag )
      outputs = self.bidirectional(embed_dropout, initial_state=states)
      #outputs = self.dropout(outputs, training = training_flag)
      output = outputs[0]
      state_f_h = outputs[1]
      state_f_c = outputs[2]
      state_b_h = outputs[3]
      state_b_c = outputs[4]
      state_h = tf.concat([state_f_h, state_b_h],-1)
      state_c = tf.concat([state_f_c, state_b_c],-1)     
      return output, state_h, state_c

  def init_states(self, batch_size):
 
      return [tf.zeros([batch_size, self.lstm_size // 2]),
              tf.zeros([batch_size, self.lstm_size // 2]),
              tf.zeros([batch_size, self.lstm_size // 2]),
              tf.zeros([batch_size, self.lstm_size // 2])]     

# Luong Attention subclass   

class LuongAttention(tf.keras.Model):
    def __init__(self, rnn_size):
        super(LuongAttention, self).__init__()
        self.wa = tf.keras.layers.Dense(rnn_size)

    def call(self, decoder_output, encoder_output):
        # Dot score: h_t (dot) Wa (dot) h_s
        # encoder_output shape: (batch_size, max_len, rnn_size)
        # decoder_output shape: (batch_size, 1, rnn_size)
        # score will have shape: (batch_size, 1, max_len)
        score = tf.matmul(decoder_output, self.wa(encoder_output), transpose_b=True)
        # alignment vector a_t
        alignment = tf.nn.softmax(score, axis=2)
        # context vector c_t is the average sum of encoder output
        context = tf.matmul(alignment, encoder_output)

        return context, alignment

## Decoder subclass    

class Decoder(tf.keras.Model):

  def __init__(self, 
               vocab_size, 
               embedding_size, 
               embedding_matrix, 
               lstm_size, 
               pre_trained_embeddings = False,
               trainable_embeddings = True):

      super(Decoder, self).__init__()
      self.attention = LuongAttention(lstm_size)
      self.lstm_size = lstm_size
      self.pre_trained_embeddings = pre_trained_embeddings
      if pre_trained_embeddings:
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                 embedding_size, 
                                                 weights=[embedding_matrix], 
                                                 trainable=trainable_embeddings) 
      else:
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                 embedding_size, 
                                                 trainable=trainable_embeddings) 
      self.lstm = tf.keras.layers.LSTM(
          lstm_size, return_sequences=True, return_state=True)
      self.wc = tf.keras.layers.Dense(lstm_size, activation='tanh')
      self.ws = tf.keras.layers.Dense(vocab_size)

  def call(self, sequence, state, encoder_output):
      # Remember that the input to the decoder
      # is now a batch of one-word sequences,
      # which means that its shape is (batch_size, 1)
      embed = self.embedding(sequence)
      # Therefore, the lstm_out has shape (batch_size, 1, rnn_size)
      lstm_out, state_h, state_c = self.lstm(embed, initial_state = state)
      # Use self.attention to compute the context and alignment vectors
      # context vector's shape: (batch_size, 1, rnn_size)
      # alignment vector's shape: (batch_size, 1, source_length)
      context, alignment = self.attention(lstm_out, encoder_output)


      # Combine the context vector and the LSTM output
      # Before combined, both have shape of (batch_size, 1, rnn_size),
      # so let's squeeze the axis 1 first
      # After combined, it will have shape of (batch_size, 2 * rnn_size)
      lstm_out = tf.concat([tf.squeeze(context, 1), tf.squeeze(lstm_out, 1)], 1)

      # lstm_out now has shape (batch_size, rnn_size)
      lstm_out = self.wc(lstm_out)
        
      # Finally, it is converted back to vocabulary space: (batch_size, vocab_size)
      logits = self.ws(lstm_out)

      return logits, state_h, state_c, alignment

def build_model(tokenizer,  
                embedding_matrix_article,
                embedding_matrix_summary,
                data_article_input,
                data_summary_input,
                dropout_rate = 0.0):
    '''

    :param tokenizer: vocab_to_int dictionary
    :param embedding_matrix_article: the matrix with embedding vectors for input in the encoder
    :param embedding_matrix_summary: the matrix with embedding vectors for input in the decoder
    :param data_article_input: list of integers that the model will take as input
    :param data_summary_input: list of integers that the model will produce as output
    :param dropout_rate: float frow 0 to 1 with the rate for dropout (for training)
    :return:
    '''

    ## clear graph and build the model: both encoder and decoder

    tf.keras.backend.clear_session()
    embedding_size = EMBEDDING_SIZE
    article_vocab_size = len(tokenizer)
    encoder = Encoder(article_vocab_size,
                      embedding_size,
                      embedding_matrix_article,
                      LSTM_SIZE,
                      pre_trained_embeddings = PRE_TRAINED_EMBEDDINGS,
                      trainable_embeddings = TRAINABLE_EMBEDDINGS,
                      dropout_rate = dropout_rate)
    
    summary_vocab_size = len(tokenizer)
    decoder = Decoder(summary_vocab_size,
                      embedding_size,
                      embedding_matrix_summary,
                      LSTM_SIZE,
                      pre_trained_embeddings = PRE_TRAINED_EMBEDDINGS,
                      trainable_embeddings = TRAINABLE_EMBEDDINGS)

    source_input = tf.constant(data_article_input[:BATCH_SIZE])
    initial_state = encoder.init_states(BATCH_SIZE)
    encoder_output, en_state_h, en_state_c = encoder(source_input, initial_state, training_flag = True)
    target_input = tf.constant(data_summary_input[:BATCH_SIZE])
    decoder_input = tf.expand_dims(target_input[:, 0], 1)
    decoder_output, de_state_h, de_state_c, alignment = decoder(decoder_input, [en_state_h, en_state_c], encoder_output)
    cloud_logger.log_text('Build model: model built and ready to receive weights')
    return encoder, decoder

def load_model(path, encoder, decoder):
    '''

    :param experiment_name: path and main name part of the files that identify encoder and decoder weights
    :param encoder: class encoder object to be filled out with the pre-trained weights
    :param decoder: class decoder object to be filled out with the pre-trained weights
    :return: encoder and decoder with loaded pre-trained weights
    '''
    ## Load model weights
    weights_encoder_load = np.load(path + 'encoder.npy', allow_pickle = True)
    weights_decoder_load = np.load(path + 'decoder.npy', allow_pickle = True)
    encoder.set_weights(weights_encoder_load)
    cloud_logger.log_text('Build model: encoder weights loaded')
    decoder.set_weights(weights_decoder_load)
    cloud_logger.log_text('Build model: decoder weights loaded')
    return encoder, decoder

def uniquify(string):
    '''
    Remove repeating words for output cleaning
    :param string: input string
    :return: original string without repeated words
    '''
    output = []
    seen = set()
    for word in string.split():
        if word not in seen:
            output.append(word)
            seen.add(word)
    return ' '.join(output) 

    
def remove_similar_sentences(sentence_list):
      '''
        use fuzzywuzzy to find similar sentences and take the longest one
        :param sentence_list: list of sentences from the output generated by seq-2-seq
        :return: sentence_list sanitized from very similar sentences
      '''
      length = len(sentence_list)
      if length == 0:
        return sentence_list
      combinations = list(itertools.combinations(list(range(length)),2))
      indexes_to_drop = []
      COMPARISON_THRESHOLD = 70
      for i in combinations:
          if fuzz.partial_ratio(sentence_list[i[0]],sentence_list[i[1]]) > COMPARISON_THRESHOLD:
            if len(sentence_list[i[0]]) > len(sentence_list[i[1]]):
              indexes_to_drop.append(i[1])
            else:
              indexes_to_drop.append(i[0])
      indexes_to_drop = list(set(indexes_to_drop))
      for index in sorted(indexes_to_drop, reverse=True):
          del sentence_list[index]
      return sentence_list


def post_processing(text):
  '''
    sanitize model output excluding special tokens (stop, padding, and OOV),
    removing repeated words and similar sentences
    :param text: raw output of the model
    :return: sanitized output
  '''
  if not(text):
    return
  if STOP_DECODING in text:
    text = text.replace(' '+STOP_DECODING,'')
  if PAD_DECODING in text:
    text = text.replace(' '+PAD_DECODING, '')
  if OOV_DECODING in text:
    text = text.replace(' '+OOV_DECODING, '')
    text = text.replace(OOV_DECODING, '')

  punctuation = ['.', '!', '?']
  s = text
  # beginning of processing
  sentence = s.split('.')
  cleaned_sentences = []
  for sen in sentence:
    sen = uniquify(sen)
    if sen:
      sen = sen[0].upper() + sen[1:]
      if sen[-1] in punctuation:
        sen = sen[:-1]
    cleaned_sentences.append(sen)
  cleaned_sentences = remove_similar_sentences(cleaned_sentences)
  s = '.\n'.join(cleaned_sentences)
  s = re.sub(r' ([.!?,:])', r'\1', s)
  s = s.strip()
  if s:
    if s[-1] not in punctuation:
       s += '.'
  return s
  

def predictions(encoder, decoder, data_article_input, vocab_to_int, int_to_vocab):
    '''
    Predict the summary for an article. Articles longer than 400 tokens are broken down into chunkcs of 400,
    and separated summaries are created for each chunk
    :param encoder: class object encoder prepared for predictions (i.e.: loaded with pretrained weights)
    :param decoder: class object dencoder prepared for predictions (i.e.: loaded with pretrained weights)
    :param data_article_input: list of tokenized chunks of articles for prediction
    :param vocab_to_int: dictionary for tokenization
    :param int_to_vocab: dictonary to return words from tokens
    :return: summaries for each chunk, already post-processed
    '''
  
    created_summaries = list()
    for article_index in range(len(data_article_input)):
        test_source_seq = data_article_input[article_index]

        en_initial_states = encoder.init_states(1)
        en_output, enc_state_h, enc_state_c = encoder(tf.constant([test_source_seq]), en_initial_states, training_flag = False)
        de_input = tf.constant([[vocab_to_int[START_DECODING]]])
        de_state_h, de_state_c = enc_state_h, enc_state_c
        out_words = []


        while True:
            de_output, de_state_h, de_state_c, alignment = decoder(
            de_input, (de_state_h, de_state_c), en_output)
            de_input = tf.expand_dims(tf.argmax(de_output, -1), 0)
            out_words.append(int_to_vocab[de_input.numpy()[0][0]])

            if out_words[-1] == STOP_DECODING or len(out_words) > MAX_TOKEN_OUTPUT:
                   break
        created_summary = ' '.join(out_words)
        created_summaries.append(created_summary)
    created_summaries= list(map(post_processing, created_summaries))
    return created_summaries


def splitter(article):
    '''
    Given an article of any length produce articles with length less or equal the max_token size
    :param article: text to be summarized
    :return: list of article chunks
    '''
    length = len(article.split())
    article_chunks = []
    if length > MAX_WORD_ARTICLES:
        chunks = (length // MAX_WORD_ARTICLES) + 1
        sentences = article.split('.')
        sentences_per_chunk = int(len(sentences) / chunks)
        for i in range(chunks):
            if i == chunks - 1:
                article_chunks.append('.'.join(sentences[i * sentences_per_chunk:]))
                #
            else:
                article_chunks.append(
                    '.'.join(sentences[i * sentences_per_chunk:i * sentences_per_chunk + sentences_per_chunk]))
    else:
        return [article]

    return article_chunks

def download_blob(bucket_name, source_blob_name, destination_file_name):
    '''
    Download files from the a bucket in GCP, Google Storage service
    :param bucket_name: the name of the bucket with the files
    :param source_blob_name: file name in the bucket
    :param destination_file_name: file name to be written on Google Cloud Functions instance
    '''
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)
    ## convert to log
    cloud_logger.log_text('Download from GS: Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))
    #print('Blob {} downloaded to {}.'.format(
    #    source_blob_name,
    #    destination_file_name))
    

## prepare model at cold start
def prepare_model():
    '''
    Prepare model at cold start, downloading tokeninzer and pre-trained weights
    :return: vocab_to_int, int_to_vocab, encoder, decoder
    '''
    # main variables
    EXPERIMENT_NAME = 'CNN_400_80_20K_sep26'
	#paths and files
    BUCKET_NAME = 'capstone-lopasso'
    PROJECT_PATH = '/tmp/'

    download_blob(BUCKET_NAME, 'abstractive/CNN_400_80_20K_sep26_decoder.npy', '/tmp/decoder.npy')
    download_blob(BUCKET_NAME, 'abstractive/CNN_400_80_20K_sep26_encoder.npy', '/tmp/encoder.npy')
    download_blob(BUCKET_NAME, 'abstractive/int_to_vocab.pkl', '/tmp/int_to_vocab.pkl')
    download_blob(BUCKET_NAME, 'abstractive/vocab_to_int.pkl', '/tmp/vocab_to_int.pkl')
    download_blob(BUCKET_NAME, 'abstractive/word_embedding_matrix.npy', '/tmp/word_embedding_matrix.npy')


    f = open(PROJECT_PATH + "vocab_to_int.pkl","rb")
    vocab_to_int = pickle.load(f)
    f.close()
    cloud_logger.log_text('Model preparation: tokenizer vocab_to_int ready with {}  tokens'.format(len(vocab_to_int)))

    f = open(PROJECT_PATH + "int_to_vocab.pkl","rb")
    int_to_vocab = pickle.load(f)
    f.close()
    cloud_logger.log_text('Model preparation: de-tokenizer int_to_vocab ready with {}  tokens'.format(len(int_to_vocab)))

    word_embedding_matrix = np.load(PROJECT_PATH + 'word_embedding_matrix.npy', allow_pickle = True)
    cloud_logger.log_text('Model preparation: word embedding matrix with size {}'.format(word_embedding_matrix.shape))


    data_article_input = np.zeros((18000,400))
    data_summary_input = np.zeros((2000,400))
    # model construction
    encoder, decoder = build_model(vocab_to_int, 
                                   word_embedding_matrix, 
                                   word_embedding_matrix,
                                   data_article_input,
                                   data_summary_input,
                                   0.0)


    ## load model parameters, if it is a continuation of previous training
    encoder, decoder = load_model(PROJECT_PATH , encoder, decoder)
    return vocab_to_int, int_to_vocab, encoder, decoder



def run_model(article):
    '''
    Function wrapper for all preprocessing, prediction and post-processing (end-to-end)
    :param article: source text to be summarized
    :return: summarized text
    '''
    preprocessed = preprocessing(article)
    article_chunks = splitter(preprocessed)
    cloud_logger.log_text('Article chunks: article was split into {} chunk'.format(len(article_chunks)))
    int_article, article_word_count, article_unk_count = convert_to_ints(article_chunks, MAX_TOKEN_INPUT)
    int_article = padding(int_article, MAX_TOKEN_INPUT )
    created_summary = predictions(encoder, decoder, int_article, vocab_to_int, int_to_vocab)
    created_summary = ' '.join(created_summary)
    return created_summary

# Declared at cold-start, but only initialized if/when the function executes
vocab_to_int = None
int_to_vocab = None 
encoder = None
decoder = None
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences
OOV_DECODING = '[UNK]' # This has a vocab id, which is used for out of vocabulary tokens
PAD_DECODING = '[PAD]'
MAX_WORD_ARTICLES = 400 #400
MAX_WORD_SUMMARIES = 80 #80
MAX_TOKEN_INPUT = 400 #400
MAX_TOKEN_OUTPUT = 80 #80
# model parameters
EMBEDDING_SIZE = 100 
LSTM_LAYERS = 1
LSTM_SIZE = 256
BATCH_SIZE = 64
PRE_TRAINED_EMBEDDINGS = True
TRAINABLE_EMBEDDINGS = True

def abstractive_summarizer(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <http://flask.pocoo.org/docs/1.0/api/#flask.Request>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>.
    """
    t_zero = time.time()
    text = [i for i in request.form.keys()][0]
    cloud_logger.log_text('API Call: text with {} characters and {} words'.format(len(text),len(text.split())))
    global vocab_to_int, int_to_vocab, encoder, decoder
    if (not vocab_to_int) and (not int_to_vocab) and (not encoder) and (not decoder):
        cloud_logger.log_text('Model preparation: function called')
        vocab_to_int, int_to_vocab, encoder, decoder = prepare_model()
    created_summary = run_model(text)
    if not created_summary:
        cloud_logger.log_text('Created Summary error: empty response')
        created_summary = 'ERROR: Empty response'
    cloud_logger.log_text('API Response: took {} seconds'.format(round(time.time() - t_zero,2)))
    return '{}'.format(created_summary)







