from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

# Path for GCP
PROJECT_PATH = './'
#CORPUS_FILE_NAME = 'corpus_clean_dataframe_with_statistics.pkl'

import numpy as np
import re
from fuzzywuzzy import fuzz
import itertools
import pickle

"""# Supporting functions

### Extract a dataset from CNN corpus
"""


"""### Preprocess data for seq2seq model

Including:
- Convert everything to lowercase
- Contraction mapping
- Remove (‘s)
- Remove parenthesis ( ), but keep text inside
- Insert space before punctuation
- Insert period to sentences missing punctuation
- Eliminate special characters
"""

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
      result = result = re.sub(r'[.]+', '', word)
      return result + ' .'

    return word



"""### Tokenize inputs and outputs"""

def convert_to_ints(text_list, max_token, start_token = False, stop_token = False):
    '''Convert words in text to an integer.
       If word is not in vocab_to_int, use UNK's integer.
       Total the number of words and UNKs.
       Add EOS token to the end of texts'''
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
  new_ints =[]
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


def truncate_data_sequence(data_sequence, truncate_factor):
    original_length = len(data_sequence)
    truncated_length = int(original_length*truncate_factor)
    truncated = np.append(data_sequence[:truncated_length], [0.0]*(original_length -  truncated_length)).astype('int32')
    return truncated

"""### Model Building Seq-2-seq unidirectional"""

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

      # LSTM layer with 
      ## return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
      ## return_state: Boolean. Whether to return the last state in addition to the output

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
     #print(encoder_output, en_state_h, en_state_c)
    #
    target_input = tf.constant(data_summary_input[:BATCH_SIZE])
    decoder_input = tf.expand_dims(target_input[:, 0], 1)
    decoder_output, de_state_h, de_state_c, alignment = decoder(decoder_input, [en_state_h, en_state_c], encoder_output)
    #
    # print('Source vocab size', article_vocab_size)
    # print('Source sequences', source_input.shape)
    # print('Encoder outputs', encoder_output.shape)
    # print('Encoder state_h', en_state_h.shape)
    # print('Encoder state_c', en_state_c.shape)
    #
    # print('\nDestination vocab size', summary_vocab_size)
    # print('Destination sequences', target_input.shape)
    # print('Decoder outputs', decoder_output.shape)
    # print('Decoder state_h', de_state_h.shape)
    # print('Decoder state_c', de_state_c.shape)
    # #print(encoder.summary())
    # print(decoder.summary())
    
    return encoder, decoder




      
      




def save_model(encoder, decoder, checkpoint = False):
    ## Save model weights for future retrieval, avoiding retraining
    if checkpoint:
      main_name = EXPERIMENT_NAME + '_checkpoint'
      print('saving checkpoint')
    else:
      main_name = EXPERIMENT_NAME
    weights_encoder = encoder.get_weights()
    np.save(SAVE_PATH + main_name +  '_weights_encoder.npy', weights_encoder)
    weights_decoder = decoder.get_weights()
    np.save(SAVE_PATH + main_name +  '_weights_decoder.npy', weights_decoder)
    tr_val_loss_df.to_pickle(SAVE_PATH + main_name + '_losses.pkl')

def load_model(experiment_name):
    ## Load model weights
    weights_encoder_load = np.load(PROJECT_PATH + experiment_name + '_encoder.npy', allow_pickle = True)
    weights_decoder_load = np.load(PROJECT_PATH + experiment_name + '_decoder.npy', allow_pickle = True)
    encoder.set_weights(weights_encoder_load)
    decoder.set_weights(weights_decoder_load)
    return encoder, decoder

def uniquify(string):
    #print('strings: ', string)
    output = []
    seen = set()
    for word in string.split():
        if word not in seen:
            output.append(word)
            seen.add(word)
    return ' '.join(output) 

    
def remove_similar_sentences(sentence_list):
  
  length = len(sentence_list)
  if length == 0:
    return sentence_list
  combinations = list(itertools.combinations(list(range(length)),2))
  indexes_to_drop = []
  for i in combinations:
      if fuzz.partial_ratio(sentence_list[i[0]],sentence_list[i[1]]) > 70:
        if len(sentence_list[i[0]]) > len(sentence_list[i[1]]):
          indexes_to_drop.append(i[1])
        else:
          indexes_to_drop.append(i[0])
  indexes_to_drop = list(set(indexes_to_drop))
  for index in sorted(indexes_to_drop, reverse=True):
      del sentence_list[index]
  return sentence_list


def post_processing(text):
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
  
    created_summaries = list()
    for article_index in range(len(data_article_input)):
        test_source_seq = data_article_input[article_index]

        en_initial_states = encoder.init_states(1)
        en_output, enc_state_h, enc_state_c = encoder(tf.constant([test_source_seq]), en_initial_states, training_flag = False)
        de_input = tf.constant([[vocab_to_int[START_DECODING]]])
        de_state_h, de_state_c = enc_state_h, enc_state_c
        out_words = []
        alignments = []

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
    return created_summaries, np.array(alignments)


    

"""# Run the Model"""

# global variables
EXPERIMENT_NAME = 'CNN_400_80_20K_sep26'

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

#paths and files
PROJECT_PATH = './'
CORPUS_FILE_NAME = 'corpus_clean_dataframe_with_statistics.pkl'
SAVE_PATH = '/content/drive/My Drive/capstone/Abstractive_approach/saved_models/'

f = open(PROJECT_PATH + "vocab_to_int.pkl","rb")
vocab_to_int = pickle.load(f)
f.close()

f = open(PROJECT_PATH + "int_to_vocab.pkl","rb")
int_to_vocab = pickle.load(f)
f.close()

word_embedding_matrix = np.load(PROJECT_PATH + 'word_embedding_matrix.npy', allow_pickle = True)

text = "you're my sunshine."
print(preprocessing(text))

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
encoder, decoder = load_model('CNN_400_80_20K_sep26')




"""# Prediction"""



def splitter(article):
   
   length = len(article.split())
   #print(length)
   article_chunks = []
   if length > MAX_WORD_ARTICLES:
        chunks = (length // MAX_WORD_ARTICLES) + 1
        #print(chunks)
        chunk_size = int(length/chunks)
        sentences = article.split('.')
        sentences_per_chunk = int(len(sentences)/chunks)
        #print(len(sentences), sentences_per_chunk)
      
        for i in range(chunks):
          if i == chunks-1:
            article_chunks.append('.'.join(sentences[i*sentences_per_chunk:]))
            #print('fim')
          else:
            #print('inicio')
            article_chunks.append('.'.join(sentences[i*sentences_per_chunk:i*sentences_per_chunk + sentences_per_chunk]))
   else:
       return [article]
   #print(article_chunks)
   return article_chunks



def run_model(article):
    preprocessed = preprocessing(article)
    article_chunks = splitter(preprocessed)
    int_article, article_word_count, article_unk_count = convert_to_ints(article_chunks, MAX_TOKEN_INPUT)
    int_article = padding(int_article, MAX_TOKEN_INPUT )
    created_summary, _ = predictions(encoder, decoder, int_article, vocab_to_int, int_to_vocab)
    created_summary = ' '.join(created_summary)
    return created_summary

article = 'JACKSONVILLE, Florida    -- Four people were killed and several injured after an explosion Wednesday at a chemical plant sent a thick plume of smoke over a section of Jacksonville, authorities said.\nA thick plume of smoke rises Wednesday at a chemical plant in Jacksonville, Florida.\n"Literally, it\'s a hellish inferno. There is no other way to describe it," said Fire Department spokesman Tom Francis.\nFourteen people were hospitalized after the blast at the T2 Lab on Faye Road, in an industrial area on the waterfront in north Jacksonville, Francis said.\nOfficials initially ordered an evacuation of nearby businesses, but by 4 p.m. the order had been lifted after tests of the air found no toxicity, Francis said.\nFirefighters were still battling hot spots, and the effort will be going on for "quite some time," he said.  See an I-Report account about the blast »\nSix of those injured were transported to Shands Hospital in Jacksonville, hospital spokeswoman Kelly Brockmeier said. A Shands official said the hospital incident command system had been activated -- something done to put the staff in high alert in anticipation of trauma patients.\nA woman who answered the T2 Lab\'s 24-hour facility emergency phone said the plant manufactures ecotane, a gasoline additive that reduces tailpipe emissions, according to the laboratory\'s Web site.  See a map of the site of the explosion »\nThe billowing black smoke could be seen from the city\'s downtown, said Florida Times-Union reporter Bridget Murphy. Murphy said she talked to several workers as they walked out of the area, and they were "shaken to the core."\n"They described a hissing noise and then a sound wave," she said.\nAntonio Padrigan was trying to get in touch with his son, who works in a plant in the area, but was having no luck reaching him on his cell phone.\n"He was shook up when he called me, but I can\'t get through to him anymore," Padrigan said. "I don\'t know if he\'s in the hospital or what."\nCNN I-Reporters Jonathan Payne and his son Calvin, 16, shot pictures of the explosion. They felt the blast shake their home, about 15 minutes away, and went to see what was going on.\nCarlton Higginbotham, 63, was working at home on Townsend Boulevard in Jacksonville when a loud boom shook his house, he said.\n"It was a gunshot-type explosion; it wasn\'t a rumble," he said.\nHigginbotham, an insurance salesman, and his neighbor ran outside and noticed thick smoke billowing from the other side of the St. Johns River, which separates his neighborhood from the site of the blast.\n"The cloud that came out of it was white, some would say mushroom-shaped," Higginbotham said. "It was followed by dark, dark smoke." E-mail to a friend\n'
#sample = corpus.sample(1)
#article = sample.article.values[0]
print(article)
created_summary = run_model(article)
print(created_summary)



