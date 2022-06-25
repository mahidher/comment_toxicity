import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import nltk
from nltk.corpus import stopwords
import pickle
from nltk.tokenize import word_tokenize
import re
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

import gradio as gr

nltk.download('stopwords')

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

print("hello")

with open('comment_tokenizer.pkl', 'rb') as file:
      
    # Call load method to deserialze
    tokenizer = pickle.load(file)
  

max_len = 1348   

model = keras.models.load_model('comment_toxicity_model.h5')

CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have",
}

def expand_contractions(sentences):
  contractions_re = re.compile('(%s)'%'|'.join(CONTRACTION_MAP.keys()))
  def exp_cont(s, contractions_dict=CONTRACTION_MAP):
    def replace(match):
      return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)
  for i in range(len(sentences)):
    sentences[i] = exp_cont(sentences[i])


def remove_newlines_and_tabs(sentences):
  
  for i in range(len(sentences)):
    sentences[i] = sentences[i].replace('\n',' ').replace('\t',' ').replace('\\', ' ')

stoplist = set(stopwords.words('english'))

def remove_stopwords(sentences):
  for i in range(len(sentences)):
    tokens = word_tokenize(sentences[i])
    
    filtered_tokens = [token for token in tokens if token.lower() not in stoplist]
    sentences[i] = " ".join(filtered_tokens)


w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


def lemmetization(sentences):
  for i in range(len(sentences)):
    lemma = [lemmatizer.lemmatize(w,'v') for w in w_tokenizer.tokenize(sentences[i])]

    sentences[i] = " ".join(lemma)


def score_comment(comment):
    sentences = [comment]
    expand_contractions(sentences)
    remove_newlines_and_tabs(sentences)
    remove_stopwords(sentences)
    lemmetization(sentences)
    tokenized = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(tokenized,maxlen=max_len,padding = 'post')
    results = model.predict(padded)
    
    text = ''
    for idx, col in enumerate(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
       'identity_hate']):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)
    print(text)
    return text

# text = 'COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK'
# score_comment(text)

interface = gr.Interface(fn=score_comment, 
                         inputs=gr.inputs.Textbox(lines=2, placeholder='Comment to score'),
                        outputs='text')

interface.launch(share=True)