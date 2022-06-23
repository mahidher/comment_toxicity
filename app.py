import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import nltk
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
import re
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

print("hello")