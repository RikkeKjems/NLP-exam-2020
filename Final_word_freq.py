# %%
# IMPORT PACKAGES
import collections
import numpy as np
from string import digits
from os import path
from pandas import DataFrame
from nltk.corpus import stopwords
from spacy.lang.da.stop_words import STOP_WORDS
import regex
import logging
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import re
from collections import Counter
from functools import reduce
from operator import add
import spacy
import nltk
import lemmy
import stanza
import pandas as pd
import sys
import glob
import os.path
import lemmy.pipe
import morfessor
from polyglot.text import Text

# %%
# Loop med df som output. filenavn + antal unique words in each file

# Nok en fil ad gangen
path = glob.glob('Data/Final_UTF8_data/ND_data/ND_Tokenfolder/*.txt')

idx = []  # filenames for rows = 22
# number = []  # burde også være 22 et tal for hver doc
for t in path:
    data = open(t, "r").read()
    words = data.split('/')
    idx.append(t)
    freqs = {}
for word in words:
    if word not in freqs:
        freqs[word] = 1
    else:
        freqs[word] += 1

    keys = freqs.keys()  # word
    values = freqs.values()  # frequency

    colm = ['Freq']
    df = pd.DataFrame(data=values, index=keys, columns=colm)
    #total_n = (len(df))
    # print(total_n)
    df2 = (df.loc[df['Freq'] == 1])
    num = (len(df2))  # går galt når jeg appender til listen "number"
    # print(num)

c = ['Unique words in doc']
big_df = pd.DataFrame(data=num, index=idx, columns=c)
big_df

df2.to_csv(r'Data/df2.csv')
