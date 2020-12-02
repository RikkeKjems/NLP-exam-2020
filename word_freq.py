# %%
# IMPORT PACKAGES
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
# Pernille arbejder her - VIRKER IKKE

### TANKEN ER AT LAVE EN CSV FIL FOR HVER DOC. I CSV FILNE ER ALLE ORD SAMT DERES FREQ OG LÆNGDE

#### DF for each doc 
#### DF with words and their freq + length sorted by freq
path = glob.glob('Data⁩/Final_UTF8_data⁩/⁨ND_data⁩/ND_Tokenfolder/*.txt')
idx=[]

for filename in path:
    f = open(filename, "r", encoding="utf8", errors="ignore")
    if f.mode == "r": 
        con = f.read()
        word = f.split()
        idx.append(filename)
#print(word)
#print(idx)
#print(filename)

# %%
# Danner ord og tilhørende freq
freqs = {}
for word in w:
    if word not in freqs:
        freqs[word] = 1
    else:
        freqs[word] += 1
    
#print(freqs)

### TIL FORMMÅL AT ADSKILLE ORD OG FREQ
# printing iniial_dictionary
print("intial_dictionary", str(freqs))

# split dictionary into keys and values
keys = freqs.keys() #ord
values = freqs.values() #freq

# printing keys and values separately
print("keys : ", str(keys))  # keys = ord
print("values : ", str(values))  # values = frequency

#Creat DF
colm = ['Freq']
df = pd.DataFrame(data=values, index=keys, columns=colm)
df

#%%
#sort df by freq asceding order
df.sort_values(by='Freq', ascending=True)

#%%
#add length
length = len(w)
 # defines text to be used
 your_file = open("file_location","r+")
 text = your_file.read

 # divides the text into lines and defines some arrays
 lines = text.split("\n")
 words = []
 eight_l_words = []

 # iterating through "lines" adding each separate word to the "words" array
 for each in lines:
     words += each.split(" ")

 # checking to see if each word in the "words" array is 8 chars long, and if so
 # appending that words to the "eight_l_word" array
 for each in words:
     if len(each) == 8:
         eight_l_word.append(each)

 # finding the number of eight letter words
 number_of_8lwords = len(eight_l_words)

 # displaying results
 print(eight_l_words)
 print("There are "+str(number_of_8lwords)+" eight letter words")