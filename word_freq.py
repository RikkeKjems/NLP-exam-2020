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
# VIRKER næsten

### TANKEN ER AT LAVE EN CSV FIL FOR HVER DOC. I CSV FILEN ER en kolonne med number of unique words pr doc og number of words with length +10 pr doc
### HVER CSV FIL BLIVER SÅ MERGED MED VORES STORE CSV FIL

## lige nu kan jeg lave en kæmpe stor csv fil, men der er en del fejl....


#### DF for each doc 
#### DF with words and their freq + length sorted by freq
path = glob.glob('Data/Final_UTF8_data/ND_data/ND_Tokenfolder/ND_token22.txt')
### brug nedenstående i stort loop
#idx = []
#dat =[]
for t in path:
    data = open(t, "r").read()
    words = data.split('/')
    #idx.append(t)
    freqs = {}
for word in words:
    if word not in freqs:
        freqs[word] = 1
    else:
        freqs[word] += 1
    
#print(freqs)

### TIL FORMMÅL AT ADSKILLE ORD OG FREQ
# split dictionary into keys and values
keys = freqs.keys() #word
values = freqs.values() #frequency
#kv=[keys, values]
#dat.append(kv)

#Create DF
colm = ['Freq']
df = pd.DataFrame(data=values, index=keys, columns=colm)
df 
    
#sort df by freq asceding order  - Ikke relevant mere
#df.sort_values(by='Freq', ascending=True)

#EXTRACT WORDS WITH FREQUENCY OF 1 = UNIQUE WORD - VIRKER 
final_df = df.loc[df['Freq'] == 1]

#CSV fil
final_df.to_csv(r'Data/word_freq_ND22.csv')
#alle ord er i en lang string. det samme gælder for frequency. derfor er filen underlig.
## dette skyldes at freqs er en dict
### Ved ikke hvad jeg skal gøre nu







#%%


 # defines text to be used
 your_file = open('Data/Final_UTF8_data/ND_data/ND_Tokenfolder/ND_token22.txt')
 text = your_file.read
 lines = data.split('/')

 words = []
 ten_l_words = []

 # iterating through "lines" adding each separate word to the "words" array
 for each in lines:
     words += each.split(" ")

 # checking to see if each word in the "words" array is 8 chars long, and if so
 # appending that words to the "eight_l_word" array
 for each in words:
     if len(each) == 10:
         ten_l_words.append(each)

 # finding the number of eight letter words
 number_of_10lwords = len(ten_l_words)

 # displaying results
 print(ten_l_words)
 print("There are "+str(number_of_10lwords)+" ten letter words")
# %%
