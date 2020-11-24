# %%
import io
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
import nltk
from nltk.corpus import stopwords
print(stopwords.words('danish'))

# %%
len(stopwords.words('danish'))
print(stop_words)
# %%
# NLTK STOPWORDS = 94


# %% VIRKER PÅ 1 FIL
# word_tokenize accepts a string as an input, not a file.
stop_words = set(stopwords.words('danish'))
file1 = open("Data/UTF8testfolder/Testtext.txt",
             "r", encoding="utf8", errors="ignore")
line = file1.read()  # Use this to read file content as a stream:
words = line.split()
for r in words:
    if not r in stop_words:
        appendFile = open('Data/UTF8testfolder/filteredtext.txt', 'a')
        appendFile.write(" "+r)
        appendFile.close()


# %% ny stopwords til alle filer
# VIRKER OVERHOVEDET IKKE
import nltk
from nltk.corpus import stopwords
print(stopwords.words('danish'))

path = glob.glob("Data/UTF8testfolder/*.txt")
stop_words = set(stopwords.words('danish'))

for file_name in path:
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    seg_lst = []  # tom liste

    if f.mode == "r":  # tjek om filen kan læses
        contents = f.read()  # læs indholdet i filen
        #print(contents)  #print indholdet - Kan undlades, tjekker om vi er inde i filen

    for words in contents:
        #segment = [if word in stop_words else word for word in words]
        segment = [word for word in contents if word not in stopwords.words('danish')]
        # gem segmentation for hver dokument i en liste
        seg_lst.append(segment)

print(seg_lst)  # print liste
