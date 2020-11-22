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


# %% SEGMENTATION EXAMPLE
path = glob.glob("Data/Final_UTF8_data/*.txt")

for file_name in path:
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    seg_lst = []  # tom liste
    if f.mode == "r":  # tjek om filen kan læses
        contents = f.read()  # læs indholdet i filen
        # print(contents)  #print indholdet - Kan undlades, tjekker om vi er inde i filen
    for words in file_name:
        segment = sent_tokenize.tokenize(contents)  # segmentation funktion
        # gem segmentation for hver dokument i en liste
        seg_lst.append(segment)

print(seg_lst)  # print liste


# %% ny stopwords til alle filer
# VIRKER OVERHOVEDET IKKE
path = glob.glob("Data/UTF8testfolder/*.txt")

for file_name in path:
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    seg_lst = []  # tom liste

    if f.mode == "r":  # tjek om filen kan læses
        contents = f.read()  # læs indholdet i filen
        # print(contents)  #print indholdet - Kan undlades, tjekker om vi er inde i filen

    for words in file_name:
        segment = [if word in stop_words else word for word in words]
        # gem segmentation for hver dokument i en liste
        seg_lst.append(segment)

print(seg_lst)  # print liste

# %%
list_of_files = glob.glob("Data/ND22.txt")

for file_name in list_of_files:
    print(file_name)  # Dette kan kommenteres ud hvis vi har lyst

    # This needs to be done *inside the loop*
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    lst = []
    for line in f:
        line.strip()
        line = re.sub(
            r"\(\D*\d?\d{4}(?:, s.? [0-9]+.?.?[0-9].?)?(([;])\D*\d{4})*\)|\(([a-zA-Z]+\d\D*\d{4}\))",
            "",
            f.read(),
        )
        lst.append(line)
    f.close()

    f = open(os.path.join("Data", os.path.basename(file_name)), "w")

    for line in lst:
        f.write(line)
    f.close()
# %%
