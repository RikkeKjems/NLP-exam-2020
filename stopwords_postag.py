# %%
# IMPORT PACKAGES
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
path = glob.glob("Data/ND_Data_For_Postag/*.txt")
i = 0
for file_name in path:
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    if f.mode == "r":  # tjek om filen kan læses
        contents = f.read()  # læs indholdet i filen
        # print(contents)
        texts = contents.replace('\n', '').split(' ')
        texts.sort()
        out = []
        for text in texts:
            if (text != ""):
                out.append(text)
        # print(out)

        pos_tagged = [postagger(text, s_nlp) for text in out]
        newFile = "ND" + str(i)  # kan ændres hvis vi vil have D og ND
        print(newFile)
        tagged_texts = open(f'Data/ND_Tagged_Output/tagged_{newFile}.txt', 'w')
        for tagged in pos_tagged:
            tagged_texts.write(str(tagged))

        i += 1


##########

path = glob.glob("Data/Final_UTF8_data/*.txt")

for file_name in path:
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    token_lst = []  # tom liste
    if f.mode == "r":  # tjek om filen kan læses
        contents = f.read()  # læs indholdet i filen
        # print(contents)  #print indholdet - Kan undlades, tjekker om vi er inde i filen
    for words in file_name:
        tokens = nltk.tokenize.word_tokenize(contents)  # tokenization function
        # gem segmentation for hvert dokument i en liste
        token_lst.append(tokens)

# print(token_lst)  # print liste

# %%
print(token_lst)

# %% STOPWORDS VIRKER

nltk.download("stopwords")

stop = set(stopwords.words("danish"))
print(stop)

stop_lst = []

for t in tokens:
    if t not in stop:
        stop_lst.append(t)

print(stop_lst)


#########
# %%
path = glob.glob("Data/Final_UTF8_data/*.txt")

for file_name in path:
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    token_lst = []  # tom liste
    if f.mode == "r":  # tjek om filen kan læses
        contents = f.read()  # læs indholdet i filen
        # print(contents)  #print indholdet - Kan undlades, tjekker om vi er inde i filen
    for words in file_name:
        tokens = nltk.tokenize.word_tokenize(contents)  # tokenization function
        # gem segmentation for hvert dokument i en liste
        token_lst.append(tokens)


#####
# %%
#text = "Så så han en kylling, og den var flot"

def stopwordser(text):
    for t in tokens:
        if t not in stop:
            stop_lst.append(t)


# %%
stopwordser(text)


# %%


path = glob.glob("Data/ND_Data_For_Postag/*.txt")
i = 0
for file_name in path:
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    if f.mode == "r":  # tjek om filen kan læses
        contents = f.read()  # læs indholdet i filen
        # print(contents)
        out = []
        for t in tokens:
            if t not in stop:
                stop_lst.append(t)
        # print(out)

        pos_tagged = [postagger(text, s_nlp) for text in out]
        newFile = "ND" + str(i)  # kan ændres hvis vi vil have D og ND
        print(newFile)
        tagged_texts = open(f'Data/ND_Tagged_Output/tagged_{newFile}.txt', 'w')
        for tagged in pos_tagged:
            tagged_texts.write(str(tagged))

        i += 1
