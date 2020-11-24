# %%
# IMPORT PACKAGES
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

# %% CLEANING
# HENTER AL DATA IND --> CLEANER MED REGEX --> GEMMER I NY MAPPE "Final_Data"
# Cleaning AL data VIRKER

list_of_files = glob.glob("Data/*.txt")

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
        line = re.sub(r'”[^"]+”', "", line)
        line = re.sub(r'"[^"]+"', "", line)
        lst.append(line)
    f.close()

    f = open(os.path.join("Data/Final_Data", os.path.basename(file_name)), "w")

    for line in lst:
        f.write(line)
    f.close()


# %% SEGMENTATION ORIGINAL
# Segmentation Data VIRKER
# Filer hentes fra Final_UTF8_data da alle filer skal være uft8
path = glob.glob("Data/Final_UTF8_data/*.txt")

for file_name in path:
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    seg_lst = []  # tom liste
    if f.mode == "r":  # tjek om filen kan læses
        contents = f.read()  # læs indholdet i filen
        # print(contents)  #print indholdet - Kan undlades, tjekker om vi er inde i filen
    for words in file_name:
        segment = sent_tokenize(contents)  # segmentation funktion
        # gem segmentation for hver dokument i en liste
        seg_lst.append(segment)

print(seg_lst)  # print liste

# %%
# DET HER SKAL VI VEL HAVE IND ET STED?
# Det nye regex fra Mikkel: = ([.?!)(?![\s]*[\d])

# %%
# Loop Tokenization - VIRKER
# Filer hentes fra Final_UTF8_data da alle filer skal være uft8
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

# %% STOPWORDS VIRKER
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

stop = set(stopwords.words("danish"))
print(stop)

stop_lst = []

for t in tokens:
    if t not in stop:
        stop_lst.append(t)

print(stop_lst)


#%% STOP WORDS FRA SPACY ER DE BEDRE?
# 219 STOP WORDS
len(STOP_WORDS)
print(STOP_WORDS)

# %%
# TOKEN FREQUENCIES - VIRKER
freq = Counter(stop_lst)

tf = freq.most_common
print(tf)
# %% FREQ LIST
import pandas as pd
from pandas import DataFrame

df = pd.DataFrame(tf, columns=["tf"])
df
