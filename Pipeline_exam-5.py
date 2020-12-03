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

# %%%
""" # ALT DET HER SKAL VI IKKE KØRE MERE. DET ER KØRT OG DER ER DANNET FILER
# %% CLEANING
# HENTER AL DATA IND --> CLEANER MED REGEX --> GEMMER I NY MAPPE "Final_Data"
# Cleaning AL data VIRKER

list_of_files = glob.glob("Data/*.txt")

for file_name in list_of_files:
    # print(file_name)  # Dette kan kommenteres ud hvis vi har lyst

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

    f = open(os.path.join("Data/Final_data", os.path.basename(file_name)), "w")

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
        # gem segmentation for hvert dokument i en liste
        seg_lst.append(segment)

# print(seg_lst)  # print liste, kan kommenteres ud

# %%
# DET HER SKAL VI VEL HAVE IND ET STED?
# Det nye regex fra Mikkel: = ([.?!)(?![\s]*[\d])
"""
"""
# RIKKE ARBEJDER HER
# NEW TOKENIZATION FUNCTION
# %%
path = glob.glob("Data/Final_UTF8_data/D_data/*.txt")
i = 1
stop = set(stopwords.words("danish"))

for file_name in path:
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    token_lst = []  # tom liste
    if f.mode == "r":  # tjek om filen kan læses
        contents = f.read()  # læs indholdet i filen
        # print(contents)  #print indholdet - Kan undlades, tjekker om vi er inde i filen

    tokens = nltk.tokenize.word_tokenize(contents)  # tokenization function
    # gem segmentation for hvert dokument i en liste
    print(len(tokens))
    for token in tokens:
        if token not in stop:
            if token not in string.punctuation:
                if token not in string.digits:
                    token_lst.append(nltk.tokenize.word_tokenize(token))
    print(len(token_lst))

    #token_lst = list(filter(lambda token: token not in string.punctuation, token_lst))

    newFile = "D_token" + str(i)  # kan ændres hvis vi vil have D og ND
    print(newFile)
    token_texts = open(
        f"Data/Final_UTF8_data/D_Data/D_Tokenfolder/{newFile}.txt", "w"
    )
    for token in token_lst:
        token_texts.write(str(token))
        token_texts.write("/")
    token_texts.close()
    i += 1

# %%
# RIKKE ARBEJDER OGSÅ LIDT HER
# POSTAGGING
stanza.download('da')

# %%VIRKER
s_nlp = stanza.Pipeline(lang='da',
                        processors='tokenize,pos,lemma',
                        use_gpu=False)

# %%VIRKER


def postagger(text, stanza_pipeline):
    """
Return lemmas as generator
"""
    doc = stanza_pipeline(text)
    postag = [(word.lemma, word.upos)
              for sent in doc.sentences
              for word in sent.words]
    return postag

# %% VIRKER 
path = glob.glob("Data/Final_UTF8_data/ND_data/ND_Tokenfolder/*.txt")
i = 0
for file_name in path:
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    if f.mode == "r":  # tjek om filen kan læses
        contents = f.read()  # læs indholdet i filen
        texts = contents.split('/')
        texts.sort()
        out = []
        for text in texts:
            new = text.replace("[", "")
            new = new.replace("]", "")
            new = new.replace("'", "")
            if (new != ""):
                out.append(new)
        # print(out)

        pos_tagged = [postagger(text, s_nlp) for text in out]
        newFile = "ND" + str(i)  # kan ændres hvis vi vil have D og ND
        print(newFile)
        tagged_texts = open(
            f'Data/Final_UTF8_data/New_ND_postagged/tagged_{newFile}.txt', 'w')
        for tagged in pos_tagged:
            tagged_texts.write(str(tagged))
        tagged_texts.close()
        i += 1
"""

# %%
# Pernille arbejder her - VIRKER IKKE

### TANKEN ER AT LAVE EN CSV FIL FOR HVER DOC. I CSV FILNE ER ALLE ORD SAMT DERES FREQ OG LÆNGDE

#### DF for each doc 
#### DF with words and their freq + length sorted by freq
path = glob.glob('Data⁩/Final_UTF8_data⁩/⁨ND_data⁩/ND_Tokenfolder/*.txt')
idx=[]

for fileN in path:
    f = open(fileN, "r", encoding="utf8", errors="ignore")
    if f.mode == "r": 
        c = f.read()
        w = f.split()
        idx.append(fileN)
print(w)
print(idx)
print(fileN)

# Danner ord og tilhørende freq
freqs = {}
for word in w:
    if word not in freqs:
        freqs[word] = 1
    else:
        freqs[word] += 1
    
#print(freqs)

### TIL FORMÅL AT ADSKILLE ORD OG FREQ
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

#sort df by freq asceding order
df.sort_values(by='Freq', ascending=True)

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

# %% VIRKER
########
# DET HER VIRKER OG FINDER TOTAL NUMBER OF WORDS VI HAR BRUG FOR OG ORDKLASSE FOR HVER FIL

for fileName in glob.iglob(r'Data/All_Tagged_Data/*.txt'):
    data = open(fileName, "r").read()
    words = data.split()
    number_words = len(words)
    Noun_occurrences = data.count("NOUN")
    Verb_occurrences = data.count("VERB")
    Adj_occurrences = data.count("ADJ")
    Pron_occurrences = data.count("PRON")
    Adv_occurrences = data.count("ADV")
    Propn_occurrences = data.count("PROPN")
    usefull_tokens = (Noun_occurrences+Verb_occurrences+Adj_occurrences +
                      Pron_occurrences+Adv_occurrences+Propn_occurrences)
    tokens_we_dont_need = (number_words-usefull_tokens)
    Noun_percentage = Noun_occurrences / usefull_tokens * 100
    Verb_percentage = Verb_occurrences / usefull_tokens * 100
    Adj_percentage = Adj_occurrences / usefull_tokens * 100
    Pron_percentage = Pron_occurrences / usefull_tokens * 100
    Adv_percentage = Adv_occurrences / usefull_tokens * 100
    Propn_percentage = Propn_occurrences / usefull_tokens * 100
    print(fileName, "\n",
          "total amount of words:", number_words, "\n",
          "Usefull tokens:", usefull_tokens, "\n",
          "tokens we don't need:", tokens_we_dont_need, "\n",
          "Noun %:", Noun_percentage, "\n",
          "Verb %:",  Verb_percentage, "\n",
          "Adj %:", Adj_percentage, "\n",
          "Pron %:", Pron_percentage, "\n",
          "Adv %:", Adv_percentage, "\n",
          "Propn %:", Propn_percentage)

##########
# CREATING A DATAFRAME
# %%
path = glob.glob('Data/All_Tagged_Data/*.txt')
ids = []
data_record = []

for fileName in path:
    data = open(fileName, "r").read()
    words = data.split()
    number_words = len(words)
    Noun_occurrences = data.count("NOUN")
    Verb_occurrences = data.count("VERB")
    Adj_occurrences = data.count("ADJ")
    Pron_occurrences = data.count("PRON")
    Adv_occurrences = data.count("ADV")
    Propn_occurrences = data.count("PROPN")
    usefull_tokens = (Noun_occurrences+Verb_occurrences+Adj_occurrences +
                      Pron_occurrences+Adv_occurrences+Propn_occurrences)
    tokens_we_dont_need = (number_words-usefull_tokens)
    Noun_percentage = Noun_occurrences / usefull_tokens * 100
    Verb_percentage = Verb_occurrences / usefull_tokens * 100
    Adj_percentage = Adj_occurrences / usefull_tokens * 100
    Pron_percentage = Pron_occurrences / usefull_tokens * 100
    Adv_percentage = Adv_occurrences / usefull_tokens * 100
    Propn_percentage = Propn_occurrences / usefull_tokens * 100
    # work out the stuff as you do, and instead of printing
    ids.append(fileName)
    record = [number_words, usefull_tokens, tokens_we_dont_need,
              Noun_percentage, Verb_percentage, Adj_percentage, Pron_percentage,
              Adv_percentage, Propn_percentage]
    data_record.append(record)

    cols = ['no_words', 'no_useful_tokens', 'no_useless_tokens',
            'noun %', 'verb %', 'adj %', 'pron %', 'adv %', 'prop %']

#df = pd.DataFrame(data=data_record, index=ids, columns=cols)
#df.to_csv(r'Data/Data.csv')
# Uncomment når DF skal laves. Overrider den eksisterende.
