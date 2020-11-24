# %%
# IMPORT PACKAGES
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

print(seg_lst)  # print liste

# %%
# DET HER SKAL VI VEL HAVE IND ET STED?
# Det nye regex fra Mikkel: = ([.?!)(?![\s]*[\d])

# FOR D DATA
# %%
# Loop Tokenization - VIRKER
# Filer hentes fra Final_UTF8_data da alle filer skal være uft8
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
        if (token not in stop):
            token_lst.append(nltk.tokenize.word_tokenize(token))
    print(len(token_lst))

    newFile = "D_token" + str(i)  # kan ændres hvis vi vil have D og ND
    print(newFile)
    token_texts = open(
        f'Data/Final_UTF8_data/D_Data/D_Tokenfolder/{newFile}.txt', 'w')
    for token in token_lst:
        token_texts.write(str(token))
        token_texts.write("/")
    token_texts.close()
    i += 1



# FOR ND DATA
# %%
# Loop Tokenization - VIRKER
# Filer hentes fra Final_UTF8_data da alle filer skal være uft8
path = glob.glob("Data/Final_UTF8_data/ND_data/*.txt")
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
        if (token not in stop):
            token_lst.append(nltk.tokenize.word_tokenize(token))
    print(len(token_lst))

    newFile = "ND_token" + str(i)  # kan ændres hvis vi vil have D og ND
    print(newFile)
    token_texts = open(
        f'Data/Final_UTF8_data/ND_Data/ND_Tokenfolder/{newFile}.txt', 'w')
    for token in token_lst:
        token_texts.write(str(token))
        token_texts.write("/")
    token_texts.close()
    i += 1



# %%
# TOKEN FREQUENCIES - VIRKER
freq = Counter(without_stop_lst)

tf = freq.most_common
print(tf)


import collections

c = collections.Counter()
data = ('Data/Final_UTF8_data/ND_Data/ND_Tokenfolder', 'rt')
for file in data:
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    for line in f:
        c.update(line.rstrip().lower())

d = {}

for token in f: 
    d[token] = d.get(token, 0) + 1

word_freq = []
for key, value in d.items():
    word_freq.append((value, key))

word_freq.sort(reverse=True) 
print(word_freq)



print=('Most common:')
    for token, count in c.most_common(3):
        print '%s: %7d' % (token, count)

from collections import Counter
Counter(word_list).most_common()


############################
data = ('Data/Final_UTF8_data/ND_Data/ND_Tokenfolder', 'rt')
for file_name in data:
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    d = []  # tom liste
    if f.mode == "r":  # tjek om filen kan læses
        contents = f.read()  # læs indholdet i filen
        # print(contents)  #print indholdet - Kan undlades, tjekker om vi er inde i filen




# counting number of times each word comes up in list of words
for key in f: 
    d[key] = d.get(key, 0) + 1

sorted(d.items(), key = lambda x: x[1], reverse = True)

#%%
import os
import re
from os.path import join
from collections import Counter, OrderedDict

directory = ('Data/Final_UTF8_data/ND_Data/ND_Tokenfolder', 'rt')
def count_words(directory): # don't use the name dir, it's a builtin function
    """Counts word frequencies in a directory of files.

    Keyword arguments:
    directory -- count_words will search this directory recursively
    ext -- the extension of files that you wish to count

    Returns an OrderedDict, from most to least frequent.
    """

    # Initialize the counter
    word_counter = Counter()

    # Personally I like to break my code into small, simple functions
    # This code could be inline in the loop below,
    # but I think it's a bit clearer this way.
    def update_counter(word_counter, filename):
        '''updates word_counter with all the words in open(filename)'''
        with open(filename, 'r') as f:

    # Using os.walk instead of glob
            for root, dirs, files in os.walk(".txt", topdown=True):
                for fname in files:
                    if fname.endswith(xt):
                        update_counter(word_counter, join(root, fname))
            

    # words_counter.most_common() does exactly the sort you are looking for
        return word_counter.most_common()
# %% FREQ LIST

df = pd.DataFrame(tf, columns=["tf"])
df


#####
# POSTAGGING

# %%
stanza.download('da')

# %%VIRKER
s_nlp = stanza.Pipeline(lang='da',
                        processors='tokenize,pos,lemma',
                        use_gpu=False)


def postagger(text, stanza_pipeline):
    """
    Return lemmas as generator
    """
    doc = stanza_pipeline(text)
    postag = [(word.lemma, word.upos)
              for sent in doc.sentences
              for word in sent.words]
    return postag


# %% VIRKER!!! SKAL RETTES TIL RIGTIGE FOLDER
# FOR D_DATA
path = glob.glob("Data/Testfolder/Tokenfolder/*.txt")
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
        print(out)

        pos_tagged = [postagger(text, s_nlp) for text in out]
        newFile = "D" + str(i)  # kan ændres hvis vi vil have D og ND
        print(newFile)
        tagged_texts = open(
            f'Data/Testfolder/Tokenfolder/tagged/tagged_{newFile}.txt', 'w')
        for tagged in pos_tagged:
            tagged_texts.write(str(tagged))
        tagged_texts.close()
        i += 1

"""
# %% STOPWORDS VIRKER

nltk.download("stopwords")

stop = set(stopwords.words("danish"))
print(stop)

without_stop_lst = []

for t in tokens:
    if t not in stop:
        without_stop_lst.append(t)

print(without_stop_lst)


# %% STOP WORDS FRA SPACY ER DE BEDRE?
# 219 STOP WORDS
len(STOP_WORDS)
print(STOP_WORDS)

"""
