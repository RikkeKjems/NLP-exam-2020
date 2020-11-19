# %%
# IMPORT PACKAGES
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
""" 
ALT DETTE KAN FAKTISK SLETTES
# %%
# CLEANING DATA - VIRKER

# Import data
# regex to remove citations/references and quotes
import re

with open('Data/D_data/Test.txt', encoding='utf8', errors='ignore') as f:
    txt = re.sub(
        r"\(\D*\d?\d{4}(?:, s.? [0-9]+.?.?[0-9].?)?(([;])\D*\d{4})*\)|\(([a-zA-Z]+\d\D*\d{4}\))", "", f.read())
    txt = re.sub(r'"[^"]+"', "", txt)
    contents = f.read()
    print(txt)


# %%
# VIRKER TIL AT GEMME NY FIL
# DER SKAL LAVES ET FOR LOOP FOR AT KLARE ALLE FILER
out = open('Data/D_data/Testfile.txt', 'w')
out.write(txt)
out.close()
"""

# %%
# HENTER DATA IND --> CLEANER MED REGEX --> GEMMER I NY MAPPE "Final_D_data" OG "Final_ND_data"
# Cleaning Dyslexia data VIRKER

list_of_files = glob.glob("Data/D_data/*.txt")

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

    f = open(os.path.join("Data/D_data/Final_D_data",
                          os.path.basename(file_name)), "w")

    for line in lst:
        f.write(line)
    f.close()

# %%
# Cleaning Non-Dyslexia data VIRKER

list_of_files = glob.glob("Data/ND_data/*.txt")

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

    f = open(
        os.path.join("Data/ND_data/Final_ND_data",
                     os.path.basename(file_name)), "w"
    )

    for line in lst:
        f.write(line)
    f.close()

# %% Prøver segmentation med for loop VIKER MÅSKE
path = glob.glob("Data/D_data/Testfolder/*.txt")

for file_name in path:
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    if f.mode == "r":
        contents = f.read()
        print(contents)  # Kan undlades, tjekker om vi er inde i filen
    for words in file_name:
        segment = sent_tokenize(contents)
        print(segment)

# %% DET HER ER DET NYESTE MED NY TESTFOLDER MED KUN TO FILER.
path = glob.glob("Data/D_data/Testfolder2/*.txt")

for file_name in path:
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    if f.mode == "r":
        contents = f.read()

    for words in contents:
        segment = sent_tokenize(contents)

print(segment)


# %% Prøver segmentation med for loop VIKER IKKE
list_of_files = glob.glob("Data/D_data/Testfolder/*.txt")

for file_name in list_of_files:
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    lst = []
    for line in f:
        print(sent_tokenize(f))


# %%
# SENTENCE SEGMENTATION - VIRKER

print(sent_tokenize(txt))

# Det nye regex fra Mikkel: = ([.?!)(?![\s]*[\d])

# %%
# TOKENIZATION - VIRKER

tokens = nltk.tokenize.word_tokenize(txt)
print(tokens)

# %%
# TOKEN FREQUENCIES - VIRKER
freq = Counter(tokens)
freq
freq.most_common

# %%
# Lemmatization VIRKER
l = lemmy.load("da")


def lemmatize_sentence(sentence, lemmatizer=l):
    return [lemmatizer.lemmatize("", word) for word in sentence]


lemmas = lemmatize_sentence(txt.split())
print(lemmas)

# %%

nlp = stanza.Pipeline(lang="da", processors="tokenize,pos,lemma")
doc = nlp(tokens)
print(
    *[
        f'word: {word.text+" "}\tlemma: {word.lemma}'
        for sent in doc.sentences
        for word in sent.words
    ],
    sep="\n",
)

# %%

# %%
# nlp = spacy.load("da_core_news_sm")
spacy.load("da_core_news_sm")


# %%
# SpaCy POS tag - VIRKER hvis lemma gør
nlp = spacy.load("da_core_news_sm")
doc = nlp(txt)

for token in doc:
    print(token.text, lemmas, token.pos_, token.is_stop)

# %%
# VIRKER, MEN ER DÅRLIG

blob = txt

text = Text(blob)

text.pos_tags

# %%
tokenlist = tokens

# %%


def postag_stanza(tokenlist):

    # pass

    nlp = stanza.Pipeline(
        lang="da", processors="pos,tokenize,lemma", tokenize_pretokenized=True
    )
    doc = nlp(tokenlist)

    """res = [
        (word.postag_stanza)
        for n_sent, sent in enumerate(doc.sentences)
        for word in sent.words
    ]
"""
    # if return_df:

    # return pd.DataFrame(res)
    # return res


# %%
nlp = postag_stanza(tokenlist=tokenlist)
print(nlp)

# %%


def lemmatize_stanza(txt):
    nlp = stanza.Pipeline(lang="da", processors="tokenize,mwt,pos,lemma")
    doc = nlp(txt)
    print(
        *[
            f'word: {word.text+" "}\tlemma: {word.lemma}\tPOS: {word.POS}'
            for sent in doc.sentences
            for word in sent.words
        ],
        sep="\n",
    )


# %%
print(word.lemmas)


# %%
lemmatizer = lemmy.load("da")
lemmatizer.lemmatize("", "elsker")
# %%

logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.DEBUG)
# %%
nlp = da.load()
