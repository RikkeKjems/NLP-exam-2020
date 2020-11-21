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
        segment = sent_tokenize.tokenize(contents)  # segmentation funktion
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

print(token_lst)  # print liste

# %% VIRKER
# 219 STOP WORDS
len(STOP_WORDS)
print(STOP_WORDS)

# %% VIRKER
# 94 STOP WORDS
nltk.download('stopwords')

words = stopwords.words('danish')
len(words)
print(words)
# %%
# TOKEN FREQUENCIES - VIRKER IKKE
freq = Counter(token_lst)
freq
freq.most_common
# %%
tokenfreq = token_lst.count
len(tokenfreq)
print(tokenfreq)
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
