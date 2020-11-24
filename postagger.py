# %%
# IMPORT PACKAGES
from stanza.pipeline.processor import ProcessorVariant, register_processor_variant
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
import stanza
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


# %%VIRKER OG PRINTER
#postagger(text, s_nlp)

# %% VIRKER!!! SKAL RETTES TIL RIGTIGE FOLDER
# FOR D_DATA
path = glob.glob("Data/D_Data_For_Postag/*.txt")
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
        newFile = "D" + str(i)  # kan ændres hvis vi vil have D og ND
        print(newFile)
        tagged_texts = open(f'Data/D_Tagged_Output/tagged_{newFile}.txt', 'w')
        for tagged in pos_tagged:
            tagged_texts.write(str(tagged))

        i += 1

# print(pos_tagged)

# %% VIRKER!!! SKAL RETTES TIL RIGTIGE FOLDER
# FOR ND_DATA
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
        newFile = "D" + str(i)  # kan ændres hvis vi vil have D og ND
        print(newFile)
        tagged_texts = open(f'Data/ND_Tagged_Output/tagged_{newFile}.txt', 'w')
        for tagged in pos_tagged:
            tagged_texts.write(str(tagged))

        i += 1

# print(pos_tagged)

# %%
text = "Dette er en tekst"


def lemmatizer(text, stanza_pipeline):
    """
    Return lemmas as generator
    """
    doc = stanza_pipeline(text)
    lemma = [(word.lemma)
             for sent in doc.sentences
             for word in sent.words]
    return lemma


# %%
lemmatizer(text, s_nlp)
# %%
