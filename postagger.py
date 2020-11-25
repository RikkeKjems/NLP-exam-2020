# %%
# IMPORT PACKAGES
import collections
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
path = glob.glob("Data/Final_UTF8_data/D_data/D_Tokenfolder/*.txt")
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
        tagged_texts = open(
            f'Data/Final_UTF8_data/D_postagged/tagged_{newFile}.txt', 'w')
        for tagged in pos_tagged:
            tagged_texts.write(str(tagged))
        tagged_texts.close()
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
        newFile = "ND" + str(i)  # kan ændres hvis vi vil have D og ND
        print(newFile)
        tagged_texts = open(f'Data/ND_Tagged_Output/tagged_{newFile}.txt', 'w')
        for tagged in pos_tagged:
            tagged_texts.write(str(tagged))

        i += 1

# print(pos_tagged)


# PRØVER FOR LEMMA
# %%

# %%VIRKER
d_nlp = stanza.Pipeline(lang='da',
                        processors='tokenize,pos,lemma',
                        use_gpu=False)

# %%
#text = "Dette er en tekst"


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
#lemmatizer(text, s_nlp)
# %%
# PRØVER STADIG FOR LEMMA
# %% virker måske
# FOR ND_DATA
path = glob.glob("Data/Testfolder/*.txt")
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

        pos_tagged = [lemmatizer(text, d_nlp) for text in out]
        newFile = "D" + str(i)  # kan ændres hvis vi vil have D og ND
        print(newFile)
        tagged_texts = open(
            f'Data/Testfolder/Tagged_test/tagged_{newFile}.txt', 'w')
        for tagged in pos_tagged:
            tagged_texts.write(str(tagged))

        i += 1


# %%
dico = {}
# just to init the dict and avoid checking if index exist...
for i in range(1, 31):
    dico[i] = 0

with open("Data/Testfolder/D1 copy.txt", encoding="utf-8") as f:  # better to use in that way
    line = f.read()
    for word in line.split(" "):
        dico[len(word)] += 1

print(dico)
# %%


def main():
    counts = collections.defaultdict(int)
    with open('Data/Testfolder/ND4 copy.txt', 'rt', encoding='utf-8') as file:
        for word in file.read().split():
            counts[len(word)] += 1
    print('length | How often a word of this length occurs')
    for i in sorted(counts.keys()):
        print('%-6d | %d' % (i, counts[i]))


if __name__ == '__main__':
    main()


# VI PRØVER LIGE MED DET HER
# %%
path = glob.glob("Data/Final_UTF8_data/D_data/D_Tokenfolder/*.txt")
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
        newFile = "D" + str(i)  # kan ændres hvis vi vil have D og ND
        print(newFile)
        tagged_texts = open(
            f'Data/Final_UTF8_data/D_postagged/tagged_{newFile}.txt', 'w')
        for tagged in pos_tagged:
            tagged_texts.write(str(tagged))
        tagged_texts.close()
        i += 1

# %%
