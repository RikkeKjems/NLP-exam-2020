# %%
# IMPORT PACKAGES
import lemmy.pipe
import regex
import logging
from polyglot.text import Text
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

# %%
# CLEANING DATA - VIRKER

# Import data
# regex to remove citations/references and quotes
import re

with open('Data/D_data/D1.txt', encoding='utf8', errors='ignore') as f:
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

# %%
# SENTENCE SEGMENTATION - VIRKER

print(sent_tokenize(txt))

# Det nye regex fra Mikkel: = ([.?!)(?![\s]*[\d])

# %%
# TOKENIZATION - VIRKER

tokens = nltk.tokenize.word_tokenize(txt)
print(tokens)

# From list to string
str_token = print(",".join(str(x) for x in tokens))
# %%
# TOKEN FREQUENCIES - VIRKER
freq = Counter(tokens)
freq
freq.most_common

# %%
# Lemmatization SpaCy - VIRKER men kun på ord ikke text

# Create an instance of the standalone lemmatizer.
lemmatizer = lemmy.load("da")

# Find lemma for the word 'akvariernes'. First argument is an empty POS tag.
lemma = lemmatizer.lemmatize("", str_token)
lemma

len(str_token)

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
# SpaCy POS tag - VIRKER hvis lemma gør
nlp = spacy.load("da_core_news_sm")
doc = nlp(txt)

for token in doc:
    print(token.text, lemma, token.pos_, token.is_stop)

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
        lang="da", processors="pos,tokenize,lemma,mwt", tokenize_pretokenized=True
    )
    doc = nlp(tokenslist)

    res = [
        (word.postag_stanza)
        for n_sent, sent in enumerate(doc.sentences)
        for word in sent.words
    ]

    # if return_df:

    # return pd.DataFrame(res)
    return res


# %%
postag_stanza(tokenlist=tokenlist)


# %%
def lemmatize_stanza(txt):
    nlp = stanza.Pipeline(lang='da', processors='tokenize,mwt,pos,lemma')
    doc = nlp(txt)
    print(*[f'word: {word.text+" "}\tlemma: {word.lemma}\tPOS: {word.POS}'
            for sent in doc.sentences for word in sent.words], sep='\n')


# %%
print(word.lemmas)


# %%
lemmatizer = lemmy.load("da")
lemmatizer.lemmatize("", "elsker")
# %%

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)
# %%
nlp = da.load()
