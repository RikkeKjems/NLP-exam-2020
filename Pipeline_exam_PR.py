# %%
# IMPORT PACKAGES
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

# %%
# CLEANING DATA - VIRKER

# Import data
# regex to remove citations/references and quotes
import re

with open("Data/D_data/D1.txt", encoding="utf8", errors="ignore") as f:
    txt = re.sub(
        r"\(\D*\d?\d{4}(?:, s.? [0-9]+.?.?[0-9].?)?(([;])\D*\d{4})*\)|\(([a-zA-Z]+\d\D*\d{4}\))",
        "",
        f.read(),
    )
    txt = re.sub(r'"[^"]+"', "", txt)
    contents = f.read()
    print(txt)

# %%
# SENTENCE SEGMENTATION - VIRKER

seg = sent_tokenize(txt)
print(seg)

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
# Lemmatization
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
# SpaCy POS tag - VIRKER hvis lemma gør
nlp = spacy.load("da_core_news_sm")
doc = nlp(txt)

for token in doc:
    print(token.text, lemmas, token.pos_, token.is_stop)
