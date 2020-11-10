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
# CLEANING DATA - VIRKER IKKE

# Import data
# regex to remove citations/references and quotes
with open('Newfolder/Dys_data_works/D1.txt', encoding="latin6", errors='ignore') as f:
    txt = re.sub(
        r"\(\D*\d?\d{4}(?:, s.? [0-9]+.?.?[0-9].?)?(([;])\D*\d{4})*\)|\(([a-zA-Z]+\d\D*\d{4}\))", "", f.read())
    txt = re.sub(r'"[^"]+"', "", txt)
    contents = f.read()
    print(a)

type(a)

# %%
"""
# TEST TEXT
txt = """
Tryk play for ord
En Intermedial analyse af podwalken Tryk play for mord Analyse og fortolkning
Didaktik dansk Aarhus Universitet


1. Opgavens rammesætning

1. 1 Problemformulering:
Problemfelt: Intermedialitet
Problemstilling: Hvordan kan man gennem analyse forstå og begribe podwalken som en litteraturoplevelser?

1. 2 Indledning:
Lydbøger streames som aldrig før, både af mennesker, der før og sideløbende læser trykte bøger, men også af folk der ikke gør. (Slots-og kulturstyrelsen, 2017)
Aktører som Mofibo og Storytel bliver mere og mere populære, både hos børn, unge og voksne. I denne opgave vil jeg sætte fokus på hvordan podwalken kan analyses,
og derigennem give den et sprog til at diskuterer om den i. Jeg vil sammenligne podwalken med lydbogen, og til sidst komme med en diskussion om hvad mediet betyder for fortællingen.
Podwalks placeres sig tæt op af lydbogen, men også i slipstrømmen af podcasts. Det kræver det samme, en telefon med internetadgang og et par høretelefoner. Også det at kunne gå.
I denne opgave er der taget udgangspunkt i podwalken Tryk play for mord af Danmarks Radio. Podwalken analyseres med Lars Elleströms modaliteter, Iben Have og Birgitte Stougaard Pedersens analysestrategier,
og med Dan Ringgaards stedbaseret læsning. Igennem disse teorier gøres det klart hvilke medie podwalken høres igennem og hvad det fysiske sted betyder for litteraturoplevelsen.
"""
"""

# %%
# SENTENCE SEGMENTATION - VIRKER

print(sent_tokenize(txt))

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
