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

# %%
# UNIQUE WORDS IN EACH DOCUMENT
diret = glob.glob("Data/Final_UTF8_data/ND_Data/ND_Tokenfolder/ND_token1.txt")
for doc in diret:
    d = open(doc, "r", encoding="utf8", errors="ignore")
    if d.mode == "r":
        content = d.read()
        # print(content)

freq_lst = []
for i in content:
    if i not in freq_lst:  # checking duplicate
        freq_lst.append(i)  # insert value in freq_lst
print(freq_lst)


# åben filens indhold
# count freq of all words
# pair word with freq
print("List\n" + str(content) + "\n")
print("Frequencies\n" + str(wordfreq) + "\n")
print("Pairs\n" + str(list(zip(wordlist, wordfreq))))
# extract words which only appear once = unique words

# %%

txt = "jeg er en fil fuld af unikke ord"
unique = []
for word in txt:
    if word not in unique:
        unique.append(word)

# sort
unique.sort()

# print
print(unique)
# %%

with open("Data/Final_UTF8_data/ND_Data/ND_Tokenfolder/ND_token1.txt", "r") as file:
    lines = file.read().splitlines()
    print(lines)
words = []
for i in lines:
    if i not in words:
        words.append(i)
print(words)
count

uniques = set()
for line in lines:
    uniques |= set(line.split())

print(f"Unique words: {len(uniques)}")

# %%

diret = glob.glob("Data/Final_UTF8_data/ND_Data/ND_Tokenfolder/ND_token1.txt")


def word_count(diret):
    with open(diret) as f:
        return Counter(f.read().split())


print(Counter)

# %%
file = open("Data/Final_UTF8_data/ND_Data", "rt")
data = file.read()
word = data.split()

print(len(word))
# %% VIRKER

data = glob.glob("Data/Final_Data/*.txt")
for file_name in data:
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    wrd_lst = []
    if f.mode == "r":  # tjek om filen kan læses
        contents = f.read()  # læs indholdet i filen
        # print(contents)  #print indholdet - Kan undlades, tjekker om vi er inde i filen

for words in file_name:
    word = contents.split()
    wl = len(word)
    wrd_lst.append(wl)
print(wrd_lst)

# %%
# %% VIRKER (SAMME SOM OVERSTÅENDE)

data = glob.glob("Data/Final_UTF8_data/D_data/D_Tokenfolder/*.txt")
for file_name in data:
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    wrd_lst = []
    if f.mode == "r":  # tjek om filen kan læses
        contents = f.read()  # læs indholdet i filen
        # print(contents)  #print indholdet - Kan undlades, tjekker om vi er inde i filen

for words in file_name:
    word = contents.split()
    wl = len(word)
    wrd_lst.append(wl)
print(wrd_lst)


# for token in tokens:
# if (token not in stop):
# token_lst.append(nltk.tokenize.word_tokenize(token))
# print(len(token_lst))

# %% DETTER VIRKER PÅ UNIQUE WORDS, MEN KUN UNIKKE. KAN IKKE ÆNDRE ANTAL OCCURENCES
with open("Data/Final_UTF8_data/ND_data/ND_Tokenfolder/ND_token6.txt", "r") as file:
    lines = file.read().splitlines()

    uniques = set()
    for line in lines:
        uniques |= set(line.split())

    print(f"Unique words: {len(uniques)}")

#%%
###    WORD FREQ - Pernille arbejder her

## VIRKER
file = open("Data/Final_UTF8_data/D_Data/D5 copy.txt", "rt")
data = file.read()
words = data.split()
number_words = len(words)
print("Total number of words:", number_words)
print(words)

freqs = {}
for word in words:
    if word not in freqs:
        freqs[word] = 1
    else:
        freqs[word] += 1
file.close()
print(freqs)

### CREATE DF WITH WORDS AND FREQ - VIRKER
d = freqs
df = pd.DataFrame(data=d, index=freqs)
df


# FREQ OF POSTAG

# %% DET HER VIRKER, MEN TÆLLER PUNCTUATION MED. DET SKAL FIKSES
file = open("Data/Final_UTF8_data/D_postagged/tagged_D11.txt", "rt")
data = file.read()
words = data.split()
number_words = len(words)
print("Total number of words:", number_words)

Noun_occurrences = data.count("NOUN")
Noun_percentage = Noun_occurrences / number_words * 100

print("Number of nouns :", Noun_occurrences)
print("Percentage of nouns:", Noun_percentage)

Verb_occurrences = data.count("VERB")
Verb_percentage = Verb_occurrences / number_words * 100

print("Number of verbs :", Noun_occurrences)
print("Percentage of verbs:", Verb_percentage)

Adj_occurrences = data.count("ADJ")
Adj_percentage = Adj_occurrences / number_words * 100
print("Number of adj :", Adj_occurrences)
print("Percentage of adj:", Adj_percentage)

# %%
### DF POS TAG - VIRKER
d = [Noun_occurrences, Verb_occurrences, Adj_occurrences]
df = pd.DataFrame(data=d)
df
