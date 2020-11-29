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
# print(words)

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
df = pd.DataFrame(data=d)
df


# FREQ OF POSTAG

# %% DET HER VIRKER, MEN TÆLLER NUMERIC & PUNCTUATION MED. DET SKAL FIKSES
file = open("Data/Final_UTF8_data/New_D_postagged/tagged_D3.txt", "rt")
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
print("Number of verbs :", Verb_occurrences)
print("Percentage of verbs:", Verb_percentage)

Adj_occurrences = data.count("ADJ")
Adj_percentage = Adj_occurrences / number_words * 100
print("Number of adj :", Adj_occurrences)
print("Percentage of adj:", Adj_percentage)

Pron_occurrences = data.count("PRON")
Pron_percentage = Pron_occurrences / number_words * 100
print("Number of pron :", Pron_occurrences)
print("Percentage of pron:", Pron_percentage)

Adv_occurrences = data.count("ADV")
Adv_percentage = Adv_occurrences / number_words * 100
print("Number of adv :", Adv_occurrences)
print("Percentage of adv:", Adv_percentage)

Propn_occurrences = data.count("PROPN")
Propn_percentage = Propn_occurrences / number_words * 100
print("Number of propn :", Propn_occurrences)
print("Percentage of propn:", Propn_percentage)

usefull_tokens = (Noun_occurrences+Verb_occurrences+Adj_occurrences+Pron_occurrences+Adv_occurrences+Propn_occurrences)
print(usefull_tokens)

tokens_we_need = (number_words-usefull_tokens)
print(tokens_we_need)

####
#%%
# Overstående komprimeret:
file = open("Data/Final_UTF8_data/New_D_postagged/tagged_D3.txt", "rt")
data = file.read()
words = data.split()
number_words = len(words)

Noun_occurrences = data.count("NOUN")
Verb_occurrences = data.count("VERB")
Adj_occurrences = data.count("ADJ")
Pron_occurrences = data.count("PRON")
Adv_occurrences = data.count("Adv")
Probn_occurrences = data.count("PROPN")

usefull_tokens = (Noun_occurrences+Verb_occurrences+Adj_occurrences+Pron_occurrences+Adv_occurrences+Propn_occurrences)
tokens_we_need = (number_words-usefull_tokens)
print(tokens_we_need)

# %%
### DF POS TAG - VIRKER
d = [Noun_occurrences, Verb_occurrences, Adj_occurrences, Adv_occurences, Propn_occurrences]
df = pd.DataFrame(data=d)
df

#%%
### FRA PIPELINE 4
# %%
# ########  VIRKER PÅ TXT FIL DER IKKE ER TOKENIZED
# Open the file in read mode
text = open("Data/Final_UTF8_data/ND_Data/ND2_copy.txt", "r")
print(text)

# Create an empty dictionary
d = dict()

# Iterate over each word in line
for word in text:
    # Check if the word is already in dictionary
    if word in d:
        # Increment count of word by 1
        d[word] = d[word] + 1
    else:
        # Add the word to dictionary with count 1
        d[word] = 1


# Print the contents of dictionary
for key in list(d.keys()):
    print(key, ":", d[key])

#%%
file = open("Data/Final_UTF8_data/ND_Data/ND_Tokenfolder/ND_token2.txt", "rt")
data = file.read()
word = data.split()

print(len(word))

#%%

path = glob.glob("Data/Final_UTF8_data/New_D_postagged/*.txt")


file = open("Data/Final_UTF8_data/New_D_postagged/*.txt", "rt")
data = file.read()
words = data.split()
number_words = len(words)

Noun_occurrences = data.count("NOUN")
Verb_occurrences = data.count("VERB")
Adj_occurrences = data.count("ADJ")
Pron_occurrences = data.count("PRON")
Adv_occurrences = data.count("Adv")
Propn_occurrences = data.count("PROPN")

usefull_tokens = (Noun_occurrences+Verb_occurrences+Adj_occurrences+Pron_occurrences+Adv_occurrences+Propn_occurrences)
tokens_we_dont_need = (number_words-usefull_tokens)

print("tokens we need:", usefull_tokens)
print("tokens we don't need:" ,tokens_we_dont_need)


#%%

Noun_occurrences = data.count("NOUN")
Verb_occurrences = data.count("VERB")
Adj_occurrences = data.count("ADJ")
Pron_occurrences = data.count("PRON")
Adv_occurrences = data.count("Adv")
Propn_occurrences = data.count("PROPN")

#print("total number of words:", number_words)
usefull_tokens = (Noun_occurrences+Verb_occurrences+Adj_occurrences+Pron_occurrences+Adv_occurrences+Propn_occurrences)
#print("tokens we need:", usefull_tokens)
tokens_we_dont_need = (number_words-usefull_tokens)
#print("tokens we don't need:" ,tokens_we_dont_need)
#%%
path = glob.glob("Data/Final_UTF8_data/New_D_postagged/*.txt")

data = file.read()
words = data.split()
number_words = len(words)

f = open(file_name, "r", encoding="utf8", errors="ignore")
token_lst = []  # tom liste
if f.mode == "r":  # tjek om filen kan læses
    contents = f.read()  # læs indholdet i filen

    for filename in path:
        print(filename)       
        print("total number of words:", number_words)
        print("tokens we need:", usefull_tokens)
        print("tokens we don't need:" , tokens_we_dont_need)
    
#%%

Noun_occurrences = data.count("NOUN")
Verb_occurrences = data.count("VERB")
Adj_occurrences = data.count("ADJ")
Pron_occurrences = data.count("PRON")
Adv_occurrences = data.count("Adv")
Propn_occurrences = data.count("PROPN")

#print("total number of words:", number_words)
usefull_tokens = (Noun_occurrences+Verb_occurrences+Adj_occurrences+Pron_occurrences+Adv_occurrences+Propn_occurrences)
#print("tokens we need:", usefull_tokens)
tokens_we_dont_need = (number_words-usefull_tokens)
#print("tokens we don't need:" ,tokens_we_dont_need)

# %%
dico = {}
for i in range(1 ,31): # just to init the dict and avoid checking if index exist...
    dico[i] = 0

with open("Data/Final_UTF8_data/New_D_postagged/tagged_D2.txt", encoding="utf-8") as f: # better to use in that way
    line = f.read()
    for word in line.split(" "):
        dico[len(word)] += 1   

print(dico)
# %%
import collections


def main():
    counts = collections.defaultdict(int)
    with open('Data/Final_UTF8_data/New_D_postagged/tagged_D1.txt', 'rt', encoding='utf-8') as file:
        for word in file.read().split():
            counts[len(word)] += 1
    print('length | How often a word of this length occurs')
    for i in sorted(counts.keys()):
        print('%-6d | %d' % (i, counts[i]))


if __name__ == '__main__':
    main()
# %%
