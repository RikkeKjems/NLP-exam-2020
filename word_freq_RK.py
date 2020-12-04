# %%
# IMPORT PACKAGES
import collections
import numpy as np
from string import digits
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
# VIRKER men med forkert unique
# Loop med df som output. filenavn + antal unique words in each file

path = glob.glob("Data/Final_UTF8_data/ND_data/ND_Tokenfolder/*.txt")

idx = []  # filenames for rows = 22
number = []  # burde også være 22 et tal for hver doc
for t in path:
    data = open(t, "r").read()
    words = data.split("/")
    idx.append(t)
    freqs = {}
for word in words:
    if word not in freqs:
        freqs[word] = 1
    else:
        freqs[word] += 1

    keys = freqs.keys()  # word
    values = freqs.values()  # frequency

    colm = ["Freq"]
    df = pd.DataFrame(data=values, index=keys, columns=colm)
    total_n = len(df)
    print(total_n)
    df2 = df.loc[df["Freq"] == 1]
    num = len(df2)  # går galt når jeg appender til listen "number"
    print(num)

c = ["Unique words in doc"]
big_df = pd.DataFrame(data=num, index=idx, columns=c)
big_df

unique_percentage = num / total_n * 100

# big_df.to_csv(r'Data/Unique_ND.csv')

# %%
# VIRKER men med forkert unique
# Loop med df som output. filenavn + antal unique words in each file

path = glob.glob("Data/Final_UTF8_data/D_data/D_Tokenfolder/*.txt")

idx = []  # filenames for rows = 24
number = []  # burde også være 24 et tal for hver doc
for t in path:
    data = open(t, "r").read()
    words = data.split("/")
    idx.append(t)
    freqs = {}
    for word in words:
        if word not in freqs:
            freqs[word] = 1
        else:
            freqs[word] += 1

        keys = freqs.keys()  # word
        values = freqs.values()  # frequency

        colm = ["Freq"]
        for v in freqs:
            df = pd.DataFrame(data=values, index=keys, columns=colm)
            df2 = df.loc[df["Freq"] == 1]
            num = len(df2)  # går galt når jeg appender til listen "number"
            number.append(num)  # Vil du kigge her?

            c = ["Unique words in doc"]
            big_df = pd.DataFrame(data=num, index=idx, columns=c)
            big_df  # Skulle gerne være forskellige værdier i kolonnen "unique"
# Når dette sker skal den laves til csv og merges med Data.csv

# big_df.to_csv(r'Data/Unique_ND.csv')


# %% #PRINTER EN MASSE OG CHRASHER NÆSTEN INTERACTIVE WINDOW

# defines text to be used
your_file = open("Data/Final_UTF8_data/ND_data/ND_Tokenfolder/ND_token22.txt")
text = your_file.read
lines = data.split("/")

words = []
ten_l_words = []

# iterating through "lines" adding each separate word to the "words" array
for each in lines:
    words += each.split(" ")

    # checking to see if each word in the "words" array is 8 chars long, and if so
    # appending that words to the "eight_l_word" array
    for each in words:
        if len(each) == 10:
            ten_l_words.append(each)

    # finding the number of eight letter words
    number_of_10lwords = len(ten_l_words)

    # displaying results
    print(ten_l_words)
    print("There are " + str(number_of_10lwords) + " ten letter words")
# %%

# RIKKE LEGER HER
# %%% #DET HER VIRKER IKKE
# path = glob.glob("Data/All_Tagged_Data/*.txt")
path = glob.glob("Data/Final_UTF8_data/D_data/D_Tokenfolder/*.txt")
word_length = []
word_occurence = []
row = []

for fileName in path:
    data = open(fileName, "r").read()
    words = data.split("/")
    row.append(fileName)
for word in words:
    length = [len(word)]
    word_length.append(length)
    occur = Counter(words)
    word_occurence.append(occur)
print(word_length)


colu = ["Word Length"]
df5 = pd.DataFrame(data=word_length, index=row, columns=colu)
df5

#########
# %% #PRØVER NOGET, VIRKER IKKE

path = glob.glob("Data/Final_UTF8_data/ND_data/ND_Tokenfolder/*.txt")

for fileName in glob.iglob(r"Data/All_Tagged_Data/*.txt"):
    data = open(fileName, "r").read()
    words = data.split()
    if __name__ == "__main__":
        main()


#####
# %% #DET HER VIRKER PÅ 1 FIL
def main():
    counts = collections.defaultdict(int)
    with open("Data/All_Tagged_Data/tagged_D1.txt", "rt", encoding="utf-8") as file:
        for word in file.read().split():
            counts[len(word)] += 1
    print("length | How often a word of this length occurs")
    for i in sorted(counts.keys()):
        print("%-6d | %d" % (i, counts[i]))


if __name__ == "__main__":
    main()

# %%
