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

path = glob.glob('Data/Final_UTF8_data/ND_data/ND_Tokenfolder/*.txt')

idx = []  # filenames for rows = 22
number = []  # burde også være 22 et tal for hver doc
freqs = {}
list_freq = []
for t in path:
    data = open(t, "r").read()
    words = data.split('/')

    for word in words:
        if word not in freqs:
            freqs[word] = 1
        else:
            freqs[word] += 1
    idx.append(t)
    list_freq.append(freqs)
    print(list_freq)
    # print(freqs) alle ord med freq i ND_tokenfolder
        for p in freqs:
            keys = freqs.keys()  # word
            values = freqs.values()  # frequency
            keys

# print(idx, freqs)


with open("Data/Final_UTF8_data/ND_Data/ND_Tokenfolder/ND_token1.txt", "r")
as file:
    lines = file.read().splitlines("/")
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

        colm = ['Freq']
        df = pd.DataFrame(data=values, index=keys, columns=colm)
        df
    # total_n = (len(df))  ikke nødvendigt
    # print(total_n) ikke nødvendigt

    df2 = (df.loc[df['Freq'] == 1])
    df2
    num = len(df2)  # går galt når jeg appender til listen "number"
    number.append(num)
  

c = ['Unique words in doc']
big_df = pd.DataFrame(data=num, index=idx, columns=c)
big_df

unique_percentage = num / total_n * 100

# big_df.to_csv(r'Data/Unique_ND.csv')

# %%
print(df)

# %%
# VIRKER men med forkert unique
# Loop med df som output. filenavn + antal unique words in each file

path = glob.glob('Data/Final_UTF8_data/D_data/D_Tokenfolder/*.txt')

idx = []  # filenames for rows = 24
number = []  # burde også være 24 et tal for hver doc
for t in path:
    data = open(t, "r").read()
    words = data.split('/')
    idx.append(t)
    freqs = {}
for word in words:
    if word not in freqs:
        freqs[word] = 1
    else:
        freqs[word] += 1

    keys = freqs.keys()  # word
    values = freqs.values()  # frequency

colm = ['Freq']
for v in freqs:
    df = pd.DataFrame(data=values, index=keys, columns=colm)
    df2 = (df.loc[df['Freq'] == 1])
    num = (len(df2))  # går galt når jeg appender til listen "number"
    number.append(num)  # Vil du kigge her?

c = ['Unique words in doc']
big_df = pd.DataFrame(data=number, index=idx, columns=c)
big_df  # Skulle gerne være forskellige værdier i kolonnen "unique"
# Når dette sker skal den laves til csv og merges med Data.csv

# big_df.to_csv(r'Data/Unique_ND.csv')


# %%

# defines text to be used
your_file = open('Data/Final_UTF8_data/ND_data/ND_Tokenfolder/ND_token22.txt')
text = your_file.read
lines = data.split('/')

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
    print("There are "+str(number_of_10lwords)+" ten letter words")
# %%

# RIKKE LEGER HER
# %%% #DET HER VIRKER IKKE
# SÅ TÆT PÅ

# path = glob.glob("Data/All_Tagged_Data/*.txt")
path = glob.glob('Data/Final_UTF8_data/D_data/D_Tokenfolder/*.txt')
word_length = []
ten_words = []
row = []

for fileName in path: 
        data = open(fileName, "r").read()
        words = data.split("/")
        row.append(fileName)
for word in words:
    if len(word) > 9:
        word_length.append(word)
        number_of_10lwords = len(word_length)
        ten_words.insert(number_of_10lwords)

colu = ['Words with length of 10 in text']
df5 = pd.DataFrame(data=number_of_10lwords, index=row, columns=colu)
df5

#########
# %% #PRØVER NOGET, VIRKER IKKE

path = glob.glob("Data/Final_UTF8_data/ND_data/ND_Tokenfolder/*.txt")

for fileName in glob.iglob(r'Data/All_Tagged_Data/*.txt'):
    data = open(fileName, "r").read()
    words = data.split()
    if __name__ == '__main__':
        main()



#####
# %% #DET HER VIRKER PÅ 1 FIL
def main():
    counts = collections.defaultdict(int)
    with open('Data/All_Tagged_Data/tagged_D1.txt', 'rt', encoding='utf-8') as file:
        for word in file.read().split():
            counts[len(word)] += 1
    print('length | How often a word of this length occurs')
    for i in sorted(counts.keys()):
        print('%-6d | %d' % (i, counts[i]))

if __name__ == '__main__':
    main()

# %%
# VIRKER PÅ EN HEL MAPPE! 
# SKAL LAVES TIL DF NEDENUNDER

path = glob.glob('Data/Final_UTF8_data/D_data/D_Tokenfolder/*.txt')
i = 0

for file_name in path:
    def main():
        counts = collections.defaultdict(int)
        file = open(file_name, 'rt', encoding='utf-8')
        for word in file.read().split('/'):
            counts[len(word)] += 1
        # print(file_name)
        print('length' , file_name)
        for i in sorted(counts.keys()):
            print(i, counts[i])

    if __name__ == '__main__':
        main()
    i +=1



# %%
# FORSØGER AT LAVE TIL DF MEN VIRKER IKKE
path = glob.glob('Data/Final_UTF8_data/D_data/D_Tokenfolder/*txt')
i = 0
ids = []
data_record = []
record = [0]

for file_name in path:
    def main():
        counts = collections.defaultdict(int)
        data = open(file_name, 'rt', encoding='utf-8')
        for word in data.read().split('/'):
            counts[len(word)] += 1
        # print(file_name)
        print('length' , file_name)
        for i in sorted(counts.keys()):
            print(i, counts[i])
        # record[len(word)] +=1

    if __name__ == '__main__':
        main()
    i +=1

    ids.append(file_name)
    data_record.append(record)

    cols = ['filename'] + [ str(i) for i in range(1,34) ]

    df = pd.DataFrame(data=data_record, index=ids, columns=cols)


# TIPS FRA ROBERT
# %%
record[0]
record[len(word)] +=1
cols = ['filename'] + [ str(i) for i in range(1,34) ]

df = df = pd.DataFrame(data=data_record, index=ids, columns=cols)

###
# %%
df = pd.read_csv('Data/newnames.csv')
# %%
# OMREGNER NUMBER OF UNIQUE WORD TIL PROCENT DEL
df.insert(8, "Unique_occ_perc", 
    ['17.55050505',
    '19.25436527',
    '22.21210742',
    '27.81546811',
    '23.35984095',
    '19.51661631',
    '14.98847041',
    '15.71925754',
    '11.72943601',
    '16.57192829',
    '17.42400501',
    '35.88328076',
    '19.96320147',
    '29.37788018',
    '14.52130096',
    '21.12131464',
    '65.25096525',
    '18.00766284',
    '11.91646192',
    '24.37446074',
    '19.23076923',
    '26.64556962',
    '33.00546448',
    '50.0',
    '58.15602837',
    '64.17112299',
    '57.28813559',
    '28.5472973',
    '31.46292585',
    '54.64480874',
    '16.57179001',
    '26.48',
    '41.32231405',
    '46.85598377',
    '48.20143885',
    '44.44444444',
    '32.51336898',
    '39.11819887',
    '36.49635036',
    '45.64220183',
    '13.90708755',
    '19.49221949',
    '29.44550669',
    '41.85022026',
    '24.54545455',
    '51.5625']
    , True)


# %%
df.to_csv(r'Data/CSV/newnames2.csv')
# %%

# %%
# %%
df.dtypes
# %%
