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


# %% UNIQUE
# VIRKER på en fil
# Loop med df som output. filenavn + antal unique words in each file

path = glob.glob("Data/Lemma_data/ND_lemma/ND21_lemma.txt")

idx = []  # filenames for rows = 22
# number = []  # burde også være 22 et tal for hver doc
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
    # total_n = (len(df))
    # print(total_n)
    df2 = df.loc[df["Freq"] == 1]
    num = len(df2)
    # print(num)

c = ["Unique words in doc"]
ND21 = pd.DataFrame(data=num, index=idx, columns=c)

df_ND = pd.concat(
    [
        ND1,
        ND1,
        ND2,
        ND3,
        ND4,
        ND5,
        ND6,
        ND7,
        ND8,
        ND9,
        ND10,
        ND11,
        ND12,
        ND13,
        ND14,
        ND15,
        ND16,
        ND17,
        ND18,
        ND19,
        ND20,
        ND21,
    ]
)

df_unique = pd.concat(df_D, df_ND)

df_ND.to_csv(r"Data/ND_unique.csv")

#%%
### CSV FIL MED ALLE UNIQUE
import os
import glob
import pandas as pd

os.chdir("Data/Unique")

extension = "csv"
all_filenames = [i for i in glob.glob("*.{}".format(extension))]

# combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
# export to csv
combined_csv.to_csv("Unique.csv", index=False, encoding="utf-8-sig")



# %% #DET HER VIRKER PÅ 1 FIL
### SLETTES?
def main():
    counts = collections.defaultdict(int)
    with open("Data/All_Tagged_Data/tagged_D1.txt", "rt", encoding="utf-8") as file:
        for word in file.read().split():
            counts[len(word)] += 1
    print(file)
    print("length | How often a word of this length occurs")
    for i in sorted(counts.keys()):
        print("%-6d | %d" % (i, counts[i]))


if __name__ == "__main__":
    main()



#%%
## LONGEST WORD IN TEXT ---- VIRKER
## Køres på hver enkelt fil manuelt
path = glob.glob("Data/Lemma_data/ND_lemma/ND21_lemma.txt")

for files in path:
    data = open(files, "r").read()
    words = data.split("/")

    for word in words:
        longest = max(words, key=len)
        length_longest = len(longest)

        word_length = len(word)
        word_length

c = ["Length of longest word"]
idx = ["ND21"]
ND21 = pd.DataFrame(data=length_longest, index=idx, columns=c)

### Most common word length
for files in path:
    data = open(files, "r").read()
    words = data.split("/")
    
    length_counter = {}
    for w in words:
        len(w)
        if len(w) in length_counter:
            length_counter[len(w)] += 1
        else:
            length_counter[len(w)] = 1

w_len = length_counter.keys() 
common = length_counter.values()

c = ['Occurence in text']
df = pd.DataFrame(data = common, index=w_len, columns=c)
yey = df.loc[df['Occurence in text'].idxmax()]

df4 = pd.DataFrame(data=yey)
df5 = pd.melt(df4)
ND21_21 =  df5.rename(index = {0:'ND21'},columns = {'variable':'Most common word Length', 'value':'Occurence'})
ND21_21

merged_ND21= ND21.merge(ND21_21, left_index=True, right_index=True)
merged_ND21

### procent
procent = (289/len(words))*100

df_procent=pd.DataFrame(data=procent, index=['ND21'], columns=['Occurence %'])


### Final df 
final_ND21=merged_ND21.merge(df_procent, left_index=True, right_index=True)
final_ND21

final_ND_df = pd.concat(
    [
        final_ND0,
        final_ND1,
        final_ND2,
        final_ND3,
        final_ND4,
        final_ND5,
        final_ND6,
        final_ND7,
        final_ND8,
        final_ND9,
        final_ND10,
        final_ND11,
        final_ND12,
        final_ND13,
        final_ND14,
        final_ND15,
        final_ND16,
        final_ND17,
        final_ND18,
        final_ND19,
        final_ND20,
        final_ND21,
    ]
)

### df --> csv
final_D_df.to_csv(r"Data/Length/D_length.csv")

#%%
### CSV FIL MED ALLE LENGTH
import os
import glob
import pandas as pd

os.chdir("Data/Length")

extension = "csv"
all_filenames = [i for i in glob.glob("*.{}".format(extension))]

# combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
# export to csv
combined_csv.to_csv("Length.csv", index=False, encoding="utf-8-sig")

#%%
#CSV WITH ALL DATA FOR CLASSIFIER
df_l = pd.read_csv('Data/CSV/Length.csv')
df_u = pd.read_csv('Data/CSV/filename_change_unique.csv')
df_d = pd.read_csv('Data/CSV/filename_change_data.csv')

dft = pd.merge(df_l, df_u, how='left', on='Unnamed: 0')
dft

dfg = pd.merge(dft, df_d, how='left', on='Unnamed: 0')
dfg

dfg.to_csv(r'Data/CSV/F.csv')
#%%
####
### SKAL ALT NEDENSTÅENDE SLTTES?
####

#%%
# Most common word -- VIRKER -- dog ikke relevant
from collections import Counter

path = glob.glob("Data/Lemma_data/ND_lemma/ND21_lemma.txt")

for files in path:
    data = open(files, "r").read()
    words = data.split("/")
    
    word_counter = {}
    for word in words:
        if word in word_counter:
            word_counter[word] += 1
        else:
            word_counter[word] = 1

popular_words = sorted(word_counter, key = word_counter.get, reverse = True)
popular_words[:1]


#length of most common word
word_length =  [len(x) for x in popular_words[:1]]
word_length


#%% 
## VIRKER 
### Most common word length
path = glob.glob("Data/Lemma_data/ND_lemma/ND21_lemma.txt")

for files in path:
    data = open(files, "r").read()
    words = data.split("/")
    
    length_counter = {}
    for w in words:
        len(w)
        if len(w) in length_counter:
            length_counter[len(w)] += 1
        else:
            length_counter[len(w)] = 1

    for v in sorted(length_counter.keys()):
        print("%-6d | %d" % (v, counts[v]))

w_len = length_counter.keys() 
w_len
common = length_counter.values() 
common

c = ['Occurence in text']
df = pd.DataFrame(data = common, index=w_len, columns=c)
df
yey = df.loc[df['Occurence in text'].idxmax()]

df4 = pd.DataFrame(data=yey)
df4
df5 = pd.melt(df4)
df5
df5.rename(columns = {'variable':'Word Lengh', 'value':'Occurence'})



#%%
print (popular_words[:1])
print(word_counter)
print(popular_words)
#word_length = len((popular_words[:1])
#x = popular_words[:1]

#####Length of most common word
### tæller '' og [] med
word_length =  [len(x) for x in popular_words[:1]]
word_length

#%%
### most common word length
path = glob.glob("Data/Lemma_data/ND_lemma/ND21_lemma.txt")
counts = collections.defaultdict(int)
for files in path:
    data = open(files, "r").read()
    words = data.split("/")
    
    for word in files:
            counts[len(word)] += 1
for i in sorted(counts.keys()):
        print("%-6d | %d" % i, counts[i])

print(leng)
    
    word_counter = {}
    for words in files:
        len(words)
        if word in word_counter:
            word_counter[word] += 1
        else:
            word_counter[word] = 1

print(len(words))
popular_words = sorted(word_counter, key = word_counter.get, reverse = True)

print (popular_words[:1])
#%%
##########################
# KAN NEDENSTÅENDE SLETTES?
########################


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
# path = glob.glob("Data//*.txt")

path = glob.glob("Data/Lemma_data/D_lemma/")

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
stanza.download("da")

# %%VIRKER
s_nlp = stanza.Pipeline(lang="da", processors="tokenize,pos,lemma", use_gpu=False)


def lemmatizer(text, stanza_pipeline):
    """
    Return lemmas as generator
    """
    doc = stanza_pipeline(text)
    lemmas = [(word.lemma) for sent in doc.sentences for word in sent.words]
    return lemmas


# LEMMATIZER VIRKER
path = glob.glob("Data/Final_UTF8_data/D_data/D_Tokenfolder/*.txt")
i = 0
for file_name in path:
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    if f.mode == "r":  # tjek om filen kan læses
        contents = f.read()  # læs indholdet i filen
        texts = contents.split("/")
        texts.sort()
        out = []
        for text in texts:
            new = text.replace("[", "")
            new = new.replace("]", "")
            new = new.replace("'", "")
            if new != "":
                out.append(new)
        # print(out)

        lemmas = [lemmatizer(text, s_nlp) for text in out]
        newFile = "D" + str(i)  # kan ændres hvis vi vil have D og ND
        print(newFile)
        tagged_texts = open(f"Data/Lemma_data/D_lemma/{newFile}_lemma.txt", "w")
        for tagged in lemmas:
            tagged_texts.write(str(tagged))
            tagged_texts.write("/")
        tagged_texts.close()
        i += 1
# %%
