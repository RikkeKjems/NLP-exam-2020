# %%
# IMPORT PACKAGES
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
import os
import glob
import pandas as pd

# %% CLEANING
#Import all data --> clean with regex --> save in a new folder

list_of_files = glob.glob("Data/*.txt")

for file_name in list_of_files:
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    lst = []
    for line in f:
        line.strip()
        line = re.sub(
            r"\(\D*\d?\d{4}(?:, s.? [0-9]+.?.?[0-9].?)?(([;])\D*\d{4})*\)|\(([a-zA-Z]+\d\D*\d{4}\))", #cleaning regex
            "",
            f.read(),
        )
        line = re.sub(r'”[^"]+”', "", line)
        line = re.sub(r'"[^"]+"', "", line)
        lst.append(line)
    f.close()

    f = open(os.path.join("Data/Final_data", os.path.basename(file_name)), "w")

    for line in lst:
        f.write(line)
    f.close()

# %% SEGMENTATION
path = glob.glob("Data/Final_UTF8_data/*.txt")

for file_name in path:
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    seg_lst = []  # empty list
    if f.mode == "r":  # check if the file can be read
        contents = f.read()  # read content in file
        # print(contents)  #print content
    for words in file_name:
        segment = sent_tokenize(contents)  # segmentation function
        seg_lst.append(segment) #save segmentation


# %% TOKENIZATION & REMOVING STOPWORDS
path = glob.glob("Data/Final_UTF8_data/D_data/*.txt")
i = 1
stop = set(stopwords.words("danish"))

for file_name in path:
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    token_lst = []  
    if f.mode == "r":  
        contents = f.read()  
        # print(contents)

    tokens = nltk.tokenize.word_tokenize(contents)  # tokenization function
    print(len(tokens))
    for token in tokens:
        if token not in stop:
            if token not in string.punctuation:
                if token not in string.digits:
                    token_lst.append(nltk.tokenize.word_tokenize(token))
    print(len(token_lst))

    newFile = "D_token" + str(i)  # needs to be changed when running No_dys
    print(newFile)
    token_texts = open(
        f"Data/Final_UTF8_data/D_Data/D_Tokenfolder/{newFile}.txt", "w"
    )
    for token in token_lst:
        token_texts.write(str(token))
        token_texts.write("/")
    token_texts.close()
    i += 1

# %% POS-TAGGING
#Downloading Danish staza for postag
stanza.download('da')

# %% POS TAG PIPELINE
s_nlp = stanza.Pipeline(lang='da',
                        processors='tokenize,pos,lemma',
                        use_gpu=False)

# %%
#defining pos tag function

def postagger(text, stanza_pipeline):
    """
Return lemmas as generator
"""
    doc = stanza_pipeline(text)
    postag = [(word.lemma, word.upos)
              for sent in doc.sentences
              for word in sent.words]
    return postag

# %%
#Looping pos tag through all files
path = glob.glob("Data/Final_UTF8_data/ND_data/ND_Tokenfolder/*.txt")
i = 0
for file_name in path:
    f = open(file_name, "r", encoding="utf8", errors="ignore")
    if f.mode == "r":  
        contents = f.read()
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
        newFile = "D" + str(i)  #Change to ND when running No_Dys
        print(newFile)
        tagged_texts = open(
            f'Data/Final_UTF8_data/New_ND_postagged/tagged_{newFile}.txt', 'w')
        for tagged in pos_tagged:
            tagged_texts.write(str(tagged))
        tagged_texts.close()
        i += 1

### UNIQUE WORDS ###
#%%
# Loop through all files in dir did not work, therefore it was done manually.

# Loop with df as output. Filename and number of unique words for each file.

path = glob.glob("Data/Lemma_data/ND_lemma/ND21_lemma.txt") # Loading one file at a time

idx = []  # empty list

#loop
for t in path:
    data = open(t, "r").read() #reading the file
    words = data.split("/") # Splitting the files by "/"
    idx.append(t) #appending filenmaes to list
    freqs = {} #empty dir
for word in words:
    if word not in freqs:
        freqs[word] = 1
    else:
        freqs[word] += 1

    keys = freqs.keys()  # word
    values = freqs.values()  # frequency

    colm = ["Freq"]
    df = pd.DataFrame(data=values, index=keys, columns=colm) #df with filename and frequency of ech word in file
   
    #total_n = (len(df))
    #print(total_n)

    df2 = df.loc[df["Freq"] == 1] #new df containing only frequency of 1
    num = len(df2)
    #print(num)

#Creating final df
c = ["Unique words in doc"]
ND21 = pd.DataFrame(data=num, index=idx, columns=c)

#Merging all df's. Change ND to D when switching folder
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

# Converting df to csv and saving it in folder 
df_ND.to_csv(r"Data/ND_unique.csv")

#%%
# combining both DF's into one csv file
os.chdir("Data/Unique")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
# combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
# export to csv
combined_csv.to_csv("Unique.csv", index=False, encoding='utf-8-sig')


### MAXIMUM WORD LENGTH ###
# %%
# Run on each file as the loop did not work on an entire folder

#Loading data
path=glob.glob("Data/Lemma_data/ND_lemma/ND21_lemma.txt")

#loop 
for files in path:
    data=open(files, "r").read() #reading file
    words=data.split("/") # splitting content of file by "/"

    for word in words:
        longest=max(words, key=len) #longest word
        length_longest=len(longest) #length of the longest word

        word_length=len(word)
        word_length
# creating df
c=["Length of longest word"]
idx=["ND21"]
ND21=pd.DataFrame(data=length_longest, index=idx, columns=c)

### MOST COMMON WORD LENGTH ###
#%%
#loop
for files in path: #same path as above
    data=open(files, "r").read() #reading file
    words=data.split("/") #splitting content of file by "/"

#loop counting letters in word
    length_counter={}
    for w in words:
        len(w)
        if len(w) in length_counter:
            length_counter[len(w)] += 1
        else:
            length_counter[len(w)]=1

w_len=length_counter.keys() #word length
common=length_counter.values() #most common word length

#creating df
c=['Occurence in text']
df=pd.DataFrame(data=common, index=w_len, columns=c)
yey=df.loc[df['Occurence in text'].idxmax()]

df4=pd.DataFrame(data=yey)
df5=pd.melt(df4)
ND21_21=df5.rename(index={0: 'ND21'}, columns={
                   'variable': 'Most common word Length', 'value': 'Occurence'})
ND21_21

merged_ND21=ND21.merge(ND21_21, left_index=True, right_index=True)
merged_ND21

#%%
# Recalculated as percentage

procent=(289/len(words))*100

df_procent=pd.DataFrame(data=procent, index=['ND21'], columns=['Occurence %'])

#%%
# Merging all df's together to one csv

#Change ND to D when folder is changed in path
final_ND21=merged_ND21.merge(df_procent, left_index=True, right_index=True)
final_ND21

final_ND_df=pd.concat(
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

# df --> csv
final_ND_df.to_csv(r"Data/Length/ND_length.csv")

### CSV FILE WITH ALL LENGTH
# %%
#Specifying dir
os.chdir("Data/Length")
#specifying only csv file
extension="csv"
all_filenames=[i for i in glob.glob("*.{}".format(extension))]

# combine all files in the list
combined_csv=pd.concat([pd.read_csv(f) for f in all_filenames])

# export to csv
combined_csv.to_csv("Length.csv", index=False, encoding="utf-8-sig")

### FINAL DF READY FOR CLASSIFIER ###
# %%
df_l=pd.read_csv('Data/CSV/Length.csv') #length data
df_u=pd.read_csv('Data/CSV/filename_change_unique.csv') #unique data
df_d=pd.read_csv('Data/CSV/filename_change_data.csv') # big data file

dft=pd.merge(df_l, df_u, how='left', on='Unnamed: 0') #merging two at a time
dft

dfg=pd.merge(dft, df_d, how='left', on='Unnamed: 0') #merging next two
dfg

dfg.to_csv(r'Data/CSV/F.csv') #final df converted into csv file. Ready for classifier

### HEREFTER SKAL DER IKKE VÆRE MERE KODE ####
# %% VIRKER
########
# DET HER VIRKER OG FINDER TOTAL NUMBER OF WORDS VI HAR BRUG FOR OG ORDKLASSE FOR HVER FIL

for fileName in glob.iglob(r'Data/All_Tagged_Data/*.txt'):
    data = open(fileName, "r").read()
    words = data.split()
    number_words = len(words)
    Noun_occurrences = data.count("NOUN")
    Verb_occurrences = data.count("VERB")
    Adj_occurrences = data.count("ADJ")
    Pron_occurrences = data.count("PRON")
    Adv_occurrences = data.count("ADV")
    Propn_occurrences = data.count("PROPN")
    usefull_tokens = (Noun_occurrences+Verb_occurrences+Adj_occurrences +
                      Pron_occurrences+Adv_occurrences+Propn_occurrences)
    tokens_we_dont_need = (number_words-usefull_tokens)
    Noun_percentage = Noun_occurrences / usefull_tokens * 100
    Verb_percentage = Verb_occurrences / usefull_tokens * 100
    Adj_percentage = Adj_occurrences / usefull_tokens * 100
    Pron_percentage = Pron_occurrences / usefull_tokens * 100
    Adv_percentage = Adv_occurrences / usefull_tokens * 100
    Propn_percentage = Propn_occurrences / usefull_tokens * 100
    print(fileName, "\n",
          "total amount of words:", number_words, "\n",
          "Usefull tokens:", usefull_tokens, "\n",
          "tokens we don't need:", tokens_we_dont_need, "\n",
          "Noun %:", Noun_percentage, "\n",
          "Verb %:",  Verb_percentage, "\n",
          "Adj %:", Adj_percentage, "\n",
          "Pron %:", Pron_percentage, "\n",
          "Adv %:", Adv_percentage, "\n",
          "Propn %:", Propn_percentage)

##########
# CREATING A DATAFRAME
# %%
path = glob.glob('Data/All_Tagged_Data/*.txt')
ids = []
data_record = []

for fileName in path:
    data = open(fileName, "r").read()
    words = data.split()
    number_words = len(words)
    Noun_occurrences = data.count("NOUN")
    Verb_occurrences = data.count("VERB")
    Adj_occurrences = data.count("ADJ")
    Pron_occurrences = data.count("PRON")
    Adv_occurrences = data.count("ADV")
    Propn_occurrences = data.count("PROPN")
    usefull_tokens = (Noun_occurrences+Verb_occurrences+Adj_occurrences +
                      Pron_occurrences+Adv_occurrences+Propn_occurrences)
    tokens_we_dont_need = (number_words-usefull_tokens)
    Noun_percentage = Noun_occurrences / usefull_tokens * 100
    Verb_percentage = Verb_occurrences / usefull_tokens * 100
    Adj_percentage = Adj_occurrences / usefull_tokens * 100
    Pron_percentage = Pron_occurrences / usefull_tokens * 100
    Adv_percentage = Adv_occurrences / usefull_tokens * 100
    Propn_percentage = Propn_occurrences / usefull_tokens * 100
    # work out the stuff as you do, and instead of printing
    ids.append(fileName)
    record = [number_words, usefull_tokens, tokens_we_dont_need,
              Noun_percentage, Verb_percentage, Adj_percentage, Pron_percentage,
              Adv_percentage, Propn_percentage]
    data_record.append(record)

    cols = ['no_words', 'no_useful_tokens', 'no_useless_tokens',
            'noun %', 'verb %', 'adj %', 'pron %', 'adv %', 'prop %']

#df = pd.DataFrame(data=data_record, index=ids, columns=cols)
# df.to_csv(r'Data/Data.csv')
# Uncomment når DF skal laves. Overrider den eksisterende.
# %%
df = pd.read_csv('Data/CSV/Data.csv', header=0)
# %%
df.insert(1, "D_or_ND", ['D', 'D', 'D', 'ND', 'ND', 'D', 'ND', 'ND', 'D', 'ND', 'ND', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D',
                         'D', 'ND', 'ND', 'ND', 'ND', 'D', 'D', 'ND', 'ND', 'ND', 'ND', 'D', 'D', 'ND', 'ND', 'ND', 'ND', 'D', 'D', 'ND', 'ND', 'ND', 'D'], True)
# %%
df
# %%
# df.to_csv(r'Data/CSV/New_Data.csv')