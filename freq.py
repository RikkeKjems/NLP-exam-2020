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
#%%

txt = "jeg er en fil fuld af unikke ord"
unique = []
for word in txt:
    if word not in unique:
        unique.append(word)

# sort
unique.sort()

# print
print(unique)
#%%

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

#%%
from collections import Counter

diret = glob.glob("Data/Final_UTF8_data/ND_Data/ND_Tokenfolder/ND_token1.txt")


def word_count(diret):
    with open(diret) as f:
        return Counter(f.read().split())


print(Counter)

#%%
file = open("Data/Final_UTF8_data/ND_Data", "rt")
data = file.read()
word = data.split()

print(len(word))
# %% VIRKER
import glob

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


# for token in tokens:
# if (token not in stop):
# token_lst.append(nltk.tokenize.word_tokenize(token))
# print(len(token_lst))

# %%
