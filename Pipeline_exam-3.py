# %%
# IMPORT PACKAGES
import lemmy.pipe
import da_custom_model as da  # name of your spaCy model
from collections import defaultdict
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import re
from collections import Counter
from functools import reduce
from operator import add
import spacy
import nltk

# %%
# CLEANING DATA - VIRKER IKKE

# Import data
# regex to remove citations/references and quotes
rgx_list = "\(\D*\d?\d{4}(?:, s.? [0-9]+.?.?[0-9].?)?(([;])\D*\d{4})*\)|\(([a-zA-Z]+\d\D*\d{4}\))"


def clean_text(rgx_list, text2):
    new_text = text2
    for rgx_match in rgx_list:
        new_text = re.sub(rgx_match, '', new_text)
    return new_text


print(new_text)

# %%
# PREPROCESSING

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
Lydbøger streames som aldrig før, både af mennesker, der før og sideløbende læser trykte bøger, men også af folk der ikke gør. (Slots- og kulturstyrelsen, 2017)
Aktører som Mofibo og Storytel bliver mere og mere populære, både hos børn, unge og voksne. I denne opgave vil jeg sætte fokus på hvordan podwalken kan analyses,
og derigennem give den et sprog til at diskuterer om den i. Jeg vil sammenligne podwalken med lydbogen, og til sidst komme med en diskussion om hvad mediet betyder for fortællingen.
Podwalks placeres sig tæt op af lydbogen, men også i slipstrømmen af podcasts. Det kræver det samme, en telefon med internetadgang og et par høretelefoner. Også det at kunne gå.
I denne opgave er der taget udgangspunkt i podwalken Tryk play for mord af Danmarks Radio. Podwalken analyseres med Lars Elleströms modaliteter, Iben Have og Birgitte Stougaard Pedersens analysestrategier,
og med Dan Ringgaards stedbaseret læsning. Igennem disse teorier gøres det klart hvilke medie podwalken høres igennem og hvad det fysiske sted betyder for litteraturoplevelsen.
"""

# %%
# SENTENCE SEGMENTATION - VIRKER MÅSKE


def sentence_segment(txt):
    """
    txt (str): Text which you want to be segmented into sentences.
    """
    # reg_ex ="(?<!\w\.\w)(?<![A-Z][a-z])[!:?.]\s"
    # reg_ex = "!|\?|\:|\.|\D|\s|\n|"
    # ?!d\.\s?\d?\s[A-Za-z]+\s?[A-Za-z]+)"
    # brug denne
    reg_ex = "(!|\?|\:|\.|\D|\s|\n|?!\d\.\s?\d?\s[A-Za-z]+\s?[A-Za-z]+)"
    # (\d\.\s\d\s[A-Za-z]).*"
    # reg_ex = "^(\d\.(?=\s\d\s[A-Za-z]+).*)"

    # reg_ex = "(r'\W+')"

    # Den her virkede ikke i mit script, derfor har jeg lavet en nu sentence function
    # sentence = [w.replace("\n", "", txt) for w in re.split(reg_ex, txt)]
    # sentence = [re.split(reg_ex, txt)]
    sentence = [re.split(reg_ex, txt)]

    return sentence


# %%
# SENTENCE SEGMENTATION - VIRKER
print(sent_tokenize(txt))


print(sentence)

# %%
# TOKENIZATION - VIRKER

tokens = nltk.tokenize.word_tokenize(txt)
print(tokens)

# %%
# TOKEN FREQUENCIES - VIRKER
fdist = FreqDist(tokens)
fdist

# VIRKER OGSÅ OG MÅSKE ENDDA PÆNERE?
freq = Counter(tokens)
print(freq)
freq
freq.most_common

# %%
# From highest to lowest frequency -Virker ikke

# One solution?
L = [frequencies]
c = Counter(L)
orted(sorted(L), key=c.get, reverse=True)


# solution?
def countWords(token):
    words = {}
    for i in range(len(token)):
        item = token[i]
        count = token.count(item)
        words[item] = count
    return sorted(words.items(), key=lambda item: item[1], reverse=False)


# solution?

dic = defaultdict(int)
for tok in frequencies:
    dic[num] += 1

s_list = sorted(dic, key=dic.__getitem__, reverse=True)

new_list = []
for tok in s_list:
    for rep in range(dic[tok]):
        new_list.append(tok)

print(new_list)

# %%
# Lemmatization SpaCy - ikke testet endnu. Stanza eller SpaCy. Tænker vi må bruge den der kører bedst.
nlp = da.load()

# create an instance of Lemmy's pipeline component for spaCy
pipe = lemmy.pipe.load()

# add the comonent to the spaCy pipeline.
nlp.add_pipe(pipe, after='tagger')

# lemmas can now be accessed using the `._.lemma` attribute on the tokens
nlp("akvariernes")[0]._.lemma

# %%
# Lemmatize Stanza
# does this change the word class of a word? Synge changes to sang
# Lemmatization is dependent on the context, nouns and verbs are sepereated.
# Is lemmatizat


def lemmatize_stanza(tokenlist, return_df=False):
    """
    tokenlist (list): A list of tokens

    lemmatize a tokenlist using stanza

    hint: examine the stanza_example.py script
    """
    pass
    import stanza

    nlp = stanza.Pipeline(
        lang="en", processors="tokenize,lemma", tokenize_pretokenized=True
    )
    doc = nlp(tokenlist)

    res = [
        (word.lemma) for n_sent, sent in enumerate(doc.sentences) for word in sent.words
    ]

    if return_df:
        import pandas as pd

        return pd.DataFrame(res)
    return res


# testing the function:
tl = [
    ["This", "is", "tokenization", "done", "my", "way!"],
    ["Sentence", "split,", "too!"],
    ["Las", "Vegas", "is", "great", "city"],
]

# this works:
lemmatize_stanza(tokenlist=tl, return_df=True)

# %%
# SpaCy POS tag - DETTE VIRKER!!
nlp = spacy.load("da_core_news_sm")
doc = nlp("Tryk play for ord En Intermedial analyse af podwalken Tryk play for mord Analyse og fortolkning Didaktik dansk Aarhus Universitet Opgavens rammesætning ")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.is_stop)

# %%


def postag_stanza(tokenlist):
    """
    tokenlist (list): A list of tokens

    add a part-of-speech (POS) tag to each tokenlist using stanza

    hint: examine the stanza_example.py script
    """
    pass

    nlp = stanza.Pipeline(
        lang="en", processors="pos,tokenize,lemma,mwt", tokenize_pretokenized=True
    )
    doc = nlp(tokenlist)

    res = [
        (word.postag_stanza)
        for n_sent, sent in enumerate(doc.sentences)
        for word in sent.words
    ]

    if return_df:
        import pandas as pd

        return pd.DataFrame(res)
    return res


# testing the function:
tl = [
    ["This", "is", "tokenization", "done", "my", "way!"],
    ["Sentence", "split,", "too!"],
    ["Las", "Vegas", "is", "great", "city"],
]

postag_stanza(tokenlist=tl, return_df=True)


class Text:
    def __init__(self, txt):
        self.sentences = sentence_segment(self.txt)
        self.tokens = tokenize(self.sentences)
        self.token_frequencies = token_frenquencies(self.tokenlist)
        self.lemmatize = lemmatize.stanza(self.tokenlist)
        self.postag = postag_stanza(self.tokenlist)

    def lemmatize(self, method="tokenlist"):
        if method == "tokenlist":
            res = lemmatize_stanza(self.tokenlist)
        else:
            raise ValueError(f"method {method} is not a valid method")
        return res

    def postag(self, method="tokenlist"):
        if method == "tokenlist":
            res = postag_stanza(self.tokenlist)
        else:
            raise ValueError(f"method {method} is not a valid method")
        return res

    def token_f(self, method="tokenlist"):
        if method == "tokenlist":
            res = token_frequencies(self.tokenlist)
        else:
            raise ValueError(f"method {method} is not a valid method")
        return res

# creating dataframe with all above

    def get_df(self):
        """
        returns a dataframe containing the columns:
        token, lemma, pos-tag
        """
        pass

    df = pd.DataFrame(
{
"token": [token_f(),
"lemma": [lemmatize()]
"pos-tag": [postag()]
},
index=[1, 2, 3, 4])

    # add methods to extract tokens, sentences
    # ner, pos-tags etc.
