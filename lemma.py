# %%
# Lemmatization VIRKER
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

# %%
# nlp = spacy.load("da_core_news_sm")
spacy.load("da_core_news_sm")


# %%
# SpaCy POS tag - VIRKER hvis lemma gør
nlp = spacy.load("da_core_news_sm")
doc = nlp(txt)

for token in doc:
    print(token.text, lemmas, token.pos_, token.is_stop)

# %%
# VIRKER, MEN ER DÅRLIG

blob = txt

text = Text(blob)

text.pos_tags

# %%
tokenlist = tokens

# %%


def postag_stanza(tokenlist):

    # pass

    nlp = stanza.Pipeline(
        lang="da", processors="pos,tokenize,lemma", tokenize_pretokenized=True
    )
    doc = nlp(tokenlist)

    """res = [
        (word.postag_stanza)
        for n_sent, sent in enumerate(doc.sentences)
        for word in sent.words
    ]
"""
    # if return_df:

    # return pd.DataFrame(res)
    # return res


# %%
nlp = postag_stanza(tokenlist=tokenlist)
print(nlp)

# %%


def lemmatize_stanza(txt):
    nlp = stanza.Pipeline(lang="da", processors="tokenize,mwt,pos,lemma")
    doc = nlp(txt)
    print(
        *[
            f'word: {word.text+" "}\tlemma: {word.lemma}\tPOS: {word.POS}'
            for sent in doc.sentences
            for word in sent.words
        ],
        sep="\n",
    )


# %%
print(word.lemmas)


# %%
lemmatizer = lemmy.load("da")
lemmatizer.lemmatize("", "elsker")
# %%

logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.DEBUG)
# %%
nlp = da.load()
