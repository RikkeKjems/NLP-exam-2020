# %%
import re

# %%
# DET HER ER EN TEST SOM VIRKER OG FJERNER ALLE REFERENCER
a = """Tryk play for "ord"
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
og med Dan Ringgaards stedbaseret læsning. Igennem disse teorier gøres det klart hvilke medie podwalken høres igennem og hvad det fysiske sted betyder for litteraturoplevelsen. """

new_file = re.sub(
    r"\(\D*\d?\d{4}(?:, s.? [0-9]+.?.?[0-9].?)?(([;])\D*\d{4})*\)|\(([a-zA-Z]+\d\D*\d{4}\))", "", a)
# print(new_file)

new_file = re.sub(r'"[^"]+"', "", new_file)
print(new_file)

#new_new_file = re.sub(r'"[^"]+"', "", new_file)
# print(new_new_file)


# %%
# DET HER VIRKER PÅ TEST.TXT OG FJERNER REFERENCER OG CITATER
f = open('Newfolder/Dys_data/Test.txt', 'r')

new_file = re.sub(
    r"\(\D*\d?\d{4}(?:, s.? [0-9]+.?.?[0-9].?)?(([;])\D*\d{4})*\)|\(([a-zA-Z]+\d\D*\d{4}\))", "", f.read())

new_file = re.sub(r'"[^"]+"', "", new_file)
print(new_file)


# %%
# PRØVER MED ANDEN TXT FIL:
# VIRKER!!!
with open('Newfolder/Dys_data_works/D2.txt', encoding="utf8", errors='ignore') as f:
    a = re.sub(
        r"\(\D*\d?\d{4}(?:, s.? [0-9]+.?.?[0-9].?)?(([;])\D*\d{4})*\)|\(([a-zA-Z]+\d\D*\d{4}\))", "", f.read())
    a = re.sub(r'"[^"]+"', "", a)
    contents = f.read()
    print(a)

#%% STOPWORDS
   import nltk
nltk.download('stopwords') 

    from nltk.corpus import stopwords
    words = stopwords.words ('danish')
    len(words)
print(words)



# %%
