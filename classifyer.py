# %%
# Libraries for analysis
from sklearn import svm
import numpy as np
import pandas as pd
# Libraries for visuals
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.2)

# %%
# import data
classifier = pd.read_csv("Data/CSV/F.csv")
classifier.head()
# classifier

# %%
# Trying to add a type column
D_or_ND = [
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "ND",
    "D",
    "D",
    "D",
    "D",
    "D",
    "D",
    "D",
    "D",
    "D",
    "D",
    "D",
    "D",
    "D",
    "D",
    "D",
    "D",
    "D",
    "D",
    "D",
    "D",
    "D",
    "D",
    "D",
    "D",
]

classifier["Type"] = D_or_ND

# Observe the result
classifier.head()

# classifier.to_csv(r'Data/Data2.csv')

#%%
#renaming
classifier.insert(1, "Type", ['ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND',
                         'ND', 'ND', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'], True)


print(list(classifier.columns))

#%%
df = classifier.rename(columns={
                        'Unnamed: 0.1': 'File',
                        'Length of longest word': 'max_word_length',
                        'Most common word Length': 'Most_common_word_length',
                        'Occurence %': 'Occurence_perc',
                        'Unique words in doc': 'Unique_words_in_doc',
                        'noun %': 'Noun_perc',
                        'verb %': 'Verb_perc',
                        'adj %': 'Adj_perc',
                        'pron %': 'Pron_perc',
                        'adv %': 'Adv_perc', 
                        'prop %': 'Prop_perc'})

#%%
df["Type"].replace({"D": "1", "ND": "0"}, inplace=True)

#%%
#dropping columns
df = df.drop(columns=['Unnamed: 0' , 'Occurence' , 'no_words', 'no_useful_tokens', 'no_useless_tokens'])
df.shape

# %%
# prepare data
# plotting two ingredients
sns.lmplot(
    "Occurence_perc",
    "Noun_perc",
    data=classifier,
    hue="Type",
    palette="Set1",
    fit_reg=False,
    scatter_kws={"s": 70},
)

# %%
# Fit the model
# Specify inputs for the model
mo = df[
    ['Type', 'max_word_length', 'Most_common_word_length',
    'Occurence_perc', 'Unique_words_in_doc', 'Noun_perc', 'Verb_perc',
    'Adj_perc', 'Pron_perc', 'Adv_perc', 'Prop_perc'
    ]
]

type_label = np.where(df['Type']=='1',0,1)

# %%
# Fit the SVM model
model = svm.SVC(kernel="linear")
model.fit(mo, type_label)

# %%
# Visualise results
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(5, 30)
yy = a * xx - (model.intercept_[0]) / w[1]

# %%
# Plot the parallels to the separating hyperplane
# that pass through the support vectors
b = model.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = model.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# %%
# Look at the margins and support vectors
sns.lmplot('max_word_length','Occurence_perc',
    data=df,
    hue="Type",
    palette="Set1",
    fit_reg=False,
    scatter_kws={"s": 70},
)
#%%
plt.plot(xx, yy, linewidth=2, color="black")
plt.plot(xx, yy_down, "k--")
plt.plot(xx, yy_up, "k--")
plt.scatter(
    model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=80, facecolors="none"
)
# Plot the hyperplane
sns.lmplot(
    "max_word_length",
    "Occurence_perc",
    data=df,
    hue="Type",
    palette="Set1",
    fit_reg=False,
    scatter_kws={"s": 70},
)
plt.plot(xx, yy, linewidth=2, color="black")

# PREDICTING NEW CASE
# %%
# plot the point to visually see where the point lies
sns.lmplot(
    "max_word_length",
    "Occurence_perc",
    data=df,
    hue="Type",
    palette="Set1",
    fit_reg=False,
    scatter_kws={"s": 70},
)
plt.plot(xx, yy, linewidth=2, color="black")
plt.plot(12, 12, "yo", markersize=9)

# Create a function to guess when a recipe is a muffin
# or a cupcake using the SVM model we created

# %%


def dys_or_nodys(max_word_length, Occurence_perc):
    if (model.predict([[max_word_length, Occurence_perc]])) == 0:
        print("You're looking at a dyslectic text!")
    else:
        print("You're looking at non dyslectic text!")


dys_or_nodys(12, 12)

# %%
