# %%
from sklearn import tree
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegressionCV
import os  # provides functions for interacting with the operating system
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, explained_variance_score, confusion_matrix, accuracy_score, classification_report, log_loss
from math import sqrt

# %%
testdata = pd.read_csv('Data/CSV/Testdata.csv')

testdata.head(5)

# %%
testdata.insert(1, "D_or_ND", ['ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND',
                               'ND', 'ND', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'], True)


print(list(testdata.columns))

# %%

df = testdata.rename(columns={
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


print(df)

df = df.drop(columns=['Unnamed: 0'])
df


# %% #NOUNS
# create a list of our conditions
conditions = [
    (df['Noun_perc'] > 33) & (df['Noun_perc'] <= 35),
    (df['Noun_perc'] > 35) & (df['Noun_perc'] <= 37),
    (df['Noun_perc'] > 37) & (df['Noun_perc'] <= 39),
    (df['Noun_perc'] > 39) & (df['Noun_perc'] <= 41),
    (df['Noun_perc'] > 41) & (df['Noun_perc'] <= 43),
    (df['Noun_perc'] > 43) & (df['Noun_perc'] <= 45),
    (df['Noun_perc'] > 45) & (df['Noun_perc'] <= 47),
    (df['Noun_perc'] > 47) & (df['Noun_perc'] <= 49),
    (df['Noun_perc'] > 49) & (df['Noun_perc'] <= 51),
    (df['Noun_perc'] > 51) & (df['Noun_perc'] <= 53),
    (df['Noun_perc'] > 53) & (df['Noun_perc'] <= 55),
    (df['Noun_perc'] > 55) & (df['Noun_perc'] <= 57),
    (df['Noun_perc'] > 57) & (df['Noun_perc'] <= 59)
]

# create a list of the values we want to assign for each condition
values = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']

# create a new column and use np.select to assign values to it using our lists as arguments
df['Nouns'] = np.select(conditions, values)

# display updated DataFrame
df.head()

# %% #VERBS
conditions = [
    (df['Verb_perc'] > 16) & (df['Verb_perc'] <= 18),
    (df['Verb_perc'] > 18) & (df['Verb_perc'] <= 20),
    (df['Verb_perc'] > 20) & (df['Verb_perc'] <= 22),
    (df['Verb_perc'] > 22) & (df['Verb_perc'] <= 24),
    (df['Verb_perc'] > 24) & (df['Verb_perc'] <= 26),
    (df['Verb_perc'] > 26) & (df['Verb_perc'] <= 28),
    (df['Verb_perc'] > 28) & (df['Verb_perc'] <= 30)
]

# create a list of the values we want to assign for each condition
values = ['1', '2', '3', '4', '5', '6', '7']

# create a new column and use np.select to assign values to it using our lists as arguments
df['Verbs'] = np.select(conditions, values)

# display updated DataFrame
df.head()

# %% #ADJS
conditions = [
    (df['Adj_perc'] > 5) & (df['Adj_perc'] <= 7),
    (df['Adj_perc'] > 7) & (df['Adj_perc'] <= 9),
    (df['Adj_perc'] > 9) & (df['Adj_perc'] <= 11),
    (df['Adj_perc'] > 11) & (df['Adj_perc'] <= 13),
    (df['Adj_perc'] > 13) & (df['Adj_perc'] <= 15),
    (df['Adj_perc'] > 15) & (df['Adj_perc'] <= 17),
]

# create a list of the values we want to assign for each condition
values = ['1', '2', '3', '4', '5', '6']

# create a new column and use np.select to assign values to it using our lists as arguments
df['Adjs'] = np.select(conditions, values)

# display updated DataFrame
df.head()

# %% #PRONS
conditions = [
    (df['Pron_perc'] > 1) & (df['Pron_perc'] <= 2),
    (df['Pron_perc'] > 2) & (df['Pron_perc'] <= 3),
    (df['Pron_perc'] > 3) & (df['Pron_perc'] <= 4),
    (df['Pron_perc'] > 4) & (df['Pron_perc'] <= 5),
    (df['Pron_perc'] > 5) & (df['Pron_perc'] <= 6),
    (df['Pron_perc'] > 6) & (df['Pron_perc'] <= 7),
]

# create a list of the values we want to assign for each condition
values = ['1', '2', '3', '4', '5', '6']

# create a new column and use np.select to assign values to it using our lists as arguments
df['Prons'] = np.select(conditions, values)

# display updated DataFrame
df.head()

# %% #ADVS
conditions = [
    (df['Adv_perc'] > 7) & (df['Adv_perc'] <= 9),
    (df['Adv_perc'] > 9) & (df['Adv_perc'] <= 11),
    (df['Adv_perc'] > 11) & (df['Adv_perc'] <= 13),
    (df['Adv_perc'] > 13) & (df['Adv_perc'] <= 15),
    (df['Adv_perc'] > 15) & (df['Adv_perc'] <= 17),
    (df['Adv_perc'] > 17) & (df['Adv_perc'] <= 19),
]

# create a list of the values we want to assign for each condition
values = ['1', '2', '3', '4', '5', '6']

# create a new column and use np.select to assign values to it using our lists as arguments
df['Advs'] = np.select(conditions, values)

# display updated DataFrame
df.head()

# %% #PROPS
conditions = [
    (df['Prop_perc'] > 0) & (df['Prop_perc'] <= 1),
    (df['Prop_perc'] > 1) & (df['Prop_perc'] <= 2),
    (df['Prop_perc'] > 2) & (df['Prop_perc'] <= 3),
    (df['Prop_perc'] > 3) & (df['Prop_perc'] <= 4),
    (df['Prop_perc'] > 4) & (df['Prop_perc'] <= 5),
    (df['Prop_perc'] > 5) & (df['Prop_perc'] <= 6),
    (df['Prop_perc'] > 6) & (df['Prop_perc'] <= 7),
    (df['Prop_perc'] > 7) & (df['Prop_perc'] <= 8),
    (df['Prop_perc'] > 12)
]

# create a list of the values we want to assign for each condition
values = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

# create a new column and use np.select to assign values to it using our lists as arguments
df['Props'] = np.select(conditions, values)

# display updated DataFrame
df.head()

# %% #MAX_WORD_LENGTH_OCC
conditions = [
    (df['Occurence_perc'] > 8) & (df['Occurence_perc'] <= 10),
    (df['Occurence_perc'] > 10) & (df['Occurence_perc'] <= 12),
    (df['Occurence_perc'] > 12) & (df['Occurence_perc'] <= 14),
    (df['Occurence_perc'] > 14) & (df['Occurence_perc'] <= 16),
    (df['Occurence_perc'] > 16) & (df['Occurence_perc'] <= 18),
    (df['Occurence_perc'] > 18) & (df['Occurence_perc'] <= 20),
    (df['Occurence_perc'] > 20)
]

# create a list of the values we want to assign for each condition
values = ['1', '2', '3', '4', '5', '6', '7']

# create a new column and use np.select to assign values to it using our lists as arguments
df['Max_word_length_occ'] = np.select(conditions, values)

# display updated DataFrame
df.head()


# %%
df.to_csv(r'Data/CSV/TryData.csv')


# %%
data = pd.read_csv('Data/CSV/TryData.csv')
data = data.drop(columns=['Unnamed: 0', 'Noun_perc', 'Verb_perc', 'Adj_perc', 'Pron_perc', 'Adv_perc',
                          'Prop_perc', 'Occurence', 'no_words', 'Unique_words_in_doc', 'no_useful_tokens', 'no_useless_tokens'])
data


# %%
data.to_csv(r'Data/CSV/Rikkedata.csv')
# %%
df = pd.read_csv('Data/CSV/Rikkedata.csv')
df = df.drop(columns='Occurence_perc')

# %%
df["File"].replace({
    'ND0': '0.0',
    'ND1': '0.1',
    'ND2': '0.2',
    'ND3': '0.3',
    'ND4': '0.4',
    'ND5': '0.5',
    'ND6': '0.6',
    'ND7': '0.7',
    'ND8': '0.8',
    'ND9': '0.9',
    'ND10': '0.10',
    'ND11': '0.11',
    'ND12': '0.12',
    'ND13': '0.13',
    'ND14': '0.14',
    'ND15': '0.15',
    'ND16': '0.16',
    'ND17': '0.17',
    'ND18': '0.18',
    'ND19': '0.19',
    'ND20': '0.20',
    'ND21': '0.21',
    'D0': '1.0',
    'D1': '1.1',
    'D2': '1.2',
    'D3': '1.3',
    'D4': '1.4',
    'D5': '1.5',
    'D6': '1.6',
    'D7': '1.7',
    'D8': '1.8',
    'D9': '1.9',
    'D10': '1.10',
    'D11': '1.11',
    'D12': '1.12',
    'D13': '1.13',
    'D14': '1.14',
    'D15': '1.15',
    'D16': '1.16',
    'D17': '1.17',
    'D18': '1.18',
    'D19': '1.19',
    'D20': '1.20',
    'D21': '1.21',
    'D22': '1.22',
    'D23': '1.23'
}, inplace=True)


df['D_or_ND'][df['D_or_ND'] == 'D'] = 1
df['D_or_ND'][df['D_or_ND'] == 'ND'] = 0

# %%
# %%
# Split the data into X & y

df = df.drop(columns=['File'])

X = df.drop('D_or_ND', axis=1)
y = df['D_or_ND']

X
y

y = y.astype(int)

print(X.shape)
print(y.shape)

# %%
# Run a Tree-based estimators (i.e. decision trees & random forests)

dt = tree.DecisionTreeClassifier()
dt.fit(X, y)

# If you want to learn how Decesion Trees work, read here: https://www.datacamp.com/community/tutorials/decision-tree-classification-python
# Official Doc: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

# %%
# Running Feature Importance

fi_col = []
fi = []

for i, column in enumerate(df.drop('D_or_ND', axis=1)):
    print('The feature importance for {} is : {}'.format(
        column, dt.feature_importances_[i]))

    fi_col.append(column)
    fi.append(dt.feature_importances_[i])
# %%

#########################################

# %%
