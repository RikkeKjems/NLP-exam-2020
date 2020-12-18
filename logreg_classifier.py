# %%
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sn
plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# %%
df = pd.read_csv('Data/CSV/Length.csv', header=0)
df = df.dropna()
print(df.shape)
print(list(df.columns))

# %%
# Ændrer nogle af navnene på columns
df = df.rename(columns={'Length of longest word': 'max_word_length'})
df = df.rename(columns={'Most common word Length': 'Most_common_word_length'})
df = df.rename(columns={'Occurence %': 'Occurence_perc'})
print(df)
df.to_csv(r'Data/newnames.csv')

# %%
# indsætter D_or_ND kolonne
df.insert(1, "D_or_ND", ['ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND',
                         'ND', 'ND', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'], True)

# %%
data = df
# %%
data['max_word_length'].unique

# %%
# Hvor mange i hver gruppe
data['D_or_ND'].value_counts()

sns.countplot(x='D_or_ND', data=data, palette='hls')
plt.show()
plt.savefig('count_plot')

# %%
# Virker ikke
count_no_sub = len(data[data['D_or_ND'] == 0])  # dette giver 0, hvorfor?
count_sub = len(data[data['D_or_ND'] == 1])  # samme hvad vil vi gerne have?
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription", pct_of_sub*100)

# %%
# try with different variables
data.groupby('D_or_ND').mean()

# %%
# usikker på hvad denne viser
data.groupby('max_word_length').mean()

# %%
# Det her burde kunne bruges.
# Plotter maximum word length og i hvilken gruppe det kommer
%matplotlib inline
pd.crosstab(data.max_word_length, data.D_or_ND).plot(kind='bar')
plt.title('Plot')
plt.xlabel('Maximum word length')
plt.ylabel('D or ND')
plt.savefig('Plot')

# %%
table = pd.crosstab(data.max_word_length, data.D_or_ND)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Maximum word length')
plt.xlabel('Maximum word length')
plt.ylabel('Proportion of Customers')
plt.savefig('Plot2')

# %%
# Kan ikke rigtig bruges
table = pd.crosstab(data.Occurence_perc, data.D_or_ND)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Maximum word length')
plt.xlabel('Maximum word length')
plt.ylabel('Proportion of Customers')
plt.savefig('Plot3')

# %%
# %%
table = pd.crosstab(data.Most_common_word_length, data.D_or_ND)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Most common word length')
plt.xlabel('Most common word length')
plt.ylabel('Proportion of students')
plt.savefig('Plot3')


# %%
# Kan bruges
data.Most_common_word_length.hist()
plt.title('Most common word length')
plt.xlabel('Most common word length')
plt.ylabel('Frequency')
plt.savefig('hist_most_common_word_length')

######
# CLASSIFIER KODE STARTER HER

# %%
# Renamer D og ND til 1 og 0
df["D_or_ND"].replace({"D": "1", "ND": "0"}, inplace=True)
df
# %%
# Setting independent (x) og dependent variables (y)
#X = df[['max_word_length', 'Most_common_word_length', 'Occurence_perc']]
X = df[['max_word_length']]
y = df['D_or_ND']
df
# %%
# Splitting data to train and test set
# test size = 25 %
# train size = 75 %
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)
# %%
# Applying the logistic regression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)
# %%
# Making confusion matrix
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=[
                               'Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
# %%
# Printing accuracy and plotting confusion matrix
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
plt.show()

# %%
print(X_test)
# %%
print(y_pred)
# %%

#
