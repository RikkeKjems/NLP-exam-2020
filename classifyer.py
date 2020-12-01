# %%
# Libraries for analysis
from sklearn import svm
# %%
import numpy as np
# %%
import pandas as pd

# %%
# Libraries for visuals
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)

# %%
#import data
classifier = pd.read_csv('Data/Data.csv')
classifier.head()
# classifier

# %%
# Trying to add a type column
D_or_ND = ['D', 'D', 'D', 'ND', 'ND', 'D', 'ND', 'ND', 'D', 'ND', 'ND', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D',
           'ND', 'ND', 'ND', 'ND', 'D', 'D', 'ND', 'ND', 'ND', 'ND', 'D', 'D', 'ND', 'ND', 'ND', 'ND', 'D', 'D', 'ND', 'ND', 'ND', 'D']

classifier['Type'] = D_or_ND

# Observe the result
classifier.head()

# classifier.to_csv(r'Data/Data2.csv')


# %%
# prepare data
# plotting two ingredients
sns.lmplot('no_words', 'noun %', data=classifier, hue='Type',
           palette='Set1', fit_reg=False, scatter_kws={"s": 70})

# %%
# Fit the model
# Specify inputs for the model
nowords_noun = classifier[['no_words', 'noun %']]  # .as_matrix()
type_label = np.where(classifier['Type'] == 'D', 0, 1)

# %%
# Fit the SVM model
model = svm.SVC(kernel='linear')
model.fit(nowords_noun, type_label)

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
sns.lmplot('no_words', 'noun %', data=classifier, hue='Type',
           palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=80, facecolors='none')

# %%
# Plot the hyperplane
sns.lmplot('no_words', 'noun %', data=classifier, hue='Type',
           palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')

# PREDICTING NEW CASE
# %%
# plot the point to visually see where the point lies
sns.lmplot('no_words', 'noun %', data=classifier, hue='Type',
           palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(12, 12, 'yo', markersize=9)

# Create a function to guess when a recipe is a muffin
# or a cupcake using the SVM model we created

# %%


def dys_or_nodys(no_words, noun):
    if(model.predict([[no_words, noun]])) == 0:
        print('You\'re looking at a dyslectic text!')
    else:
        print('You\'re looking at non dyslectic text!')


dys_or_nodys(12, 12)

# %%
