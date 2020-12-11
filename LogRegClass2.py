##### IMPORTING / INSTALLING LIBRARIES ####
# %%
# Packages / libraries
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

%matplotlib inline

# To change scientific numbers to float
np.set_printoptions(formatter={'float_kind': '{:f}'.format})

# Increases the size of sns plots
sns.set(rc={'figure.figsize': (12, 10)})

### LOADING THE RAW DATA ###
# %%
# Loading the data
raw_data = pd.read_csv('Data/CSV/F.csv')

# print the shape
print(raw_data.shape)

# runs the first 5 rows
raw_data.head(5)

#%%
#### EDITING RAW DATA
raw_data.insert(1, "D_or_ND", ['ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND',
                         'ND', 'ND', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'], True)


print(list(raw_data.columns))


df = raw_data.rename(columns={
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

raw_data

print(df)

df = df.drop(columns=['Unnamed: 0'])
df

df.to_csv(r'Data/newnames.csv')

#%%
#### New data with updated columns
raw_data = pd.read_csv('Data/newnames.csv')

# print the shape
print(raw_data.shape)

# runs the first 5 rows
raw_data.head(5)

raw_data.drop(columns=['Unnamed: 0'])
# %%
#### DATA PREPROCCESSING #####
#### EXPLORATORY DATA ANALYSIS ####
""" 
#Det her burde vi ikke skulle bruge, da vi ikke har nogle null felter
#%%
# Checking for null values
raw_data.isnull().sum()

#%%
# Visualize the NULL observations
raw_data[raw_data['Employment History'].isnull()]

#%%
# Deleting the NULL values
raw_data = raw_data.dropna(subset = ['Employment History'])

# Printing the shape
print(raw_data.shape)

# Visualize the NULL observations
raw_data.isnull().sum()
"""

# %%
# Investigate all the elements whithin each Feature
### Hvad er ideen bag dette?

for column in raw_data:
    unique_values = np.unique(raw_data[column])
    nr_values = len(unique_values)
    if nr_values <= 10:
        print("The number of values for feature {} is: {} -- {}".format(column,
                                                                        nr_values, unique_values))
    else:
        print("The number of values for feature {} is: {}".format(column, nr_values))


# %%
# Visualize the data using seaborn Pairplots
%matplotlib inline
g = sns.pairplot(raw_data)

# Notes: Do not run this on a big dataset. Filter the columns first


#### DATA CLEANING ####
# %%
# Deleting the outlier
# hard to see if we have any outliers

raw_data = raw_data[raw_data['max_word_length'] < 60]

raw_data.shape

# %%
# Visualize the data using seaborn Pairplots
g = sns.pairplot(raw_data, hue='D_or_ND')

# %%
# Investigating the distr of y
sns.countplot(x='D_or_ND', data=raw_data, palette='Set3')

# %%

raw_data.columns

# %%
# Looping through all the features by our y variable - see if there is relationship

features = ['File', 'max_word_length', 'Most_common_word_length',
            'Occurence', 'Occurence_perc',
            'Unique_words_in_doc', 'no_words', 'no_useful_tokens',
       'no_useless_tokens', 'Noun_perc', 'Verb_perc', 'Adj_perc', 'Pron_perc',
       'Adv_perc', 'Prop_perc']

for f in features:
    sns.countplot(x=f, data=raw_data, palette='Set3', hue='D_or_ND')
    plt.show()

# %%
raw_data.head()
raw_data

# %%
# Making categorical variables into numeric representation
new_raw_data=raw_data
new_raw_data



print(raw_data.shape)
# print the shape
print(new_raw_data.shape)


# Creating a new 0-1 y variable
new_raw_data["File"].replace({
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
    'ND13':'0.13',
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
    'D10':'1.10',
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


new_raw_data['D_or_ND'][new_raw_data['D_or_ND'] == 'D'] = 1
new_raw_data['D_or_ND'][new_raw_data['D_or_ND'] == 'ND'] = 0

# Visualizing the data
new_raw_data

# Notes:
# We do not need to normalize / standardize the data in Logistic Regression due to the logistic function (0 or 1)
# Once a value crosses the decision boundary (0.5 threshold), it saturates
# After the 0.5 or before, there is no additional value to be added from smaller or larger values

#### FEATURE SELECTION ###
# In this example, we do not have many variables so we might use all of the data but in some cases, you have thousands of variables and you will need to filter them in order to save computational time
# Steps of Running Feature Importance:
# -Split the data into X & y
# -Run a Tree-based estimators (i.e. decision trees & random forests)
# -Run Feature Importance

# %%
# Split the data into X & y

X = raw_data.drop('D_or_ND', axis=1).values
y = raw_data['D_or_ND']

y = y.astype(int)

print(X.shape)
print(y.shape)

# %%
# Run a Tree-based estimators (i.e. decision trees & random forests)

dt = DecisionTreeClassifier(random_state=15, criterion = 'entropy', max_depth = 10)
dt.fit(X, y)

# If you want to learn how Decesion Trees work, read here: https://www.datacamp.com/community/tutorials/decision-tree-classification-python
# Official Doc: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

# %%
# Running Feature Importance

fi_col = []
fi = []

for i, column in enumerate(new_raw_data.drop('D_or_ND', axis=1)):
    print('The feature importance for {} is : {}'.format(
        column, dt.feature_importances_[i]))

    fi_col.append(column)
    fi.append(dt.feature_importances_[i])

# %%
# Creating a Dataframe
fi_col
fi

fi_df = zip(fi_col, fi)
fi_df = pd.DataFrame(fi_df, columns=['Feature', 'Feature Importance'])
fi_df


# Ordering the data
fi_df = fi_df.sort_values('Feature Importance', ascending=False).reset_index()

# Creating columns to keep
columns_to_keep = fi_df['Feature'][0:40]

fi_df

# SPLITTING THE RAW DATA - HOLD-OUT VALIDATION
# %%
# Print the shapes

print(new_raw_data.shape)
print(new_raw_data[columns_to_keep].shape)

# new_raw_data = new_raw_data[columns_to_keep]

# %%
# Split the data into X & y

X = new_raw_data[columns_to_keep].values
X

y = new_raw_data['Good Loan']
y = y.astype(int)
y

print(X.shape)
print(y.shape)

# %%
# Hold-out validation

# first one
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=15)

# Second one
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, train_size=0.9, test_size=0.1, random_state=15)

print(X_train.shape)
print(X_test.shape)
print(X_valid.shape)

print(y_train.shape)
print(y_test.shape)
print(y_valid.shape)

# Official Doc: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

# %%
# Investigating the distr of all ys

ax = sns.countplot(x=y_valid, palette="Set3")


#### RUNNING LOGISTIC REGRESSION ####
# %%
# Training my model

log_reg = LogisticRegression(random_state=10, solver='lbfgs')

log_reg.fit(X_train, y_train)

# SKLearn doc: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# %%
# Methods we can use in Logistic

# predict - Predict class labels for samples in X
log_reg.predict(X_train)
y_pred = log_reg.predict(X_train)

# predict_proba - Probability estimates
pred_proba = log_reg.predict_proba(X_train)

# coef_ - Coefficient of the features in the decision function
log_reg.coef_

# score- Returns the mean accuracy on the given test data and labels - below


#### EVALUATING THE MODEL ####
# %%
# Accuracy on Train
print("The Training Accuracy is: ", log_reg.score(X_train, y_train))

# Accuracy on Test
print("The Testing Accuracy is: ", log_reg.score(X_test, y_test))


# Classification Report
print(classification_report(y_train, y_pred))

# %%
# Confusion Matrix function


def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, cmap="YlGnBu", xticklabels=classes, yticklabels=classes,
                    vmin=0., vmax=1., annot=True, annot_kws={'size': 50})
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# %%
# Visualizing cm


cm = confusion_matrix(y_train, y_pred)
cm_norm = cm / cm.sum(axis=1).reshape(-1, 1)

plot_confusion_matrix(cm_norm, classes=log_reg.classes_,
                      title='Confusion matrix')

# %%
log_reg.classes_

# %%
cm.sum(axis=1)
cm_norm

# %%
cm

# %%
cm.sum(axis=0)

# %%
np.diag(cm)

# %%
# Calculating False Positives (FP), False Negatives (FN), True Positives (TP) & True Negatives (TN)

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)


# Sensitivity, hit rate, recall, or true positive rate
TPR = TP / (TP + FN)
print("The True Positive Rate is:", TPR)

# Precision or positive predictive value
PPV = TP / (TP + FP)
print("The Precision is:", PPV)

# False positive rate or False alarm rate
FPR = FP / (FP + TN)
print("The False positive rate is:", FPR)


# False negative rate or Miss Rate
FNR = FN / (FN + TP)
print("The False Negative Rate is: ", FNR)


# Total averages :
print("")
print("The average TPR is:", TPR.sum()/2)
print("The average Precision is:", PPV.sum()/2)
print("The average False positive rate is:", FPR.sum()/2)
print("The average False Negative Rate is:", FNR.sum()/2)

# Logarithmic loss - or Log Loss - or cross-entropy loss
# Log Loss is an error metric
# This is the loss function used in (multinomial) logistic regression and extensions of it such as neural networks, defined as the negative log-likelihood of the true labels given a probabilistic classifier’s predictions.
# Why it's important? For example, imagine having 2 models / classifiers that both predict one observation correctly (Good Loan). However, 1 classifier has a predicted probability of 0.54 and the other 0.95. Which one will you choose? Classification Accuracy will not help here as it will get both on 100%
# Doc: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html

######
# %%
# Running Log loss on training
print("The Log Loss on Training is: ", log_loss(y_train, pred_proba))

# Running Log loss on testing
pred_proba_t = log_reg.predict_proba(X_test)
print("The Log Loss on Testing Dataset is: ", log_loss(y_test, pred_proba_t))


##### HYPER PARAMETER TUNING ###
# We will loop over parameter C (Inverse of regularization strength).
# Inverse of regularization strength helps to avoid overfitting - it penalizes large values of your parameters
# It also helps to find Global Minimum by moving to better "solutions" from local minimum to global minimum
# The values of C to search should be n-equally-spaced values in log space ranging from 1e-5 to 1e5
# %%
np.geomspace(1e-5, 1e5, num=20)

# %%
# Creating a range for C values
np.geomspace(1e-5, 1e5, num=20)

# ploting it
plt.plot(np.geomspace(1e-5, 1e5, num=20))  # uniformly distributed in log space
# uniformly distributed in linear space, instead of log space
plt.plot(np.linspace(1e-5, 1e5, num=20))
# plt.plot(np.logspace(np.log10(1e-5) , np.log10(1e5) , num=20)) # same as geomspace

# %%
# Looping over the parameters

C_List = np.geomspace(1e-5, 1e5, num=20)
CA = []
Logarithmic_Loss = []

for c in C_List:
    log_reg2 = LogisticRegression(random_state=10, solver='lbfgs', C=c)
    log_reg2.fit(X_train, y_train)
    score = log_reg2.score(X_test, y_test)
    CA.append(score)
    print("The CA of C parameter {} is {}:".format(c, score))
    pred_proba_t = log_reg2.predict_proba(X_test)
    log_loss2 = log_loss(y_test, pred_proba_t)
    Logarithmic_Loss.append(log_loss2)
    print("The Logg Loss of C parameter {} is {}:".format(c, log_loss2))
    print("")

# %%
# putting the outcomes in a Table

# reshaping
CA2 = np.array(CA).reshape(20,)
Logarithmic_Loss2 = np.array(Logarithmic_Loss).reshape(20,)

# zip
outcomes = zip(C_List, CA2, Logarithmic_Loss2)

# df
df_outcomes = pd.DataFrame(
    outcomes, columns=["C_List", 'CA2', 'Logarithmic_Loss2'])

# print
df_outcomes

# Ordering the data (sort_values)
df_outcomes.sort_values("Logarithmic_Loss2", ascending=True).reset_index()

# %%
# Another way of doing the above
# Scikit-learn offers a LogisticRegressionCV module which implements Logistic Regression
# with builtin cross-validation to find out the optimal C parameter

kf = KFold(n_splits=3, random_state=0, shuffle=True)

# Logistic Reg CV
Log_reg3 = LogisticRegressionCV(random_state=15, Cs=C_List, solver='lbfgs')
Log_reg3.fit(X_train, y_train)
print("The CA is:", Log_reg3.score(X_test, y_test))
pred_proba_t = Log_reg3.predict_proba(X_test)
log_loss3 = log_loss(y_test, pred_proba_t)
print("The Logistic Loss is: ", log_loss3)

print("The optimal C parameter is: ", Log_reg3.C_)


# Doc: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html

"""
K-fold cross validation:
Advantage: K-fold cross validation uses all the training data to train the model, 
by applying k different splits; repeated train-test splits converge to the true accuracy given that the training data 
is representable for the underlying distribution; however in practise this is often overoptimistic.
 Disadvantage: The disadvantage of this method is that the training algorithm has to be rerun from the beginning k times,
  which means it takes k times as much computation to get an evaluation. Additionally, if you want to test the performance
   on a completely new dataset that the algorithm has never seen, you cannot do this with k-fold cross validation.
Hold-out:
Advantage: The advantage of Hold-out is that you can test how your model performs on completely unseen data that you haven't used when training the model. Additionally, Hold-out is usually much faster and less computationally expensive. Disadvantage: The evaluation may depend heavily on which data points end up in the training set and which end up in the test set, and thus the evaluation may be significantly different depending on how the division is made.
"""

# %%
# Maybe we have a different metric we want to track

# Looping over the parameters

C_List = np.geomspace(1e-5, 1e5, num=20)
CA = []
Logarithmic_Loss = []

for c in C_List:
    log_reg2 = LogisticRegression(random_state=10, solver='lbfgs', C=c)
    log_reg2.fit(X_train, y_train)
    score = log_reg2.score(X_test, y_test)
    CA.append(score)
    print("The CA of C parameter {} is {}:".format(c, score))
    pred_proba_t = log_reg2.predict_proba(X_test)
    log_loss2 = log_loss(y_test, pred_proba_t)
    Logarithmic_Loss.append(log_loss2)
    print("The Logg Loss of C parameter {} is {}:".format(c, log_loss2))
    print("")

    y_pred = log_reg2.predict(X_train)
    cm = confusion_matrix(y_train, y_pred)
    cm_norm = cm / cm.sum(axis=1).reshape(-1, 1)
    plot_confusion_matrix(cm_norm, classes=log_reg.classes_,
                          title='Confusion matrix')
    plt.show()

# %%
# Training a Dummy Classifier


dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
score = dummy_clf.score(X_test, y_test)

pred_proba_t = dummy_clf.predict_proba(X_test)
log_loss2 = log_loss(y_test, pred_proba_t)

print("Testing Acc:", score)
print("Log Loss:", log_loss2)


# Doc: https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html


# FINAL MODEL WITH SELECTED PARAMETERS
# %%
# Final Model

log_reg3 = LogisticRegression(random_state=10, solver='lbfgs', C=784.759970)
log_reg3.fit(X_train, y_train)
score = log_reg3.score(X_valid, y_valid)

pred_proba_t = log_reg3.predict_proba(X_valid)
log_loss2 = log_loss(y_valid, pred_proba_t)

print("Testing Acc:", score)
print("Log Loss:", log_loss2)

# %%
