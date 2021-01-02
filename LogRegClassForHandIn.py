#%%
# Packages / libraries
from sklearn import tree
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegressionCV
import os
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


# %% IMPORT DATA
df = pd.read_csv('Data/CSV/newnames2.csv')

# %% DROP COLUMNS WE DO NOT NEED
df = df.drop(columns=['Unnamed: 0', 'File', 'Occurence', 'Unique_words_in_doc',
                      'no_words', 'no_useful_tokens', 'no_useless_tokens'])


# Changing max_word_length because it counts '' and []
df['max_word_length'] = df['max_word_length'] - 4

df['Most_common_word_length'] = df['Most_common_word_length'] - 4


# %%
# CHECKING TYPES IN DATA
df.dtypes

# %%
# CHANGE TO FLOAT IF NEEDED
df['Unique_occ_perc'].astype(float)

######## DIVIDING ALL VARIABLES INTO QUARTILES ########
# %%
# MAX_WORD_LENGTH
print(df['max_word_length'].describe())
print('')

labels_4 = ['1', '2', '3', '4']
df['max_word_length_quar'] = pd.qcut(df['max_word_length'],
                                     q=4,
                                     labels=labels_4)
print(df.head())
print('')
print(df['max_word_length_quar'].value_counts())

# %%
# MOST COMMON WORD LENGTH
df['Most_common_word_length'].astype(float)

print(df['Most_common_word_length'].describe())
print('')

labels_4 = ['1', '2', '3', '4']
df['Most_common_word_length_quar'] = pd.qcut(df['Most_common_word_length'].rank(method='first'),
                                             4,
                                             labels=labels_4)
print(df.head())
print('')
print(df['Most_common_word_length_quar'].value_counts())


# %%
# OCCURENCE_PERC
print(df['Occurence_perc'].describe())
print('')

labels_4 = ['1', '2', '3', '4']
df['Occurence_perc_quar'] = pd.qcut(df['Occurence_perc'],
                                    q=4,
                                    labels=labels_4)
print(df.head())
print('')
print(df['Occurence_perc_quar'].value_counts())

# %%
# UNIQUE_OCCURENCE_PERC
print(df['Unique_occ_perc'].describe())
print('')

labels_4 = ['1', '2', '3', '4']
df['Unique_occ_perc_quar'] = pd.qcut(df['Unique_occ_perc'],
                                     q=4,
                                     labels=labels_4)
print(df.head())
print('')
print(df['Unique_occ_perc_quar'].value_counts())


# %%
# NOUN PERC
print(df['Noun_perc'].describe())
print('')

labels_4 = ['1', '2', '3', '4']
df['Noun_perc_quar'] = pd.qcut(df['Noun_perc'],
                               q=4,
                               labels=labels_4)
print(df.head())
print('')
print(df['Noun_perc_quar'].value_counts())


# %%
# VERB PERC
print(df['Verb_perc'].describe())
print('')

labels_4 = ['1', '2', '3', '4']
df['Verb_perc_quar'] = pd.qcut(df['Verb_perc'],
                               q=4,
                               labels=labels_4)
print(df.head())
print('')
print(df['Verb_perc_quar'].value_counts())

# %%
# ADJ PERC
print(df['Adj_perc'].describe())
print('')

labels_4 = ['1', '2', '3', '4']
df['Adj_perc_quar'] = pd.qcut(df['Adj_perc'],
                              q=4,
                              labels=labels_4)
print(df.head())
print('')
print(df['Adj_perc_quar'].value_counts())

# %%
# PRON_PERC
print(df['Pron_perc'].describe())
print('')

labels_4 = ['1', '2', '3', '4']
df['Pron_perc_quar'] = pd.qcut(df['Pron_perc'],
                               q=4,
                               labels=labels_4)
print(df.head())
print('')
print(df['Pron_perc_quar'].value_counts())


# %%
# ADV PERC
print(df['Adv_perc'].describe())
print('')

labels_4 = ['1', '2', '3', '4']
df['Adv_perc_quar'] = pd.qcut(df['Adv_perc'],
                              q=4,
                              labels=labels_4)
print(df.head())
print('')
print(df['Adv_perc_quar'].value_counts())

# %%
# %% SAVING EVERYTHING INTO A BIG DATAFRAME
# df.to_csv(r'Data/CSV/percandquar.csv')

# %%
df = pd.read_csv('Data/CSV/finalpercandquar.csv')

# %% DROP PERCENTAGE COLUMNS AND SAVE DF INTO A DATAFRAME WITH ONLY QUARTILES
df = df.drop(columns=['max_word_length', 'Occurence_perc', 'Unique_occ_perc', 'Noun_perc', 'Verb_perc',
                      'Adj_perc', 'Pron_perc', 'Adv_perc', 'Most_common_word_length'])
# %%
df.to_csv(r'Data/CSV/Final_Data.csv')

###### CLASSIFICATION STARTS HERE #######
# %%
# Loading the data
df = pd.read_csv('Data/CSV/Final_Data.csv')

# print the shape
print(df.shape)

df['D_or_ND'][df['D_or_ND'] == 'D'] = 1
df['D_or_ND'][df['D_or_ND'] == 'ND'] = 0

# runs the first 5 rows
df.head(5)

# %%
#Drop columns we do not need
df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])

df.head(5)

# %%
# Investigate all the elements whithin each Feature

for column in df:
    unique_values = np.unique(df[column])
    nr_values = len(unique_values)
    if nr_values <= 10:
        print("The number of values for feature {} is: {} -- {}".format(column,
                                                                        nr_values, unique_values))
    else:
        print("The number of values for feature {} is: {}".format(column, nr_values))

# %%
# Getting the distribution of D & ND
sns.countplot(x='D_or_ND', data=df, palette='Set3')

# %%
# Looping through all the features by our y variable - see if there is relationship

features = ['D_or_ND', 'max_word_length_quar', 'Most_common_word_length_quar',
            'Occurence_perc_quar', 'Unique_occ_perc_quar', 'Noun_perc_quar',
            'Verb_perc_quar', 'Adj_perc_quar', 'Pron_perc_quar', 'Adv_perc_quar']


# %%
# Split the data into X & y

X = df.drop('D_or_ND', axis=1)
y = df['D_or_ND']

X
y

y = y.astype(int)

print(X.shape)
print(y.shape)

# %%
# Running Tree-based estimators

dt = DecisionTreeClassifier(random_state=1, criterion='gini')
dt.fit(X, y)

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
# Creating a Dataframe
fi_col
fi

fi_df = zip(fi_col, fi)
fi_df = pd.DataFrame(fi_df, columns=['Feature', 'Feature Importance'])
fi_df

# %%
# Ordering the data
fi_df = fi_df.sort_values('Feature Importance', ascending=False).reset_index()
fi_df

# %%
# Creating columns to keep
columns_to_keep = fi_df['Feature'][0:5]

fi_df
# %%

# SPLITTING THE RAW DATA - HOLD-OUT VALIDATION
# %%
# Print the shapes

print(df.shape)
print(df[columns_to_keep].shape)

# new_raw_data = new_raw_data[columns_to_keep]

# %%
# Split the data into X & y

X = df[columns_to_keep].values
X

y = df['D_or_ND']
y = y.astype(int)
y

print(X.shape)
print(y.shape)

# %%
# Hold-out validation

# first one
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, test_size=0.3, random_state=15)

# Second one
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, train_size=0.8, test_size=0.2, random_state=15)

print(X_train.shape)
print(X_test.shape)
print(X_valid.shape)

print(y_train.shape)
print(y_test.shape)
print(y_valid.shape)

# %%
# Investigating the distribution of all ys

ax = sns.countplot(x=y_valid, palette="Set3")


#### RUNNING LOGISTIC REGRESSION ####
# %%
# Training model

log_reg = LogisticRegression(random_state=10, solver='liblinear')

log_reg.fit(X_train, y_train)

# %%
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
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes,
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

# %%
# Running Log loss on training
print("The Log Loss on Training is: ", log_loss(y_train, pred_proba))

# Running Log loss on testing
pred_proba_t = log_reg.predict_proba(X_test)
print("The Log Loss on Testing Dataset is: ", log_loss(y_test, pred_proba_t))


##### HYPER PARAMETER TUNING ###
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
    log_reg2 = LogisticRegression(random_state=10, solver='liblinear', C=c)
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
# Trying to find the optimal C parameter with cross validation
kf = KFold(n_splits=3, random_state=0, shuffle=True)

# Logistic Reg CV
Log_reg3 = LogisticRegressionCV(
    cv=kf, random_state=15, Cs=C_List, solver='liblinear')
Log_reg3.fit(X_train, y_train)
print("The CA is:", Log_reg3.score(X_test, y_test))
pred_proba_t = Log_reg3.predict_proba(X_test)
log_loss3 = log_loss(y_test, pred_proba_t)
print("The Logistic Loss is: ", log_loss3)

print("The optimal C parameter is: ", Log_reg3.C_)
#Gives a lower accuracy than if calculating the C parameter manually

# %%
# Dummy Classifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
score = dummy_clf.score(X_test, y_test)

pred_proba_t = dummy_clf.predict_proba(X_test)
log_loss2 = log_loss(y_test, pred_proba_t)

print("Testing Acc:", score)
print("Log Loss:", log_loss2)


# %%
# FINAL MODEL WITH SELECTED PARAMETERS
# Final Model

log_reg3 = LogisticRegression(random_state=10, solver='liblinear', C=0.162378)
log_reg3.fit(X_train, y_train)
score = log_reg3.score(X_valid, y_valid)

pred_proba_t = log_reg3.predict_proba(X_valid)
log_loss2 = log_loss(y_valid, pred_proba_t)

print("Testing Acc:", score)
print("Log Loss:", log_loss2)


#%%
# DET HER VIRKER IKKE HERFRA OG NED.
def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes,
                    vmin=0., vmax=1., annot=True, annot_kws={'size': 50})
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#%%
log_reg.predict(X_valid)
y_pret = log_reg.predict(X_valid)
print(y_pret)
# %%
# Visualizing cm
cm = confusion_matrix(y_valid, y_pret)
cm_norm = cm / cm.sum(axis=1).reshape(-1, 1)

plot_confusion_matrix(cm_norm, classes=log_reg.classes_,
                      title='Confusion matrix')
# %%
# Accuracy on Validation
print("The Validation Accuracy is: ", log_reg.score(X_valid, y_valid))

# Classification Report
print(classification_report(y_valid, y_pret))
# %%
