# %%
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# %%
data = pd.read_csv('Data/CSV/New_Data.csv', header=0)
data = data.dropna()
print(data.shape)
print(list(data.columns))

# %%
data['adj %'].unique

# %%
data['D_or_ND'].value_counts()

# %%
sns.countplot(x='D_or_ND', data=data, palette='hls')
plt.show()
plt.savefig('count_plot')

# %%
count_no_sub = len(data[data['D_or_ND'] == 0])
count_sub = len(data[data['D_or_ND'] == 1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription", pct_of_sub*100)

# %%
# try with different variables
data.groupby('D_or_ND').mean()

# %%
data.groupby('noun %').mean()

# %%
%matplotlib inline
pd.crosstab(data.no_useless_tokens, data.D_or_ND).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('no_words')
plt.ylabel('D or ND')
plt.savefig('purchase_fre_job')


######

# %%
