
# coding: utf-8

# In[7]:


import pandas as pd
import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import scatter_matrix

# Load the data
data = pd.read_csv("data/data_for_analysis.csv")


# In[8]:


import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[9]:


data['name'] = data['Author last name'].str.strip() + data['Author first name'].str.strip()
data['name'] = data['name'].str.lower()


# In[10]:


data.rename(columns={'Birth Year': 'birthyear'}, inplace=True)


# In[11]:


data.columns


# In[13]:


set(data['Region'])


# In[14]:


data['Region'].fillna(value = 'Other', inplace = True)
data['Region'].replace(['Various', 'Unknown', 'British'], 'Other', inplace = True)
set(data['Region'])


# In[16]:


model_dyn = smf.mixedlm("Dynamism ~ C(cave_canem_indicator) + C(graduate_study_indicator) + C(Region, Treatment('Caribbean'))", data, groups=data["name"])
dyn = model_dyn.fit()
print()
print(dyn.summary())


# In[17]:


model_pauserate = smf.mixedlm("MeanPauseDuration ~ C(cave_canem_indicator) + C(undergrad_study_indicator) + C(Region, Treatment('Caribbean'))", data, groups=data["name"])
pauserate = model_pauserate.fit()
print(pauserate.summary())

