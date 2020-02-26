import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("data/processed_data.csv")
#data = data.dropna()

data['MeanPauseDuration'].fillna(value = data['MeanPauseDuration'].mean(), inplace=True)

edu_bs = np.array([1-int(x) for x in data['Community college or university'].isnull()])
edu_gra = np.array([1-int(x) for x in data['Graduate'].isnull()])

edu = np.repeat()


edu = [0]*len(data)
for i in range(len(edu_gra)):
    if edu_gra[i] == 1:
        edu[i] = 1
    else: 
        edu[i] = 0

edu = np.array(edu)


data.columns


pitch = data[['f0Mean', 'f0Range2sd', 'f0Entropy']]
pause = data[['PauseRate', 'PauseDutyCycle', 'MeanPauseDuration']]
complexity = data[['ComplexityAllPauses', 'ComplexitySyllables', 'ComplexityPhrases']]
intensity = data[['IntensitySegmentMeanSD', 'IntensityMeanAbsVelocity', 'IntensityMeanAbsAccel']]
pitch.shape, pause.shape, complexity.shape, intensity.shape


features = {'pitch': pitch, 'pause': pause, 'complexity': complexity, 'intensity': intensity}


def plot_cluster(f, ncluster):
    X = features[f]
    #kmeans = KMeans(n_clusters=ncluster, random_state=0).fit(X)
    fig = plt.figure(figsize=(6, 5))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    cdict = {0: 'orange', 1: 'red', 2: 'blue', 3: 'green'}
    col = X.columns
    for i in range(ncluster):
        idx = np.array((np.where(edu==i)))[0]
        ax.scatter3D(X[col[0]][idx], X[col[1]][idx], X[col[2]][idx], c=cdict[i], label = i)
        ax.legend()
        ax.set_xlabel(col[0])
        ax.set_ylabel(col[1])
        ax.set_zlabel(col[2])
    plt.show()
    return
    


# In[264]:


plot_cluster('pitch',2)


# In[265]:


plot_cluster('intensity',2)


# In[266]:


plot_cluster('pause',2)


# In[267]:


plot_cluster('complexity',2)


# In[268]:


col_names = ['f0Mean',
       'f0Range2sd', 'f0Entropy', 'f0MeanAbsVelocity', 'f0MeanAbsAccel',
       'PauseCount', 'PauseRate', 'PauseDutyCycle', 'MeanPauseDuration',
       'ComplexityAllPauses', 'ComplexitySyllables', 'ComplexityPhrases',
       'IntensitySegmentMeanSD', 'IntensityMeanAbsVelocity',
       'IntensityMeanAbsAccel', 'Dynamism']


# In[269]:


import scipy.stats as stats


# In[328]:


def f(group, fea):
    if group == 'edu':
        g0 = np.array((np.where(edu==0)))[0]
        g1 = np.array((np.where(edu==1)))[0]
    else:
        g0 = np.array(np.where(data[group] == 0))[0]
        g1 = np.array(np.where(data[group] == 1))[0]
    alpha = 0.05 
    data0 = data[fea][g0]
    data1 = data[fea][g1]
    F = np.var(data0, ddof=1) / np.var(data1, ddof=1)
    df1 = len(data0)-1
    df2 = len(data1)-1
    fdistribution1 = stats.f(df1,df2)
    fdistribution2 = stats.f(df2,df1)
    p_value = 1 - max(fdistribution1.cdf(F), fdistribution2.cdf(1/F)) + min(fdistribution1.cdf(F), fdistribution2.cdf(1/F))
    f_critical1 = fdistribution1.ppf(0.025)
    f_critical2 = fdistribution1.ppf(0.975)
    #print(p_value1, p_value2, f_critical1, F, f_critical2)
    #return 1-(p_value > alpha) == (f_critical1<=F and F<=f_critical2)
    #'''
    if p_value > alpha:
        print(fea, p_value, stats.ttest_ind(data0,data1, equal_var = False))
    else: 
        print(fea, p_value, stats.ttest_ind(data0,data1))
    #'''
    


# In[329]:


groups = ['Public 0 /Private 1', 'edu', 'Cave Canem?']
for g in groups:
    count = 0
    print(g)
    for fea in col_names:
        f(g, fea)
    

