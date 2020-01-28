import pandas as pd
import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

data = pd.read_csv("./data/100AfAmWomenPoetsVoxitResults - 100AfAmWomenPoetsVoxitResults.csv")
data = data.dropna()

##############
# Clustering #
##############
kmeans = KMeans(n_clusters=4, random_state=0)
predicted_label_vet = kmeans.fit_predict(data)


#######
# PCA #
#######
scaler = sklearn.preprocessing.StandardScaler()
standardized_data_array = scaler.fit_transform(data)

pca = PCA()
pca.fit(standardized_data_array)
plt.plot(np.arange(1, data.shape[1] + 1), pca.explained_variance_ratio_)
plt.show()

pca.components_.shape
principal_components_array = pca.transform(standardized_data_array)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(principal_components_array[:, 0], principal_components_array[:, 1], principal_components_array[:, 2])
plt.show()


# Plot according to cluster
cdict = {1: 'red', 2: 'blue', 3: 'green', 0: "black"}
#fig, ax = plt.subplots()
fig = plt.figure()
ax = plt.axes(projection="3d")
for g in np.unique(predicted_label_vet):
    ix = np.where(predicted_label_vet == g)
#    ax.scatter(principal_components_array[ix, 0], principal_components_array[ix, 1], c = cdict[g], label = g, s = 100)
    ax.scatter3D(principal_components_array[ix, 0], principal_components_array[ix, 1], principal_components_array[ix, 2],
                 c = cdict[g], label = g, s = 100)
ax.legend()
plt.show()





