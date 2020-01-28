import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


################
# Data loading #
################
data = pd.read_csv("./data/100AfAmWomenPoetsVoxitResults.csv")
data = data.dropna()
poems = data['file']
data = data.drop(['file'], axis=1)

# Inspect data
data.columns
data.shape



#######################################
# Clustering on one feature category. #
#######################################
# Categorize features.
pitch = data[['f0Mean', 'f0Range2sd', 'f0Entropy']]
pause = data[['PauseRate', 'PauseDutyCycle', 'MeanPauseDuration']]
complexity = data[['ComplexityAllPauses', 'ComplexitySyllables', 'ComplexityPhrases']]
intensity = data[['IntensitySegmentMeanSD', 'IntensityMeanAbsVelocity', 'IntensityMeanAbsAccel']]
pitch.shape, pause.shape, complexity.shape, intensity.shape



features = {'pitch': pitch, 'pause': pause, 'complexity': complexity, 'intensity': intensity}


def plot_cluster(feature_category_vet, ncluster):
    """
    Perform K-means clustering, provide a 3d plot, and return poems
    :param feature_category_vet: A vector which should be one of the feature category defined above.
     Ie. pitch, pause, complexity or intensity. The function assumes that each feature category has exactly 3 features.
    :param ncluster: An integer which specifies the number of clusters.
    :return:
    poems_cluster_dictionary: A dictionary of which each key is a cluster index and the values are the poems in the
    cluster.
    """
    X = features[feature_category_vet]
    kmeans = KMeans(n_clusters=ncluster, random_state=0).fit(X)
    fig = plt.figure(figsize=(6, 5))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    cdict = {0: 'orange', 1: 'red', 2: 'blue', 3: 'green'}
    col = X.columns
    for i in range(ncluster):
        idx = (kmeans.labels_ == i)
        ax.scatter3D(X[col[0]][idx], X[col[1]][idx], X[col[2]][idx], c=cdict[i], label = i)
        ax.legend()
        ax.set_xlabel(col[0])
        ax.set_ylabel(col[1])
        ax.set_zlabel(col[2])
    plt.show()
    poems_cluster_dictionary = {}
    for i in range(ncluster):
        poems_cluster_dictionary[i] = poems[(kmeans.labels_ == i)]
    return poems_cluster_dictionary



pitch_cluster_dictionary = plot_cluster('pitch', 2)
pitch_cluster_dictionary[0]

pause_cluster_dictionary = plot_cluster('pause', 4)

complexity_cluster_dictionary = plot_cluster('complexity', 4)

intensity_cluster_dictionary = plot_cluster('intensity', 2)

