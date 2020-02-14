import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


################
# Data loading #
################
data = pd.read_csv("data/data_for_analysis.csv")


#######################################
# Clustering on one feature category. #
#######################################
# Categorize features.
pitch_name_vet = ['f0Mean', 'f0Range2sd', 'f0Entropy']
pause_name_vet = ['PauseRate', 'PauseDutyCycle', 'MeanPauseDuration']
complexity_name_vet = ['ComplexityAllPauses', 'ComplexitySyllables', 'ComplexityPhrases']
intensity_name_vet = ['IntensitySegmentMeanSD', 'IntensityMeanAbsVelocity', 'IntensityMeanAbsAccel']

def plot_cluster(measurement_name_vet, ncluster, data_frame):
    """
    Perform K-means clustering, provide a 3d plot, and return poets names.
    :param measurement_name_vet: A vector of length 3 contains names of measurement to be plotted.
    :param ncluster: An integer which specifies the number of clusters.
    :param data_frame: A pandas data frame. The default is the data frame produced by the data_preprocessing script.
    :return:
    poets_cluster_dictionary: A dictionary of which each key is a cluster index and the values are names of poets
    belongs to the cluster.
    """
    measurement_data_frame = data_frame[measurement_name_vet]
    column_names_vet = measurement_data_frame.columns
    names_series = data_frame['Author last name'] + " " + data_frame['Author first name']

    kmeans = KMeans(n_clusters=ncluster, random_state=0).fit(measurement_data_frame)

    fig = plt.figure(figsize=(6, 5))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    color_dict = {0: 'orange', 1: 'red', 2: 'blue', 3: 'green'}

    for i in range(ncluster):
        idx = (kmeans.labels_ == i)
        ax.scatter3D(measurement_data_frame[column_names_vet[0]][idx], measurement_data_frame[column_names_vet[1]][idx],
                     measurement_data_frame[column_names_vet[2]][idx], color=color_dict[i], label=i)
        ax.legend()
        ax.set_xlabel(color_dict[0])
        ax.set_ylabel(color_dict[1])
        ax.set_zlabel(color_dict[2])
    plt.show()
    poets_cluster_dictionary = {}
    for i in range(ncluster):
        poets_cluster_dictionary[i] = names_series[(kmeans.labels_ == i)]
    return poets_cluster_dictionary


pitch_cluster_dictionary = plot_cluster(measurement_name_vet=pitch_name_vet, ncluster=2, data_frame=data)
pause_cluster_dictionary = plot_cluster(measurement_name_vet=pause_name_vet, ncluster=4, data_frame=data)
complexity_cluster_dictionary = plot_cluster(measurement_name_vet=complexity_name_vet, ncluster=4, data_frame=data)
intensity_cluster_dictionary = plot_cluster(measurement_name_vet=intensity_name_vet, ncluster=2, data_frame=data)

