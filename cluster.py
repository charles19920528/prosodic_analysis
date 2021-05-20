import pandas as pd
import sklearn.cluster as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_cluster(measurement_name_vet, data_frame, n_cluster, cluster_method=sc.KMeans, **kwargs):
    """
    Perform K-means clustering, provide a 2d or 3d plot, and return poets names.
    :param measurement_name_vet: A vector of length 3 contains names of measurement to be plotted.
    :param n_cluster: An integer which specifies the number of clusters.
    :param data_frame: A pandas data frame. The default is the data frame produced by the data_preprocessing script.
    :param cluster_method: A class in sklearn.cluster module.
    :param kwargs: additional keyword arguments passed to the fit method of the cluster_method.
    :return:
    poets_cluster_dictionary: A dictionary of which each key is a cluster index and the values are names of poets
    belongs to the cluster.
    """
    measurement_data_frame = data_frame[measurement_name_vet]
    column_names_vet = measurement_data_frame.columns
    names_series = data_frame['author_last_name'] + " " + data_frame['author_first_name']

    kmeans = cluster_method(**kwargs).fit(measurement_data_frame)

    fig = plt.figure(figsize=(6, 5))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    color_dict = {0: 'orange', 1: 'red', 2: 'blue', 3: 'green'}

    for i in range(n_cluster):
        idx = (kmeans.labels_ == i)
        if len(measurement_name_vet) == 3:
            ax.scatter3D(measurement_data_frame[column_names_vet[0]][idx],
                         measurement_data_frame[column_names_vet[1]][idx],
                         measurement_data_frame[column_names_vet[2]][idx], color=color_dict[i], label=i)
        else:
            ax.scatter3D(measurement_data_frame[column_names_vet[0]][idx],
                         measurement_data_frame[column_names_vet[1]][idx], color=color_dict[i], label=i)
        ax.legend()
        ax.set_xlabel(column_names_vet[0])
        ax.set_ylabel(column_names_vet[1])
        if len(measurement_name_vet) == 3:
            ax.set_zlabel(column_names_vet[2])
    plt.show()
    poets_cluster_dictionary = {}
    for i in range(n_cluster):
        poets_cluster_dictionary[i] = pd.concat([names_series[kmeans.labels_ == i],
                                                 data_frame.loc[kmeans.labels_ == i, "poem_title"]], axis=1)

    return poets_cluster_dictionary

if __name__ == "__main__":
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

    pitch_cluster_dictionary = plot_cluster(measurement_name_vet=pitch_name_vet, n_cluster=2, data_frame=data)
    pause_cluster_dictionary = plot_cluster(measurement_name_vet=pause_name_vet, n_cluster=4, data_frame=data)
    complexity_cluster_dictionary = plot_cluster(measurement_name_vet=complexity_name_vet, n_cluster=4, data_frame=data)
    intensity_cluster_dictionary = plot_cluster(measurement_name_vet=intensity_name_vet, n_cluster=2, data_frame=data)

    cluster_to_report_dictionary = plot_cluster(measurement_name_vet=['IntensityMeanAbsVelocity',
                                                                      'MeanPauseDuration'], n_cluster=2, data_frame=data)
    cluster_to_report_dictionary[0].to_csv("cluster_0.csv")
    cluster_to_report_dictionary[1].to_csv("cluster_1.csv")

