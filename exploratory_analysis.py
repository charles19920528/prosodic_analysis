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

# Categorize features.
pitch_name_vet = ['f0Mean', 'f0Range2sd', 'f0Entropy']
pause_name_vet = ['PauseRate', 'PauseDutyCycle', 'MeanPauseDuration']
complexity_name_vet = ['ComplexityAllPauses', 'ComplexitySyllables', 'ComplexityPhrases']
intensity_name_vet = ['IntensitySegmentMeanSD', 'IntensityMeanAbsVelocity', 'IntensityMeanAbsAccel']

full_measurement_name_vet = pitch_name_vet + pause_name_vet + intensity_name_vet + complexity_name_vet + ["Dynamism"]

features = {'pitch': pitch_name_vet, 'pause': pause_name_vet, 'complexity': complexity_name_vet,
            'intensity': intensity_name_vet}

###################
# 3D Scatter plot #
###################
# We didn't use loop to simplify the code here because the pop up window doesn't interact well when there are more than
# one windows.
def visualize_3d(measurement_name_vet, name_of_categorical_var, data_frame):
    """
    Plot the data in 3D. The color of each point is based on the value of the categorical variable supplied.
    :param measurement_name_vet: A vector of length 3 contains names of measurement to be plotted.
    :param name_of_categorical_var: A string which is the name of the categorical variable to divide data. We assume
    the categorical variable has less than 6 values
    :param data_frame: A pandas data frame. The default is the data frame produced by the data_preprocessing script.
    """
    sub_data_frame = data_frame[measurement_name_vet]
    column_names = sub_data_frame.columns
    categorical_series = data_frame[name_of_categorical_var]

    unique_categorical_values = np.unique(categorical_series)
    assert len(unique_categorical_values) <= 6, "The category has more than 6 levels."
    color_vet = ["b", "g", "r", "c", "m", "y"]

    fig = plt.figure(figsize=(6, 5))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    for i, level in enumerate(unique_categorical_values):
        level_equal_boolean = categorical_series == level
        ax.scatter3D(sub_data_frame.loc[level_equal_boolean, column_names[0]],
                     sub_data_frame.loc[level_equal_boolean, column_names[1]],
                     sub_data_frame.loc[level_equal_boolean, column_names[2]], c=color_vet[i], label = level)

        ax.legend()
        ax.set_xlabel(column_names[0])
        ax.set_ylabel(column_names[1])
        ax.set_zlabel(column_names[2])

        ax.set_title(name_of_categorical_var)

    plt.show()


# Examine if appearing on the cave canem affect the response. No effect.
visualize_3d(measurement_name_vet=intensity_name_vet, name_of_categorical_var="cave_canem_indicator", data_frame=data)
visualize_3d(measurement_name_vet=pitch_name_vet, name_of_categorical_var="cave_canem_indicator", data_frame=data)
visualize_3d(measurement_name_vet=complexity_name_vet, name_of_categorical_var="cave_canem_indicator", data_frame=data)
visualize_3d(measurement_name_vet=pause_name_vet, name_of_categorical_var="cave_canem_indicator", data_frame=data)

# No effect.
visualize_3d(measurement_name_vet=intensity_name_vet, name_of_categorical_var="public_private_indicator",
             data_frame=data)
visualize_3d(measurement_name_vet=pitch_name_vet, name_of_categorical_var="public_private_indicator",
             data_frame=data)
visualize_3d(measurement_name_vet=complexity_name_vet, name_of_categorical_var="public_private_indicator",
             data_frame=data)
visualize_3d(measurement_name_vet=pause_name_vet, name_of_categorical_var="public_private_indicator",
             data_frame=data)

# No effect
visualize_3d(measurement_name_vet=intensity_name_vet, name_of_categorical_var="undergrad_study_indicator",
             data_frame=data)
visualize_3d(measurement_name_vet=pitch_name_vet, name_of_categorical_var="undergrad_study_indicator",
             data_frame=data)
visualize_3d(measurement_name_vet=complexity_name_vet, name_of_categorical_var="undergrad_study_indicator",
             data_frame=data)
visualize_3d(measurement_name_vet=pause_name_vet, name_of_categorical_var="undergrad_study_indicator",
             data_frame=data)

# No effect
visualize_3d(measurement_name_vet=intensity_name_vet, name_of_categorical_var="graduate_study_indicator",
             data_frame=data)
visualize_3d(measurement_name_vet=pitch_name_vet, name_of_categorical_var="graduate_study_indicator",
             data_frame=data)
visualize_3d(measurement_name_vet=complexity_name_vet, name_of_categorical_var="undergrad_study_indicator",
             data_frame=data)
visualize_3d(measurement_name_vet=pause_name_vet, name_of_categorical_var="undergrad_study_indicator",
             data_frame=data)

########################
# Matrix scatter plots #
########################
# Potentially intersting
scatter_matrix(data[["Birth Year"] + pitch_name_vet])
scatter_matrix(data[["Birth Year", 'Dynamism']])

# Not useful
scatter_matrix(data[["Birth Year"] + intensity_name_vet])
scatter_matrix(data[["Birth Year"] + pause_name_vet])
scatter_matrix(data[["Birth Year"] + complexity_name_vet])



####################
# Boxplot and test #
####################
def boxplot(measure_name_vet, data_frame):
    for measure_name in measure_name_vet:
        response_column = data_frame[measure_name]
        column_name_vet = ["undergrad_study_indicator", "graduate_study_indicator",
                           "cave_canem_indicator", "public_private_indicator"]
        categorical_data_frame = data_frame[column_name_vet]

        fig, ax = plt.subplots(2, 2)
        for i, sub_ax in enumerate(ax.reshape(-1)):
            response_vet = [response_column[categorical_data_frame.iloc[:, i] == category] for category in
                            np.unique(categorical_data_frame.iloc[:, i])]
            sub_ax.boxplot(response_vet)
            sub_ax.set_title(column_name_vet[i])

        fig.suptitle(measure_name)


boxplot(measure_name_vet=full_measurement_name_vet, data_frame=data)



####################################
# Clustering for PCA visualization #
####################################
kmeans = KMeans(n_clusters=4, random_state=0)
measurement_data = data[pitch_name_vet + intensity_name_vet + pause_name_vet +complexity_name_vet + ["Dynamism"]]
predicted_label_vet = kmeans.fit_predict(measurement_data)


#######
# PCA #
#######
scaler = sklearn.preprocessing.StandardScaler()
standardized_data_array = scaler.fit_transform(measurement_data)

pca = PCA()
pca.fit(standardized_data_array)
plt.plot(np.arange(1, measurement_data.shape[1] + 1), pca.explained_variance_ratio_)
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


