import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import scatter_matrix

# Load the data
data = pd.read_csv("data/data_for_analysis.csv")

# Categorize features.
pitch_name_vet = ['f0Mean', 'f0Range2sd', 'f0Entropy', 'f0MeanAbsVelocity', 'f0MeanAbsAccel']
pause_name_vet = ['PauseRate', 'PauseDutyCycle', 'MeanPauseDuration']
complexity_name_vet = ['ComplexityAllPauses', 'ComplexitySyllables', 'ComplexityPhrases']
intensity_name_vet = ['IntensitySegmentMeanSD', 'IntensityMeanAbsVelocity', 'IntensityMeanAbsAccel']

full_measurement_name_vet = pitch_name_vet + pause_name_vet + intensity_name_vet + complexity_name_vet + ["Dynamism"]

features = {'pitch': pitch_name_vet, 'pause': pause_name_vet, 'complexity': complexity_name_vet,
            'intensity': intensity_name_vet}

########################
# Matrix scatter plots #
########################
# Conclusions for clustering.
# Examine the relationship between measurements. Try to find dimensions for clustering.
# 'f0MeanAbsVelocity', 'f0MeanAbsAccel' are highly correlated which is not a surpose as accel is the derivative of vel.
# Both of them are highly correlated with Dynamism. This question the necessity of creating Dynamism.
# Dynamism seems to have linear relationship with all other measurements. We propose that use either dynamism or
# f0MeanAbsVelocity for one of the dimension to perform cluster on.

# birth year seems to be have a linear relationship with these measurements.
scatter_matrix(data[["birth_year", "region", "Dynamism"] + pitch_name_vet])

# We definitely want to use 'IntensityMeanAbsVelocity' for clustering.
# No other realtionship are found. Intensity measurements are also not related to the pitch measurements.
scatter_matrix(data[["birth_year", "Dynamism"] + intensity_name_vet])
scatter_matrix(data[intensity_name_vet + pitch_name_vet])

# Pause measurements don't seem to related to birth year. They are highly correlated between each other.
# Uuse "PauseRate" for clustering.
scatter_matrix(data[["birth_year", "Dynamism"] + pause_name_vet])

# By definition, Dynamism should be correlated with ComplexitySyllables. The graph supports the hypothesis.
# 'ComplexityAllPauses' and PauseRate are almost perfectly correlated.
scatter_matrix(data[["birth_year", "Dynamism", "PauseRate", 'IntensityMeanAbsVelocity'] + complexity_name_vet])

# Measurements for clustering: 'IntensityMeanAbsVelocity', "Dynamism" or "f0MeanAbsVelocity", PauseRate。

# This plot is used for report
scatter_matrix(data[["birth_year", "Dynamism", 'MeanPauseDuration', 'f0Entropy', 'f0MeanAbsVelocity', "IntensityMeanAbsVelocity"]])


####################
# Boxplot and test #
####################
def boxplot(measure_name_vet, data_frame):
    for measure_name in measure_name_vet:
        response_column = data_frame[measure_name]
        column_name_vet = ["undergrad_study_indicator", "graduate_study_indicator",
                           "cave_canem_indicator", "public_private_indicator", "region", "Ivy"]
        label_vet=["Undergraduate", "Graduate", "Cave Canem", "Public or Private", "Region", "Ivy"]
        categorical_data_frame = data_frame[column_name_vet]

        fig, ax = plt.subplots(3, 2)
        for i, sub_ax in enumerate(ax.reshape(-1)):
            category_vet = np.unique(categorical_data_frame.iloc[:, i])
            response_vet = [response_column[categorical_data_frame.iloc[:, i] == category] for category in
                            category_vet]
            if column_name_vet[i] == "region":
                region_name_vet = ['Caribbean', 'Midwest', 'NYC', 'Northeast', 'South', 'West', 'Other']
                region_response_vet = []
                for region in region_name_vet:
                    region_response_vet.append(response_vet[np.where(category_vet == region)[0][0]])

                sub_ax.boxplot(region_response_vet, labels=region_name_vet)
                sub_ax.set_title(label_vet[i])
            else:
                sub_ax.boxplot(response_vet, labels=category_vet)
                sub_ax.set_xticklabels(["No", "Yes"])
                sub_ax.set_title(label_vet[i])

        fig.suptitle(measure_name)

boxplot(measure_name_vet=["Dynamism", "MeanPauseDuration", 'IntensityMeanAbsVelocity'], data_frame=data)



# frange_sd_2
boxplot(measure_name_vet=["f0Range2sd"], data_frame=data)
boxplot(measure_name_vet=["f0MeanAbsVelocity"], data_frame=data)
boxplot(measure_name_vet=["IntensityMeanAbsVelocity"], data_frame=data)

# Pause
# boxplot(measure_name_vet=full_measurement_name_vet, data_frame=data)
boxplot(measure_name_vet=pitch_name_vet + ["Dynamism"], data_frame=data)




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



