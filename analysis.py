import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


###################
# Data Processing #
###################
data = pd.read_csv("data/processed_data.csv")

# Impute missing data using the mean.
data['MeanPauseDuration'].fillna(value=data['MeanPauseDuration'].mean(), inplace=True)

# Add the edu columns
edu_bs = np.array([1-int(x) for x in data['Community college or university'].isnull()])
edu_gra = np.array([1-int(x) for x in data['Graduate'].isnull()])
