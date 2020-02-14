import pandas as pd
import numpy as np

data = pd.read_csv("data/data_for_analysis.csv")
data.rename(columns = {'Cave Canem?': "cave_canem_indicator", 'Public 0 /Private 1': "public_private_indicator"},
            inplace=True)

# Impute missing data with mean
data['MeanPauseDuration'].fillna(data['MeanPauseDuration'].mean(), inplace=True)

# create indicator variables for education
undergrad_study_indicator = pd.Series(np.array([1-int(x) for x in data['Community college or university'].isnull()]))
graduate_study_indicator = pd.Series(np.array([1-int(x) for x in data['Graduate'].isnull()]))
new_data_dict = {"undergrad_study_indicator": undergrad_study_indicator,
                 "graduate_study_indicator": graduate_study_indicator}

# Change some column names
data = pd.concat([data, pd.DataFrame(new_data_dict)], axis=1)
data.to_csv("data/data_for_analysis.csv")
