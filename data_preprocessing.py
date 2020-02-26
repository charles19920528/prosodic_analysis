import pandas as pd
import numpy as np

data = pd.read_csv("data/processed_data.csv")
data.rename(columns = {'Cave Canem?': "cave_canem_indicator", 'Public 0 /Private 1': "public_private_indicator"},
            inplace=True)

# Impute missing data with mean
data['MeanPauseDuration'].fillna(data['MeanPauseDuration'].mean(), inplace=True)

# process region
data['Region'].fillna('Other', inplace = True)
data['Region'].replace(['Various', 'Unknown', 'British'], 'Other', inplace=True)
set(data['Region'])

# create indicator variables for education
undergrad_study_indicator = pd.Series(np.array([1-int(x) for x in data['Community college or university'].isnull()]))
graduate_study_indicator = pd.Series(np.array([1-int(x) for x in data['Graduate'].isnull()]))

# create name column
names_series = data['Author last name'].str.strip() + " " + data['Author first name'].str.strip()

new_data_dict = {"undergrad_study_indicator": undergrad_study_indicator,
                 "graduate_study_indicator": graduate_study_indicator,
                 "poet_full_name": names_series}

data = pd.concat([data, pd.DataFrame(new_data_dict)], axis=1)

# change some column names
data = data.rename(columns={"Birth Year": "birth_year", "Region": "region"})




data.to_csv("data/data_for_analysis.csv")


