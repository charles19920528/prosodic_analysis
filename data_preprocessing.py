import pandas as pd
import numpy as np

data = pd.read_excel("data/101AfAmWomenPoetsVoxitResults5-22-21.xlsx", engine='openpyxl', nrows=203)

data = data[data.columns[:28].append(data.columns[-1:]).append(data.columns[28:-1])]
data.columns = ["_".join(x.strip().lower().split(" ")) for x in data.columns[:27]] + data.columns[27:].to_list()

data.rename({"cave_canem_(fellow,_faculty,_or_board)": "cave_canem",
             'public_0_/private_1': "private_school", "undergrad/comm._college": "undergraduate_school",
             'recording:_live,_studio,_or_self-recorded': "recording",
             'audience:_academic_(poetry_festivals_and_universities)?_spoken_word?_public_reading_space_(bookstores,_bars,_galleries)?':
            "cave_canem_indicator", 'major_award_(1),_none_(0)': 'major_award', "Poem Title": "poem_title"}, axis=1, inplace=True)

data["birth_year"] = data["birth_year"].replace({0: np.nan})
data[["spoken_word", "cave_canem"]] = data[["spoken_word", "cave_canem"]].replace({0: False, 1: True})

# drop rows without prosodic measurements
data = data.loc[data["f0Mean"].notna(), :]

data['region'] = data['region'].replace(['Puerto Rico/West', 'Unknown', 'British', 'South/West'], 'Other')

# No school means not attending?
# Some poets have grad schools but not undergrad instituions.
data.loc[data['graduate_school'].notna(), 'graduate_school'] = True
data.loc[data['graduate_school'].isna(), 'graduate_school'] = False
data.loc[data["undergraduate_school"].notna(), "undergraduate_school"] = True
data.loc[data["undergraduate_school"].isna(), "undergraduate_school"] = False

# Strip out spaces in author names.
for col in data.columns[:2]:
    data[col] = data[col].str.strip()
data.replace({'Alison C': 'Alison C.'}, inplace=True)

data = data.rename(columns={
    'Dynamism Original Formula': "Dynamism",
    'private_school': "public_private_indicator",
    'undergraduate_school': 'undergrad_study_indicator',
    'graduate_school': 'graduate_study_indicator',
    'ivy': 'Ivy',
    'cave_canem': 'cave_canem_indicator',
})

# make author names in the format (last_name first_name)
data['poet_full_name'] = data['author_last_name'] + ' ' + data['author_first_name']

data.to_csv("data/data_for_analysis.csv", index=False)


