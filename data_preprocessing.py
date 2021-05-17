import pandas as pd
import numpy as np

data = pd.read_excel("data/101AfAmWomenPoetsVoxitResults5-3-21.xlsx", engine='openpyxl', nrows=202)
data = data[data.columns[:26].append(data.columns[-1:]).append(data.columns[26:-1])]
data.columns = ["_".join(x.strip().lower().split(" ")) for x in data.columns[:27]] + data.columns[27:].to_list()

data.rename({"cave_canem_(fellow,_faculty,_or_board)": "cave_canem",
             'public_0_/private_1': "private_school", "undergrad/comm._college": "undergraduate_school",
             'recording:_live,_studio,_or_self-recorded': "recording",
             'audience:academic?_spoken_word?_poetry_reading_space?_poetry_festival?_cave_canem_channel?_bar_reading_series?_book_fair?_residence?_gallery?_bookstore?_other?':
            "audience", 'major_award_(1),_none_(0)': 'major_award'}, axis=1, inplace=True)
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

data.to_csv("data/data_for_analysis.csv", index=False)


