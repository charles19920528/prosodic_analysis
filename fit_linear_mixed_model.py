import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

data = pd.read_csv("data/data_for_analysis.csv")


linear_mixed_model_data = data.dropna(subset=["birth_year", "cave_canem_indicator"])
# Fit model for f0Range2sd
linear_mixed_model = smf.mixedlm("f0Range2sd ~ birth_year + cave_canem_indicator", linear_mixed_model_data,
                                 groups=linear_mixed_model_data["poet_full_name"])
linear_mixed_model_fit = linear_mixed_model.fit()
print(linear_mixed_model_fit.summary())




