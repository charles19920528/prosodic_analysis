import pandas as pd
import statsmodels.formula.api as smf
data = pd.read_csv("data/data_for_analysis.csv")


model_dyn = smf.mixedlm("Dynamism ~ C(cave_canem_indicator) + C(graduate_study_indicator) + "
                        "C(region, Treatment('Caribbean'))", data, groups=data['poet_full_name'])
dyn = model_dyn.fit()
print(dyn.summary())


model_pause_rate = smf.mixedlm("MeanPauseDuration ~ C(cave_canem_indicator) + C(undergrad_study_indicator) + "
                              "C(region, Treatment('Caribbean'))", data, groups=data['poet_full_name'])
pause_rate = model_pause_rate.fit()
print(pause_rate.summary())

