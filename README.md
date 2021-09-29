# prosodic_analysis

The repository is for the collaborators of the prosodic analysis project. For privacy reason, the data hasn't been released yet.
To use the code locally, first create a directory with the name "data" under the content root directory and place the "VoxitData101BWPClean9-27-2021.xlsx" file under the "data" directory. 
Then type run `python data_preprocessing.py` in the command line. After running the script, the user should be able to use the functions in cluster.py and data_exploration.ipynb.
Users need to run the fit_linear_mixed_model.ipynb before executing the q_value.R file to run the q-methodology.
