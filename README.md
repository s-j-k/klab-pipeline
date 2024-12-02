run this first to do the preprocessing. edit the following files:
1. meta.yaml, located in the main folder
2. default_ops.py, located in the preprocessing folder
3. pipeline.py, located in the main folder

then, once you are done preprocessing, run the thalcor-tuning repo to plot the data. to do this, refer to the following:
for tuning
1. go to the s2p branch of the thalcor-tuning repo and find nb_frequency_map.ipynb

for behavior
1. under utils>data_loader.py there are two implementations of data loader class. One is called SessionDataV2. note this expects data from both channels
2. there are some code samples for data visualization in nb_imaging_session_analysis_gng.ipynb
