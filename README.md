run this first to do the preprocessing. edit the following files:
1. meta.yaml, located in the main folder
2. default_ops.py, located in the preprocessing folder
3. pipeline.py, located in the main folder

The main data folder should have two folders within it--the raw data folder, the roi folder. the script will add the suite2p folder to this main data folder after it runs the registration, and later the extracted traces after it calls the rois from the roi folder to run extraction

1. take the mean image using the raw tif stack (using ImageJ) and run cell profiler on this image to extract the axon masks
2. take the extracted axon masks and put them in a folder called /rois
3. move the raw data to this same folder with the /rois, the raw data should be in a separate folder called /raw
4. change the parameters for where the data is, the frame rate, the pixels per micron, etc. in the meta.yaml.
5. verify that the meta.yaml and the pipeline.py files you are using are in the same directory, and that the correct meta.yaml file is being called in your pipeline.py file.
6. run the pipeline file 

If you set accepted_only=False in the data loader it will ignore the iscell variable

then, once you are done preprocessing, run the thalcor-tuning repo to plot the data. to do this, refer to the following:
for tuning, use popTuningSingleChan.py

to run the summary plots, you will need to set the paths to the folders that contain the raw files and rois, by changing the following lines at the end of the pipeline script:
figure_directory = Path(r"Figure Directory")
os.makedirs(figure_directory, exist_ok=True)
data = TuningDataSingleChannel(r"Raw Data, Suite2p, and ROI path", r"Stimulus info.yaml Path", accepted_only=False)
population_summary(data, figure_directory)
population_average_summary(data, figure_directory)

for behavior
1. under utils>data_loader.py there are two implementations of data loader class. One is called SessionDataV2. note this expects data from both channels
2. there are some code samples for data visualization in nb_imaging_session_analysis_gng.ipynb


to run this on the computational cluster rockfish:
1. log into rockfish using windows powershell by using the command:
 ssh -X sjkim1@login.rockfish.jh.edu and enter your password. if this is the first time:
2. run the following commands:
module load gcc/9.3.0
module load anaconda
3. create a suite2p environment:
```
conda create --name suite2p python=3.9
python -m pip install suite2p[io]
conda install pyyaml 
```
suite2p[io] will install suite2p and all of the dependencies
to read the yaml files, you will need to install the module yaml. if you use pip/conda install yaml this will not work and after you run the job in slurm, an error will appear saying the module yaml is not found. to avoid this use conda install yaml

you may get problems with your environment, such as "the environment is inconsistent," but this is fine and it will run the job regardless

5. format the data as the following:
sessionName > raw > tiff file
and within these sessionName folders, have folders for the rois. then, generate the masks in cell profiler and put them in the rois folder. 
within this sessionName folder, the code will create the suite2p folder within it
7. produce a .yaml file that has your metadata
8. produce a slurm file by running the following in the terminal
```nano job.slurm```
and paste the following:
```
#!/bin/bash
#SBATCH --job-name=tuning_axon	  # create a short name for your job
#SBATCH --nodes=1         # node count
#SBATCH --ntasks=1         # total number of tasks across all nodes
#SBATCH --cpus-per-task=1     # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64G     # memory per cpu-core (4G is default)
#SBATCH --time=16:00:00      # total run time limit (HH:MM:SS)
#SBATCH --partition=parallel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sjkim1@jh.edu

module purge
module load gcc/9.3.0
module load anaconda3
conda activate preprocessing
cd /vast/kkuchib1/mohammad/kuchibhotlalab-pipeline

python pipeline_tuning_single_channel.py
```

and save the file
8. when you want to run the task, in the terminal, type
```sbatch job.slurm```
9. then, a job id will be generated and a file named slurm-jobid.log will be created that outputs text from the job you are running. jobid is a string of numbers
10. check progress use ```sqme```
12. to check the job, use ```cat slurm-jobid.out```
