# Basic information


Code for the SFCN model is taken from : https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain

Paper: Accurate brain age prediction with lightweight deep neural networks

The code by default uses the age range [42, 82] and trains the outermost layer on the user provided data.

We currently use the whole T1 image.


# Installation
**Install following python packages:**
1. Pytorch/torch and torchvision (https://pytorch.org/get-started/locally/)
2. pandas (https://pypi.org/project/pandas/)
3. nilearn (https://nilearn.github.io/stable/index.html)
4. matplotlib (https://matplotlib.org/)

**Install using requirements.txt:**
1. Create a virtual enviornment and activate it\
python -m venv ~/.venvs/cnn_env\
source ~/.venvs/cnn_env/bin/activate 

2. Install from requirements.txt file\
pip install -r requirements.txt


# Folder contents

**dl_brainage** contains following folders and files: 

1. **brain_age**: contains the parameters/weights file of model trained by Hang Peng (link above)

2. **dp_model**: contains the models (SFCN, resnet3d) and additional functions for loss and other utils (link above)

3. **train_model.py**: This is main file for model training. It is also called by the train_script.submit script. It takes in 10 arguments:
    - `input_file` : Path to the input file, a csv file with 2 required columns age and file_path (e.g. ixi_subject_list_train.csv: contains site, participant_id, age, sex and file_path)
    - `save_path`: Path to save results
    - `save_filename`: Output filename name prefix (multiple files will be created, e.g. `_model.pt` and `_checkpoint.pt`)
    - `learning_rate`: Learning rate for optimizer, a small number e.g. 0.0001
    - `num_epochs`: Number of epochs for model training
    - `batch_size`: Number of samples to be used in each batch, this also depends on the GPU memory usually ~3
    - `train_all_flag`: Define whether to train all the layers of the model or just the outermost layer (1 or 0)
    - `optimizer_name`: the optimizer to be used for training ('adam' or 'sgd')
    - `age_range`: Age range in the dataset used (eg: [42, 82])
    - `trained_weights`: Path to the pre-trained weights file

12. **test_model.py**: This is the file to used to test the model. It is also called by test_script.submit. It takes 5 arguments:
    - `input_file` : Path to the input file for testing, a csv file with 2 required columns age and file_path (e.g. ixi_subject_list_test.csv: contains site, participant_id, age, sex and file_path)
    - `save_path`: Path to save results
    - `save_filename`: Output filename name prefix (multiple files will be created, e.g. `_predictions.csv` and `_true_vs_pred.png`)
    - `age_range`: Age range in the dataset used
    - `trained_weights`: Path to the weights file for the trained model which you want to use to make predictions
    

12. **train_validation.py**: contains functions to train and test the model. Also, it contains SFCN_mod class which is modified version of SFCN which can be used to be train on data from any age range. This CNN model is used for training and testing.

13. **load_data**.py: reads the input csv file (`input_file`) which contains subject information. Age and file_path are needed for the analysis.

14. **train_submit.submit**: calls the train_model.py script and submits the job to HT condor

14. **train_submit.submit**: calls the test_model.py script and submits the job to HT condor

15. **ixi_subject_list.csv**: contains subject information (site, participant_id, age, sex and file_path) for the IXI dataset

15. **get_train_test_csv**: creates train and test csv files given an input file

15. **ixi_subject_list_train.csv**: contains subject information (site, participant_id, age, sex and file_path) for the IXI dataset for training

15. **ixi_subject_list_test.csv**: contains subject information (site, participant_id, age, sex and file_path for the IXI dataset for testing


# To run the code: 


##### Train a model:

- *Pre-trained model:* We can perform transfer learning starting with the a pre-trained model (trained on the UKB data) provided in the folder `brain_age`. 
- *New data:* You will need to provide a file containing age and nifti file names to use, e.g. `ixi_subject_list_train.csv`. 
- The newly provided data is split into 75% train and 25% validation to detect when to stop training.
- You can also provide the age range to use, the pre-prained model used age range `42,82`.


You can execute the `train_model.py` with 10 arguments. For example:

`python3 train_model.py --input_file ./data/T1/ixi_fsl_bet_flirt_antsBC_train.csv --save_path ./results --save_filename test --learning_rate 0.0001 --num_epochs 240 --batch_size 3 --train_all_flag 0 --optimizer_name adam --age_range 42,82 --trained_weights ./brain_age/run_20190719_00_epoch_best_mae.p`

In this case, the trained model and dictionary of checkpoint (`_model.pt` and `_checkpoint.pt`) will be saved in the results folder in dl_brainage directory


You can also submit this as a job to a cluster, e.g. using HTcondor for model training:

`cd dl_brainage`

`condor_submit train_submit.submit`

This runs the `train_model.py` with the 10 arguments mentioned above. You will need to change these arguments to your settings in the `train_submit.submit` file.


##### For prediction/testing: 

You can execute the `test_model.py` with 5 arguments. For example:

`python3 test_model.py --input_file ./data/T1/ixi_fsl_bet_flirt_antsBC_test.csv --save_path ./results --save_filename xx1 --age_range 42,82 --trained_weights ./brain_age/run_20190719_00_epoch_best_mae.p`

In this case, the predictions for the test data and scatter plot between true age and predicted age (`_predictions.csv` and `_true_vs_pred.png`) will be saved in the results folder in dl_brainage directory

You can also submit it as a job to a cluster running HTcondor for model testing:

`cd dl_brainage`

`condor_submit test_script.submit`

This runs the `test_model.py` with the 5  arguments mentioned above. You will need to change these arguments in the `test_script.submit` file.
