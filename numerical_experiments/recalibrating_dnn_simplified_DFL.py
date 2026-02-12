"""
Simplified example for using the DNN model for forecasting prices with daily recalibration
"""

# Author: Jesus Lago

# License: AGPL-3.0 License
import os

import numpy as np

import opt_T3



# Number of layers in DNN
nlayers = 2

# Market under study. If it not one of the standard ones, the file name
# has to be provided, where the file has to be a csv file
dataset = 'BE'

# Number of years (a year is 364 days) in the test dataset.
years_test = 1

# Boolean that selects whether the validation and training datasets were shuffled when
# performing the hyperparameter optimization. Note that it does not select whether
# shuffling is used for recalibration as for recalibration the validation and the
# training datasets are always shuffled.
shuffle_train = 1

# Boolean that selects whether a data augmentation technique for DNNs is used
data_augmentation = 0

# Boolean that selects whether we start a new recalibration or we restart an existing one
new_recalibration = 1

# Number of years used in the training dataset for recalibration
calibration_window = 3

# Unique identifier to read the trials file of hyperparameter optimization
experiment_id = 1

# Optional parameters for selecting the test dataset, if either of them is not provided, 
# the test dataset is built using the years_test parameter. They should either be one of
# the date formats existing in python or a string with the following format
# "%d/%m/%Y %H:%M"
begin_test_date = None
end_test_date = None

# Set up the paths for saving data (this are the defaults for the library)
path_datasets_folder = os.path.join('.', 'datasets')
path_recalibration_folder = os.path.join('.', 'experimental_files_d3')
path_hyperparameter_folder = os.path.join('.', 'experimental_files_d3')

#Define optmodel
T = 24
C= int(10.0*100)
x_on= int(5.0*100)
s_i= int(0.0*100)
d = np.array(np.ones(T)*0.3*C, dtype=int)
S = np.array(np.ones(T)*2.5*C, dtype=int)
optmodel  = opt_T3.LotSizingModel(T, C, x_on, d, S, s_i)

#Define loss
loss = 'FY'
os.environ["KERAS_BACKEND"] = "torch"

days=1

from epftoolbox.models import evaluate_once_dnn_in_test_dataset_DFL
evaluate_once_dnn_in_test_dataset_DFL(experiment_id, optmodel, loss=loss, path_hyperparameter_folder=path_hyperparameter_folder, 
                               path_datasets_folder=path_datasets_folder, shuffle_train=shuffle_train,  days=days,
                               path_recalibration_folder=path_recalibration_folder, 
                               nlayers=nlayers, dataset=dataset, years_test=years_test, 
                               data_augmentation=data_augmentation, calibration_window=calibration_window, 
                               new_recalibration=new_recalibration, begin_test_date=begin_test_date, 
                               end_test_date=end_test_date)