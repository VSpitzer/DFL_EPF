"""
Example for optimizing the hyperparameter and features of the DNN model
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np

from epftoolbox.models import hyperparameter_optimizer_DFL

import opt_T3


# Number of layers in DNN
nlayers = 2

# Market under study. If it not one of the standard ones, the file name
# has to be provided, where the file has to be a csv file
dataset = 'NP'

# Number of years (a year is 364 days) in the test dataset.
years_test = 2

# Optional parameters for selecting the test dataset, if either of them is not provided, 
# the test dataset is built using the years_test parameter. They should either be one of
# the date formats existing in python or a string with the following format
# "%d/%m/%Y %H:%M"
begin_test_date = None
end_test_date = None

# Boolean that selects whether the validation and training datasets are shuffled
shuffle_train = 1

# Boolean that selects whether a data augmentation technique for DNNs is used
data_augmentation = 0

# Boolean that selects whether we start a new hyperparameter optimization or we restart an existing one
new_hyperopt = 1

# Number of years used in the training dataset for recalibration
calibration_window = 4

# Unique identifier to read the trials file of hyperparameter optimization
experiment_id = 1

# Number of iterations for hyperparameter optimization
max_evals = 2

path_datasets_folder = "./datasets/"
path_hyperparameters_folder = "./experimental_files/"

#Define optmodel
T = 24
C= int(10.0*100)
x_on= int(5.0*100)
s_i= int(0.0*100)
d = np.array(np.round(np.random.rand(T),2)*C, dtype=int)
S = np.array(np.round(2+np.random.rand(T),2)*C, dtype=int)
optmodel  = opt_T3.LotSizingModel(T, C, x_on, d, S, s_i)

#Define loss
loss = 'SPO'

# Check documentation of the hyperparameter_optimizer for each of the function parameters
# In this example, we optimize a model for the PJM market.
# We consider two directories, one for storing the datasets and the other one for the experimental files.
# We start a hyperparameter optimization from scratch. We employ 1500 iterations in hyperopt,
# 2 years of test data, a DNN with 2 hidden layers, a calibration window of 4 years,
# we avoid data augmentation,  and we provide an experiment_id equal to 1
hyperparameter_optimizer_DFL(optmodel, loss, path_datasets_folder=path_datasets_folder, 
                         path_hyperparameters_folder=path_hyperparameters_folder, 
                         new_hyperopt=new_hyperopt, max_evals=max_evals, nlayers=nlayers, dataset=dataset, 
                         years_test=years_test, calibration_window=calibration_window, 
                         shuffle_train=shuffle_train, data_augmentation=0, experiment_id=experiment_id,
                         begin_test_date=begin_test_date, end_test_date=end_test_date)

