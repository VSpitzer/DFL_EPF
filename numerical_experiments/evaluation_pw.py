"""
Simplified example for using the DNN model for forecasting prices with daily recalibration
"""
from epftoolbox.data import read_data

from epftoolbox.evaluation import Abs_Cost
from epftoolbox.evaluation import Regret
from epftoolbox.evaluation import MAE
from epftoolbox.evaluation import sMAPE

# Author: Jesus Lago

# License: AGPL-3.0 License
import os

import numpy as np
import pandas as pd

import opt_T3_pw


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

#Determine the studied loss
loss = 'FY'
DFL = True

#problem
load = 3

#Define optmodel
T = 3*24
C= 10*100
N_pw = 4
tab_ta = np.array([t*12+6 for t in range(N_pw)])
tab_td = np.array([6+2*12+t*12 for t in range(N_pw)])
d = np.round(np.array([ load*0.1*12 for n in range(N_pw) ])*C,0)
x_on=2*100
s_i=0

tab_t1 = np.sort(np.concatenate((tab_ta,tab_td+1)))
N_t1 = len(tab_t1)
d_pw = np.zeros(T)
d_pw[tab_td] = d
S = np.array([ sum(d[n] for n in range(N_pw) if (t>=tab_ta[n] and t<tab_td[n])) for t in range(T)])

optmodel  = opt_T3_pw.LotSizingModel_pw(T, N_t1, tab_t1, C, x_on, d_pw, S, s_i)

forecast_file_path = 'experimental_files_pw_'+str(load)+'/DNN_forecast_'+'DFL_'*DFL+ 'nl' + str(nlayers) + '_dat' + str(dataset) + '_' + loss + \
                       '_YT' + str(years_test) + '_SFH' + str(shuffle_train) + \
                       '_DA' * (data_augmentation) + '_CW' + str(calibration_window) + \
                       '_' + str(experiment_id)+'.csv'
path_datasets_folder = "./datasets/"

# Defining train and testing data
df_train, df_test = read_data(dataset=dataset, years_test=years_test, path=path_datasets_folder, 
                              begin_test_date=begin_test_date, end_test_date=end_test_date)
df_test = df_test.loc[df_test.index[0] + pd.Timedelta(weeks=1):,:]

# Defining forecast and the real values to be predicted in a more friendly format
forecast = pd.read_csv(forecast_file_path, index_col=0)
forecast.index = pd.to_datetime(forecast.index)

df_price = df_test.loc[:, ['Price']]
real_values = df_price.values.squeeze()
n = int((len(df_price) - T )*1./24) + 1
real_values = np.array([real_values[i*24:i*24+T] for i in range(n)])
real_values = pd.DataFrame(real_values, index=forecast.index, columns=forecast.columns)


# Evaluation
mae = np.mean(MAE(real_values.values,forecast.values.squeeze()))
smape = np.mean(sMAPE(real_values.values,forecast.values.squeeze())) * 100
true_cost = np.sum(Abs_Cost(real_values.values, forecast.values.squeeze(),  optmodel))
oracle_cost = np.sum(Abs_Cost(real_values.values, real_values.values.squeeze(),  optmodel))
abs_regret = (true_cost-oracle_cost)*100./oracle_cost
avg_regret = np.mean(Regret(real_values.values, forecast.values.squeeze(),  optmodel))
print('{} - sMAPE: {:.2f}%  |  MAE: {:.3f} | Abs Regret {:.2f}% | Avg Regret {:.2f}% '.format('Final metrics', smape, mae, abs_regret, avg_regret))