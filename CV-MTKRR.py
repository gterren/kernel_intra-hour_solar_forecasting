import pickle, glob, sys, os, warnings, csv, gpytorch, torch
import numpy as np
import math
from datetime import datetime
from mpi4py import MPI
import time

from sklearn.model_selection import KFold, LeaveOneOut, GridSearchCV
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVR

from scipy.stats import multivariate_normal

import scipy.special as ss

from feature_extraction_utils import _load_file, _save_file, _get_node_info
from solar_forecasting_utils_v2 import *

# Do not display warnings in the output file
warnings.filterwarnings('ignore')

# List of Experiments
def _get_experiment(i):
    exp_ = []
    for kernel, degree in zip(['linear', 'RBF', 'RQ', 'poly', 'matern', 'matern', 'matern'], [0, 0, 0, 2, 1./2., 3./2., 5./2.]):
        exp_.append([kernel, degree])
    return exp_[i]

# Kernels
def _kernel(X_, Y_, kernel, degree, gamma = None, beta = None):
    # Implementation of Kernel Functions...
    def __K(X, Y, kernel, degree, gamma, beta):
        if kernel is 'linear':
            gamma = torch.tensor(gamma, dtype = torch.float)
            # Compute Linear Kernel
            return gamma*(X @ Y.t())
        if kernel is 'poly':
            # Numpy to Torch
            gamma  = torch.tensor(gamma,  dtype = torch.float)
            beta   = torch.tensor(beta,   dtype = torch.float)
            degree = torch.tensor(degree, dtype = torch.float)
            # Compute Polynomial Kernel
            return (gamma*X @ Y.t() + beta)**degree
        if kernel is 'RBF':
            # Numpy to Torch
            gamma  = torch.tensor(gamma, dtype = torch.float)
            # Compute Radial Basis Function Kernel
            return torch.exp(-gamma*euclidean_distances(X, Y)**2)
        if kernel is 'RQ':
            # Numpy to Torch
            gamma  = torch.tensor(gamma, dtype = torch.float)
            beta   = torch.tensor(beta,  dtype = torch.float)
            # Compute Rational Quadratic Kernel
            return (1. + (gamma*euclidean_distances(X, Y)**2)/(2./beta) )**(-beta)
        if kernel is 'matern':
            # Numpy to Torch
            gamma  = torch.tensor(gamma,  dtype = torch.float)
            degree = torch.tensor(degree, dtype = torch.float)
            D_ = torch.sqrt(2.*degree)*(gamma*euclidean_distances(X, Y)**2)
            D_[D_ == 0.0] += torch.tensor(np.finfo(float).eps, dtype = torch.float)
            # This function is not implemented in torch
            A_ = ss.kn(degree, D_)
            # Compute Matern Kernel
            return ( (2**(1 - degree) )/ torch.tensor(ss.gamma(degree), dtype = torch.float)) * ( D_**degree ) * A_

    return __K(X_, Y_, kernel, degree, gamma, beta) + 1.


def _MTKRR_fit(X_, Y_, kernel, degree, gamma, beta, alpha, scale):
    def __get_correlation_matrix(N_tasks, scale):
        O_ = np.zeros((N_tasks, N_tasks))
        for i in range(N_tasks):
            for j in range(N_tasks):
                O_[i, j] = np.exp((N_tasks - abs(i-j))/(N_tasks*scale))
        return O_ / O_.max()

    N_tasks = Y_.shape[1]
    # Numpy to Torch
    X_ = torch.tensor(X_, dtype = torch.float)
    Y_ = torch.tensor(Y_, dtype = torch.float)
    # Output Correlation Matrix
    O_ = __get_correlation_matrix(N_tasks, scale)
    # Evaluate kernel function for training and testing
    K_ = _kernel(X_, X_, kernel, degree, gamma = gamma, beta = beta)
    # Extend Kernel Function and predictors for MO-KRR
    y_tilde_ = torch.flatten(Y_.t())
    K_tilde_ = torch.tensor(np.kron(O_, K_), dtype = torch.float)
    # MTKRR Least-Square Solution
    a_star_ = _robust_pinverse(K_tilde_ + alpha*torch.eye(K_tilde_.shape[0])) @ y_tilde_
    return [a_star_, X_, O_, N_tasks, [kernel, degree, gamma, beta]]

# Calculing prediction for new sample
def _MTKRR_predict(Y_, model_):
    def __predict(Y_, model_):
        N_star = Y_.shape[0]
        # Numpy to Torch
        Y_ = torch.tensor(Y_, dtype = torch.float)
        # Model parameters and hyperparameters
        a_star_, X_, O_, N_tasks, theta_ = model_
        kernel, degree, gamma, beta      = theta_
        # Evaluate kernel function for Testing
        K_star_     = _kernel(Y_, X_, kernel, degree, gamma = gamma, beta = beta)
        # Make a prediction
        y_star_hat_ = torch.tensor(np.kron(O_, K_star_), dtype = torch.float) @ a_star_
        # Torch to Numpy
        return np.squeeze(y_star_hat_.detach().numpy()).reshape(N_tasks, N_star).T
    # Model parameters and hyperparameters
    a_star_, X_, O_, N_tasks, theta_ = model_
    # Make Prediction in batches
    N_samples_per_batch = 2500
    N_batches           = math.ceil(Y_.shape[0] / N_samples_per_batch)
    Y_hat_              = np.zeros((Y_.shape[0], N_tasks))
    for i in range(N_batches):
        Y_batch_ = Y_[i*N_samples_per_batch:(i + 1)*N_samples_per_batch, :]
        Y_hat_[i*N_samples_per_batch:(i + 1)*N_samples_per_batch, :] = __predict(Y_batch_, model_)
    return Y_hat_

# Calculing prediction for new sample
def _get_MTKRR_prediction(data_, theta_, kernel, degree):
    N       = data_[0][1][0].shape[0]
    D       = data_[0][1][0].shape[1]
    N_tasks = data_[0][1][1].shape[1]
    t_tr = 0.
    t_ts = 0.
    # Loop over independet outputs
    x_tr_, y_tr_ = data_[0][0]
    x_ts_, y_ts_ = data_[0][1]
    # Training Model
    t_init = time.time()
    _MTKRR = _MTKRR_fit(x_tr_, y_tr_, kernel, degree, alpha = theta_[0],
                                                      gamma = theta_[1],
                                                      beta  = theta_[2],
                                                      scale = theta_[3])
    t_tr   = time.time() - t_init
    # Testing Model
    t_init = time.time()
    Y_hat_ = _MTKRR_predict(x_ts_, _MTKRR)
    t_ts   = time.time() - t_init
    return Y_hat_, [t_tr, t_ts], _MTKRR

def _KFold_CV(data_, theta_, kernel, degree, n_kfolds = 3):
    N_tasks = data_[0][1].shape[1]
    e_      = np.zeros((n_kfolds, N_tasks))
    j       = 0
    for idx_val_tr_, idx_val_ts_ in KFold(n_splits     = n_kfolds,
                                          random_state = None,
                                          shuffle      = False).split(data_[0][0]):
        data_val_   = []
        scaler_val_ = []
        # Validation training and testing sets
        X_val_tr_ = data_[0][0][idx_val_tr_, :]
        Y_val_tr_ = data_[0][1][idx_val_tr_, :]
        X_val_ts_ = data_[0][0][idx_val_ts_, :]
        Y_val_ts_ = data_[0][1][idx_val_ts_, :]
        # Get Outliers Index
        outliers_idx_val_tr_ = _get_outliers_index(X_val_tr_, n_samples = 1250)
        X_val_tr_ = X_val_tr_[outliers_idx_val_tr_, :]
        Y_val_tr_ = Y_val_tr_[outliers_idx_val_tr_, :]
        # Define Data Standarization
        _scaler_x = StandardScaler().fit(X_val_tr_)
        _scaler_y = StandardScaler().fit(Y_val_tr_)
        # Performe Data Standarization
        X_val_tr_prime_ = _scaler_x.transform(X_val_tr_)
        Y_val_tr_prime_ = _scaler_y.transform(Y_val_tr_)
        X_val_ts_prime_ = _scaler_x.transform(X_val_ts_)
        print(j, X_val_tr_prime_.shape, Y_val_tr_prime_.shape, X_val_ts_prime_.shape, Y_val_ts_.shape)
        # Save Dataset
        data_val_.append([[X_val_tr_prime_, Y_val_tr_prime_], [X_val_ts_prime_, Y_val_ts_]])
        scaler_val_.append([_scaler_x, _scaler_y])
        #try:
        # Train and predict MTKRR
        Y_val_ts_hat_prime_ = _get_MTKRR_prediction(data_val_, theta_, kernel, degree)[0]
        # Undo Normalization of the prediction
        Y_val_ts_hat_ = scaler_val_[0][1].inverse_transform(Y_val_ts_hat_prime_)
        Y_val_ts_     = data_val_[0][1][1]
        e_[j, :]      = root_mean_squared_error(Y_val_ts_, Y_val_ts_hat_)
        print(e_[j, :])
        #Compute Validation error metrics
        # except:
        #     print('ERROR!')
        #     print(theta_)
        #     e_[j] = 1e10
        j += 1
    return np.mean(e_, axis = 0)

# MTKRR K-Fold Cross-Validation of the model Parameters
def _get_KRR_cross_validation(data_, kernel, degree, n_grid, n_kfolds):
    # Define MTKRR parameters to validate
    alpha_ = np.logspace(-7, 7, n_grid)
    gamma_ = np.array((0., ))
    beta_  = np.array((0., ))
    scale_ = np.logspace(-2, 2, n_grid)
    if kernel == 'linear':
        gamma_ = np.logspace(-4, 0, n_grid)
        theta_ = np.meshgrid(alpha_, gamma_, beta_, scale_)
    if kernel == 'RBF':
        gamma_ = np.logspace(-4, 2, n_grid)
        theta_ = np.meshgrid(alpha_, gamma_, beta_, scale_)
    if kernel == 'matern':
        gamma_ = np.logspace(-4, 2, n_grid)
        theta_ = np.meshgrid(alpha_, gamma_, beta_, scale_)
    if kernel == 'poly':
        gamma_ = np.logspace(-4, 2, n_grid)
        beta_  = np.logspace(-1, 2, n_grid)
        theta_ = np.meshgrid(alpha_, gamma_, beta_, scale_)
    if kernel == 'RQ':
        gamma_ = np.logspace(-4, 2, n_grid)
        beta_  = np.logspace(-1, 2, n_grid)
        theta_ = np.meshgrid(alpha_, gamma_, beta_, scale_)
    # Constants Initialization
    theta_  = np.array(theta_).T.reshape(-1, len(theta_))
    N, D    = data_[0][0].shape
    N_tasks = data_[0][1].shape[1]
    M, P    = theta_.shape
    e_      = np.zeros((M, N_tasks))
    print(N, D, N_tasks, M, P)
    # Loop over combinations of parameters
    for i in np.arange(M, dtype = int):
        # Kfold Cross-validation Implementation
        e_[i, :] = _KFold_CV(data_, theta_[i, :], kernel, degree, n_kfolds)
    # Find Best Model
    i_ = np.argmin(np.mean(e_, axis = 1), axis = 0)
    return e_[i_, :], theta_[i_, :]

# MTKRR Model validation without kernels
def _meta_KRR_cross_validation(data_, kernel, degree, n_grid, n_kfolds):
    # eSVM Parameters Cross-validation
    e_val_, theta_ = _get_KRR_cross_validation(data_, kernel, degree, n_grid, n_kfolds)
    return e_val_, theta_

# Model Traning and Testing
def _KRR_traing_and_testing(data_, theta_, kernel, degree):
    N_tasks  = data_[0][0][1].shape[1]
    dataset_ = []
    scaler_  = []
    # Validation training and testing sets
    X_tr_ = data_[0][0][0]
    Y_tr_ = data_[0][0][1]
    X_ts_ = data_[0][1][0]
    Y_ts_ = data_[0][1][1]
    # Define Data Standarization
    _scaler_x = StandardScaler().fit(X_tr_)
    _scaler_y = StandardScaler().fit(Y_tr_)
    # Performe Data Standarization
    X_tr_prime_ = _scaler_x.transform(X_tr_)
    Y_tr_prime_ = _scaler_y.transform(Y_tr_)
    X_ts_prime_ = _scaler_x.transform(X_ts_)
    print(X_tr_prime_.shape, Y_tr_prime_.shape, X_ts_prime_.shape, Y_ts_.shape)
    # Save Dataset
    dataset_.append([[X_tr_prime_, Y_tr_prime_], [X_ts_prime_, Y_ts_]])
    scaler_.append([_scaler_x, _scaler_y])
    #try:
    # Train and predict MTKRR
    Y_ts_hat_prime_, time_, _MTKRR = _get_MTKRR_prediction(dataset_, theta_, kernel, degree)
    print(Y_ts_hat_prime_.shape)
    # Undo Normalization of the prediction
    Y_ts_hat_ = scaler_[0][1].inverse_transform(Y_ts_hat_prime_)
    Y_ts_     = dataset_[0][1][1]
    e_ts_     = mean_absolute_percentage_error(Y_ts_, Y_ts_hat_)
    #Compute Validation error metrics
    # except:
    #     print('ERROR!')
    #     print(theta_)
    #     e_[j] = 1e10
    return e_ts_, Y_ts_hat_, time_, [_MTKRR, scaler_]

def _get_covariates(i_cov = None, i_task = None):
    # CSI = 0 // PYRA = 2
    idx_pred = 0
    idx_pred_horizon_ = [0, 1, 2, 3, 4, 5]
    # Dataset Covariantes and Predictors Definition
    if i_task == 'persistence': return [idx_pred, idx_pred_horizon_, [0], [], 0, 0, [], [0, 1, 2, 3, 4, 5], []]
    # CSI = 0 // PYRA = 2
    idx_pred = 0
    idx_pred_horizon_ = [0, 1, 2, 3, 4, 5]
    idx_cov_horizon_  = [0, 1, 2, 3, 4, 5]
    # Cross-validation of AR
    cov_idx_0_ = [idx_pred, idx_pred_horizon_, [0, 1, 2, 3, 4, 5], [], 0, 0, [], idx_cov_horizon_, []]
    # Cross-validation of AR + Angles
    cov_idx_1_ = [idx_pred, idx_pred_horizon_, [0, 1, 2, 3, 4, 5], [0, 1], 0, 0, [], idx_cov_horizon_, []]
    # Cross-validation of AR + Angles + Raw Temperatures
    cov_idx_2_ = [idx_pred, idx_pred_horizon_, [0, 1, 2, 3, 4, 5], [0, 1], 0, 0, [0, 1], idx_cov_horizon_, [0]]
    # Cross-validation of AR + Angles + Processed Temperatures
    cov_idx_3_ = [idx_pred, idx_pred_horizon_, [0, 1, 2, 3, 4, 5], [0, 1], 3, 0, [0, 1], idx_cov_horizon_, [0]]
    # Cross-validation of AR + Angles + Processed Heights
    cov_idx_4_ = [idx_pred, idx_pred_horizon_, [0, 1, 2, 3, 4, 5], [0, 1], 0, 2, [0, 1], idx_cov_horizon_, [1]]
    # Cross-validation of AR + Angles + Raw Temperatures + Processed Heights
    cov_idx_5_ = [idx_pred, idx_pred_horizon_, [0, 1, 2, 3, 4, 5], [0, 1], 0, 2, [0, 1], idx_cov_horizon_, [0, 1]]
    # Cross-validation of AR + Angles + Raw Temperatures + Processed Heights + Magnitude
    cov_idx_6_ = [idx_pred, idx_pred_horizon_, [0, 1, 2, 3, 4, 5], [0, 1], 0, 2, [0, 1], idx_cov_horizon_, [0, 1, 2]]
    # Cross-validation of AR + Angles + Raw Temperatures + Processed Heights + Magnitude + Divergence
    cov_idx_7_ = [idx_pred, idx_pred_horizon_, [0, 1, 2, 3, 4, 5], [0, 1], 0, 2, [0, 1], idx_cov_horizon_, [0, 1, 2, 4]]
    # Cross-validation of AR + Angles + Raw Temperatures + Processed Heights + Magnitude + Divergence + Vorticity
    cov_idx_8_ = [idx_pred, idx_pred_horizon_, [0, 1, 2, 3, 4, 5], [0, 1], 0, 2, [0, 1], idx_cov_horizon_, [0, 1, 2, 3, 4]]
    # Index of all Covariances
    return [cov_idx_0_,  cov_idx_1_,  cov_idx_2_,  cov_idx_3_,  cov_idx_4_,  cov_idx_5_, cov_idx_6_, cov_idx_7_, cov_idx_8_][i_cov]

# Training and testing without shuffling the samples
def _split_dataset(X_, Y_, Z_, idx_tr_, idx_ts_):
    return X_[idx_tr_, :], Y_[idx_tr_, :], Z_[idx_tr_, :], X_[idx_ts_, :], Y_[idx_ts_, :], Z_[idx_ts_, :]

# Add this day samples to the training dataset
def _add_samples_to_training_dataset(idx_tr_, idx_ts_, labels_idx_tr_, labels_idx_ts_, day_idx_ts_):
    # Add index to the training set
    idx_tr_prime_ = np.concatenate((idx_tr_, idx_ts_[day_idx_ts_]), axis = 0)
    labels_idx_tr_prime_ = labels_idx_tr_.copy()
    # Add index to the labels traning set
    for i_label in range(len(labels_idx_tr_)):
        labels_idx_tr_prime_[i_label] = np.concatenate((labels_idx_tr_[i_label],
                                                        labels_idx_ts_[i_label][day_idx_ts_]), axis = 0)
    return idx_tr_prime_, labels_idx_tr_prime_

# Local Outlier Factor Indexes
def _get_outliers_index(X_, n_samples, n_neighbors = 3):
    _LOF = LocalOutlierFactor(n_neighbors = n_neighbors)
    _LOF.fit_predict(X_)
    p_val_tr_ = _LOF.negative_outlier_factor_
    idx_      = np.zeros(p_val_tr_.shape[0], dtype = bool)
    idx_[np.argsort(p_val_tr_)[-n_samples:]] = True
    return idx_

# Nodes and jobs information for communication from MPI
i_ker   = int(sys.argv[1])
i_cov   = int(sys.argv[2])
i_sec   = 0
i_nor   = 1
i_label = int(sys.argv[3])
# Get Experiment for the i-th Job
kernel, degree = _get_experiment(i_ker)
print(i_ker, i_cov, i_sec, i_nor, kernel, degree)
# Load Dataset
dataset_  = pickle.load(open('/users/terren/solar_forecasting/data/dataset_v31-1.pkl','rb'))
# Index of training samples with no detected clouds
idx_0_tr_ = pickle.load(open('/users/terren/solar_forecasting/data/clear_sky_index_v31-1.pkl', 'rb'))
# Load Weather Features
W_tr_, W_ts_ = pickle.load(open('/users/terren/solar_forecasting/data/weather_features_v31-1.pkl','rb'))
# Load Training and Testing indexes
idx_tr_, idx_ts_ = pickle.load(open('/users/terren/solar_forecasting/data/training_testing_index_v31-1.pkl','rb'))
# Load Persistent Pyranometer and Clear Sky Index
P_ts_, P_ts_hat_persistence_ = pickle.load(open('/users/terren/solar_forecasting/data/pyra_persistence_v31-1.pkl','rb'))
C_ts_, C_ts_hat_persistence_ = pickle.load(open('/users/terren/solar_forecasting/data/csi_persistence_v31-1.pkl','rb'))
# Load Atmospheric condition label indexes
labels_idx_tr_, labels_idx_ts_ = pickle.load(open('/users/terren/solar_forecasting/data/labels_index_v31-1.pkl','rb'))


N_tasks   = 6
data_val_ = []
# Generate database
X_, Y_, Z_ = _generate_database(dataset_, cov_idx_  = _get_covariates(i_cov))
# Traning and Testing Dataset
X_tr_, Y_tr_, Z_tr_, X_ts_, Y_ts_, Z_ts_ = _split_dataset(X_, Y_, Z_, idx_tr_, idx_ts_)
# Get Traning Index with only Clear Sky days when label is of a clear sky day
idx_val_tr_ = labels_idx_tr_[i_label]
# Get Validation Dataset
X_tr_ = X_tr_[idx_val_tr_, :]
Y_tr_ = Y_tr_[idx_val_tr_, :]
print(X_tr_.shape, Y_tr_.shape)
# Save Dataset
data_val_.append([X_tr_, Y_tr_])

# Cross-Validate Kernel Learning Model
e_val_machine_, theta_ = _meta_KRR_cross_validation(data_val_, kernel, degree, n_grid   = 3,
                                                                               n_kfolds = 3)
print(e_val_machine_)
print(theta_)

data_ = []
# Generate database
X_, Y_, Z_ = _generate_database(dataset_, cov_idx_  = _get_covariates(i_cov))
# Split in Training and testing Dataset
X_tr_, Y_tr_, Z_tr_, X_ts_, Y_ts_, Z_ts_ = _split_dataset(X_, Y_, Z_, idx_tr_, idx_ts_)
# Get Traning Index with only Clear Sky days when label is of a clear sky day
idx_tr_prime_ = labels_idx_tr_[i_label]
idx_ts_prime_ = labels_idx_ts_[i_label]
# Get what is Not Outliers Index
outliers_idx_tr_prime_ = _get_outliers_index(X_tr_[idx_tr_prime_, :], n_samples = 2500)
# Select Training and Testing data
X_tr_ = X_tr_[idx_tr_prime_, :][outliers_idx_tr_prime_, :]
Y_tr_ = Y_tr_[idx_tr_prime_, :][outliers_idx_tr_prime_, :]
X_ts_ = X_ts_[idx_ts_prime_, :]
Y_ts_ = Y_ts_[idx_ts_prime_, :]
print(X_tr_.shape, Y_tr_.shape, X_ts_.shape, Y_ts_.shape)
data_.append([[X_tr_, Y_tr_], [X_ts_, Y_ts_]])

# Training and Testing of the Cross-Validate Kernel Learning Model
e_ts_machine_, Y_ts_hat_, time_, _model = _KRR_traing_and_testing(data_, theta_, kernel, degree)
print(e_ts_machine_)
print(time_)

# Define directory Roor
root = '/users/terren/solar_forecasting'
# Save Errors
name = r'{}/logs/kernel_learning/KRRs/CV-MTKRR_v31-1_{}.csv'.format(root, i_label)
with open(name, 'a', newline = '\n') as f:
    writer = csv.writer(f)
    writer.writerow([i_ker, i_cov, i_sec, i_nor] + time_ + e_val_machine_.tolist() + e_ts_machine_.tolist())
# Save Results
name = r'{}/data/kernel_learning/KRRs/CV-MTKRR_v31-1_{}{}{}{}-{}.pkl'.format(root, i_ker, i_cov, i_sec, i_nor, i_label)
with open(name, 'wb') as handle:
    pickle.dump(Y_ts_hat_, handle, protocol = pickle.HIGHEST_PROTOCOL)
# Save Models
name = r'{}/model/kernel_learning/KRRs/CV-MTKRR_v31-1_{}{}{}{}-{}.pkl'.format(root, i_ker, i_cov, i_sec, i_nor, i_label)
with open(name, 'wb') as handle:
    pickle.dump(_model, handle, protocol = pickle.HIGHEST_PROTOCOL)
