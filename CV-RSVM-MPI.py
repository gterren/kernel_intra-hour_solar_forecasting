import pickle, glob, sys, os, warnings, csv, gpytorch, torch
import numpy as np

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
            # Compute Linear Kernel
            return gamma*X @ Y.T
        if kernel is 'poly':
            # Compute Polynomial Kernel
            return (gamma*X @ Y.T + beta)**degree
        if kernel is 'RBF':
            # Compute Radial Basis Function Kernel
            return np.exp(-gamma*euclidean_distances(X, Y)**2)
        if kernel is 'RQ':
            # Compute Rational Quadratic Kernel
            return (1. + (gamma*euclidean_distances(X, Y)**2)/(2./beta) )**(-beta)
        if kernel is 'matern':
            D_ = np.sqrt(2.*degree)*(gamma*euclidean_distances(X, Y)**2)
            D_[D_ == 0.0] += np.finfo(float).eps
            # This function is not implemented in torch
            A_ = ss.kn(degree, D_)
            # Compute Matern Kernel
            return ( (2**(1 - degree) )/ ss.gamma(degree)) * ( D_**degree ) * A_

    return __K(X_, Y_, kernel, degree, gamma, beta)

# Epsilon-SVM single output for regression
def _SVM_fit(X_, y_, kernel, degree, C, epsilon, gamma = None, beta = None, scale = None):
    # Define Kernel
    K_ = _kernel(X_, X_, kernel, degree, beta = beta, gamma = gamma)
    # Fit model
    _SVM  = SVR(kernel = 'precomputed', C = C, epsilon = epsilon).fit(K_, y_)
    # Compute Bias
    bias_ = y_[_SVM.support_] - K_[_SVM.support_, _SVM.support_] @ _SVM.dual_coef_.T - epsilon
    b     = np.mean(bias_)
    # Save Model Parameters
    theta_ = [C, epsilon, kernel, degree, gamma, beta]
    return [_SVM.dual_coef_, _SVM.intercept_, _SVM.support_, X_, theta_]

# Calculing prediction for a new sample
def _SVM_predict(Y_, model_):
    # Model parameters and hyperparameters
    alpha_, b, idx_SVs_, X_, theta_         = model_
    C, epsilon, kernel, degree, gamma, beta = theta_
    # Evaluate kernel function for Testing
    K_star_ = _kernel(Y_, X_, kernel, degree, gamma = gamma, beta = beta)
    # Make a prediction
    y_hat_  = K_star_[:, idx_SVs_] @ alpha_.T + b
    return np.squeeze(y_hat_)

# Support Vector Machine Chain
def _get_SVM_prediction_chain(data_, theta_, kernel, degree):
    N_tr            = data_[0][0][0].shape[0]
    N_ts            = data_[0][1][0].shape[0]
    N_tasks         = len(data_)
    y_recurrent_tr_ = np.empty((N_tr, 0))
    y_recurrent_ts_ = np.empty((N_ts, 0))
    y_ts_hat_       = np.zeros((N_ts, N_tasks))
    t_tr            = 0.
    t_ts            = 0.
    _model          = []
    # Loop over independet outputs
    for i_task in range(N_tasks):
        x_tr_, y_tr_ = data_[i_task][0]
        x_ts_, y_ts_ = data_[i_task][1]
        # Cross-validation or testing
        t_init = time.time()
        x_tr_ = np.concatenate((x_tr_, y_recurrent_tr_), axis = 1)
        x_ts_ = np.concatenate((x_ts_, y_recurrent_ts_), axis = 1)
        if theta_.shape[0] == N_tasks:
            # Define Model
            _SVM = _SVM_fit(x_tr_, y_tr_, kernel  = kernel,
                                          degree  = degree,
                                          C       = theta_[i_task, 0],
                                          epsilon = theta_[i_task, 1],
                                          gamma   = theta_[i_task, 2],
                                          beta    = theta_[i_task, 3])
        else:
            # Define Model
            _SVM = _SVM_fit(x_tr_, y_tr_, kernel  = kernel,
                                          degree  = degree,
                                          C       = theta_[0],
                                          epsilon = theta_[1],
                                          gamma   = theta_[2],
                                          beta    = theta_[3])
        t_tr += time.time() - t_init
        # Testing Model
        t_init               = time.time()
        y_ts_hat_[:, i_task] = _SVM_predict(x_ts_, _SVM)
        t_ts                += time.time() - t_init
        # Update Covariate for Nex Chain prediction
        y_recurrent_tr_ = np.concatenate((y_recurrent_tr_, y_tr_), axis = 1)
        y_recurrent_ts_ = np.concatenate((y_recurrent_ts_, y_ts_hat_[:, i_task][:, np.newaxis]), axis = 1)
        _model.append([_SVM, theta_])
    return y_ts_hat_, [t_tr, t_ts], _SVM


def _KFold_CV(data_, theta_, kernel, degree, n_kfolds):
    N_tasks = len(data_)
    e_ = np.zeros((n_kfolds, N_tasks))
    j  = 0
    for idx_val_tr_, idx_val_ts_ in KFold(n_splits     = n_kfolds,
                                          random_state = None,
                                          shuffle      = False).split(data_[0][0]):
        data_val_   = []
        scaler_val_ =[]
        for i_task in range(N_tasks):
            # Validation training and testing sets
            X_val_tr_ = data_[i_task][0][idx_val_tr_, :]
            Y_val_tr_ = data_[i_task][1][idx_val_tr_, :]
            X_val_ts_ = data_[i_task][0][idx_val_ts_, :]
            Y_val_ts_ = data_[i_task][1][idx_val_ts_, :]
            # Get Outliers Index
            outliers_idx_val_tr_ = _get_outliers_index(X_val_tr_, n_samples = 2300)
            X_val_tr_ = X_val_tr_[outliers_idx_val_tr_, :]
            Y_val_tr_ = Y_val_tr_[outliers_idx_val_tr_, :]
            # Define Data Standarization
            _scaler_x = StandardScaler().fit(X_val_tr_)
            _scaler_y = StandardScaler().fit(Y_val_tr_)
            # Performe Data Standarization
            X_val_tr_prime_ = _scaler_x.transform(X_val_tr_)
            Y_val_tr_prime_ = _scaler_y.transform(Y_val_tr_)
            X_val_ts_prime_ = _scaler_x.transform(X_val_ts_)
            print(j, i_task, X_val_tr_prime_.shape, Y_val_tr_prime_.shape, X_val_ts_prime_.shape, Y_val_ts_.shape)
            # Save Dataset
            data_val_.append([[X_val_tr_prime_, Y_val_tr_prime_], [X_val_ts_prime_, Y_val_ts_]])
            scaler_val_.append([_scaler_x, _scaler_y])

        #try:
        # Training and testion SVM
        Y_val_ts_hat_prime_ = _get_SVM_prediction_chain(data_val_, theta_, kernel, degree)[0]
        # Undo Normalization of the prediction
        Y_val_ts_hat_ = np.zeros(Y_val_ts_hat_prime_.shape)
        Y_val_ts_     = np.zeros(Y_val_ts_hat_prime_.shape)
        for i_task in range(N_tasks):
            Y_val_ts_hat_[:, i_task] = scaler_val_[i_task][1].inverse_transform(Y_val_ts_hat_prime_[:, i_task][:, np.newaxis])[:, 0]
            Y_val_ts_[:, i_task]     = data_val_[i_task][1][1][:, 0]
        e_[j, :] = mean_absolute_percentage_error(Y_val_ts_, Y_val_ts_hat_)
        print(e_[j, :])
        #Compute Validation error metrics
        # except:
        #     print('ERROR!')
        #     print(theta_)
        #     e_[j] = 1e10
        j += 1
    return np.mean(e_, axis = 0)

# eSVM K-Fold Cross-Validation of the model Parameters
def _get_SVM_cross_validation(data_, kernel, degree, n_grid, n_kfolds):
    _comm.Barrier()
    # Define eSVM parameters to validate
    C_       = np.logspace(-3, 0, n_grid)
    epsilon_ = np.logspace(-3, 0, n_grid)
    gamma_ = np.array((0., ))
    beta_  = np.array((0., ))
    scale_ = np.array((0., ))
    if kernel == 'linear':
        gamma_ = np.logspace(-4, 0, n_grid)
        theta_ = np.meshgrid(C_, epsilon_, gamma_, beta_, scale_)
    if kernel == 'RBF':
        gamma_ = np.logspace(-4, 2, n_grid)
        theta_ = np.meshgrid(C_, epsilon_, gamma_, beta_, scale_)
    if kernel == 'matern':
        gamma_ = np.logspace(-4, 2, n_grid)
        theta_ = np.meshgrid(C_, epsilon_, gamma_, beta_, scale_)
    if kernel == 'poly':
        gamma_ = np.logspace(-4, 2, n_grid)
        beta_  = np.logspace(-2, 2, n_grid)
        theta_ = np.meshgrid(C_, epsilon_, gamma_, beta_, scale_)
    if kernel == 'RQ':
        gamma_ = np.logspace(-4, 2, n_grid)
        beta_  = np.logspace(-1, 2, n_grid)
        theta_ = np.meshgrid(C_, epsilon_, gamma_, beta_, scale_)
    # Constants Initialization
    theta_  = np.array(theta_).T.reshape(-1, len(theta_))
    N, D    = data_[0][0].shape
    N_tasks = len(data_)
    M, P    = theta_.shape
    e_      = np.zeros((M, N_tasks))
    print(N, D, N_tasks, M, P)
    # Parallelization
    idx_   = np.arange(M, dtype = int)
    error_ = np.zeros((M, N_tasks))
    print(i_job, idx_[i_job::N_jobs])
    # Loop over combinations of parameters
    for i in idx_[i_job::N_jobs]:
        t = time.time()
        # Kfold Cross-validation Implementation
        e_[i, :] = _KFold_CV(data_, theta_[i, :], kernel, degree, n_kfolds)
        #print(i_job, i, time.time() - t)
    # Parallelization
    _comm.Barrier()
    _comm.Reduce(e_, error_, op = MPI.SUM, root = 0)
    print(i_job)
    if i_job == 0:
        # Find Best Model
        i_ = np.argmin(error_, axis = 0)
        return error_[i_, [0, 1, 2, 3, 4, 5]], theta_[i_, :]
    else:
        return None, None

# eSVM Model validation without kernels
def _meta_SVM_cross_validation(data_, kernel, degree, n_grid, n_kfolds):
    # eSVM Parameters Cross-validation
    e_val_, theta_ = _get_SVM_cross_validation(data_, kernel, degree, n_grid, n_kfolds)
    # Train and Test model with optimal parameters
    if i_job == 0:
        return e_val_, theta_
    else:
        return None, None

# Model Traning and Testing
def _SVM_traing_and_testing(data_, theta_, kernel, degree):
    N_tasks  = len(data_)
    dataset_ = []
    scaler_  = []
    # Loop over Task doing the stardarization
    for i_task in range(N_tasks):
        # Validation training and testing sets
        X_tr_ = data_[i_task][0][0]
        Y_tr_ = data_[i_task][0][1]
        X_ts_ = data_[i_task][1][0]
        Y_ts_ = data_[i_task][1][1]
        # Define Data Standarization
        _scaler_x = StandardScaler().fit(X_tr_)
        _scaler_y = StandardScaler().fit(Y_tr_)
        # Performe Data Standarization
        X_tr_prime_ = _scaler_x.transform(X_tr_)
        Y_tr_prime_ = _scaler_y.transform(Y_tr_)
        X_ts_prime_ = _scaler_x.transform(X_ts_)
        print(i_task, X_tr_prime_.shape, Y_tr_prime_.shape, X_ts_prime_.shape, Y_ts_.shape)
        # Save Dataset
        dataset_.append([[X_tr_prime_, Y_tr_prime_], [X_ts_prime_, Y_ts_]])
        scaler_.append([_scaler_x, _scaler_y])

    #try:
    # Training and testing SVM Chain
    Y_ts_hat_prime_, time_, _RSVM = _get_SVM_prediction_chain(dataset_, theta_, kernel, degree)
    print(Y_ts_hat_prime_.shape)
    # Undo Normalization of the prediction
    Y_ts_hat_ = np.zeros(Y_ts_hat_prime_.shape)
    Y_ts_     = np.zeros(Y_ts_hat_prime_.shape)
    # Loop over Task undoing the stardarization
    for i_task in range(N_tasks):
        Y_ts_hat_[:, i_task] = scaler_[i_task][1].inverse_transform(Y_ts_hat_prime_[:, i_task][:, np.newaxis])[:, 0]
        Y_ts_[:, i_task]     = dataset_[i_task][1][1][:, 0]
    e_ts_ = mean_absolute_percentage_error(Y_ts_, Y_ts_hat_)
    #Compute Validation error metrics
    # except:
    #     print('ERROR!')
    #     print(theta_)
    #     e_[j] = 1e10
    return e_ts_, Y_ts_hat_, time_, [_RSVM, scaler_]

def _get_covariates(i_task, i_cov = None, i_sec = None):
    # CSI = 0 // PYRA = 2
    idx_pred = 0
    idx_pred_horizon_ = [0, 1, 2, 3, 4, 5]
    # Dataset Covariantes and Predictors Definition
    if i_task == 'persistence': return [idx_pred, idx_pred_horizon_, [0], [], 0, 0, [], [0, 1, 2, 3, 4, 5], []]
    # CSI = 0 // PYRA = 2
    idx_pred = 0
    idx_pred_horizon_ = [i_task]
    # All
    if (i_task == 0) and (i_sec == 0): idx_cov_horizon_  = [0, 1, 2, 3, 4, 5]
    if (i_task == 1) and (i_sec == 0): idx_cov_horizon_  = [0, 1, 2, 3, 4, 5]
    if (i_task == 2) and (i_sec == 0): idx_cov_horizon_  = [0, 1, 2, 3, 4, 5]
    if (i_task == 3) and (i_sec == 0): idx_cov_horizon_  = [0, 1, 2, 3, 4, 5]
    if (i_task == 4) and (i_sec == 0): idx_cov_horizon_  = [0, 1, 2, 3, 4, 5]
    if (i_task == 5) and (i_sec == 0): idx_cov_horizon_  = [0, 1, 2, 3, 4, 5]
    # Neibors order 1
    if (i_task == 0) and (i_sec == 1): idx_cov_horizon_  = [0, 1, 2, 3]
    if (i_task == 1) and (i_sec == 1): idx_cov_horizon_  = [0, 1, 2, 3, 4]
    if (i_task == 2) and (i_sec == 1): idx_cov_horizon_  = [0, 1, 2, 3, 4, 5]
    if (i_task == 3) and (i_sec == 1): idx_cov_horizon_  = [0, 1, 2, 3, 4, 5]
    if (i_task == 4) and (i_sec == 1): idx_cov_horizon_  = [1, 2, 3, 4, 5]
    if (i_task == 5) and (i_sec == 1): idx_cov_horizon_  = [2, 3, 4, 5]
    # Neibors order 2
    if (i_task == 0) and (i_sec == 2): idx_cov_horizon_  = [0, 1, 2]
    if (i_task == 1) and (i_sec == 2): idx_cov_horizon_  = [0, 1, 2, 3]
    if (i_task == 2) and (i_sec == 2): idx_cov_horizon_  = [1, 2, 3, 4]
    if (i_task == 3) and (i_sec == 2): idx_cov_horizon_  = [1, 2, 3, 4]
    if (i_task == 4) and (i_sec == 2): idx_cov_horizon_  = [2, 3, 4, 5]
    if (i_task == 5) and (i_sec == 2): idx_cov_horizon_  = [3, 4, 5]
    # Neibors order 3
    if (i_task == 0) and (i_sec == 3): idx_cov_horizon_  = [0, 1]
    if (i_task == 1) and (i_sec == 3): idx_cov_horizon_  = [0, 1, 2]
    if (i_task == 2) and (i_sec == 3): idx_cov_horizon_  = [1, 2, 3]
    if (i_task == 3) and (i_sec == 3): idx_cov_horizon_  = [2, 3, 4]
    if (i_task == 4) and (i_sec == 3): idx_cov_horizon_  = [3, 4, 5]
    if (i_task == 5) and (i_sec == 3): idx_cov_horizon_  = [4, 5]
    # Neibors order 4
    if (i_task == 0) and (i_sec == 4): idx_cov_horizon_  = [0, 1, 2, 3]
    if (i_task == 1) and (i_sec == 4): idx_cov_horizon_  = [0, 1, 2]
    if (i_task == 2) and (i_sec == 4): idx_cov_horizon_  = [1, 2, 3]
    if (i_task == 3) and (i_sec == 4): idx_cov_horizon_  = [2, 3, 4]
    if (i_task == 4) and (i_sec == 4): idx_cov_horizon_  = [2, 3, 4, 5]
    if (i_task == 5) and (i_sec == 4): idx_cov_horizon_  = [4, 5]
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
i_job, N_jobs, _comm = _get_node_info(verbose = True)
i_ker   = int(sys.argv[1])
i_cov   = int(sys.argv[2])
i_sec   = int(sys.argv[3])
i_nor   = 1
i_label = int(sys.argv[4])
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
# Loop over forcasting Horizons
for i_task in range(N_tasks):
    # Generate database
    X_, Y_, Z_ = _generate_database(dataset_, cov_idx_  = _get_covariates(i_task, i_cov, i_sec))
    # Traning and Testing Dataset
    X_tr_, Y_tr_, Z_tr_, X_ts_, Y_ts_, Z_ts_ = _split_dataset(X_, Y_, Z_, idx_tr_, idx_ts_)
    # Get Traning Index with only Clear Sky days when label is of a clear sky day
    idx_val_tr_ = labels_idx_tr_[i_label]
    # Get Validation Dataset
    X_tr_ = X_tr_[idx_val_tr_, :]
    Y_tr_ = Y_tr_[idx_val_tr_, :]
    print(i_task, X_tr_.shape, Y_tr_.shape)
    # Save Dataset
    data_val_.append([X_tr_, Y_tr_])

# Cross-Validate Kernel Learning Model
e_val_machine_, theta_ = _meta_SVM_cross_validation(data_val_, kernel, degree, n_grid   = 3,
                                                                               n_kfolds = 3)
print(e_val_machine_)
print(theta_)

# Do no Paralize this part of the code
if i_job == 0:

    data_ = []
    # Loop over forcasting Horizons
    for i_task in range(N_tasks):
        # Generate database
        X_, Y_, Z_ = _generate_database(dataset_, cov_idx_  = _get_covariates(i_task, i_cov, i_sec))
        # Split in Training and testing Dataset
        X_tr_, Y_tr_, Z_tr_, X_ts_, Y_ts_, Z_ts_ = _split_dataset(X_, Y_, Z_, idx_tr_, idx_ts_)
        # Get Traning Index with only Clear Sky days when label is of a clear sky day
        idx_tr_prime_ = labels_idx_tr_[i_label]
        idx_ts_prime_ = labels_idx_ts_[i_label]
        # Get what is Not Outliers Index
        outliers_idx_tr_prime_ = _get_outliers_index(X_tr_[idx_tr_prime_, :], n_samples = 3500)
        # Select Training and Testing data
        X_tr_ = X_tr_[idx_tr_prime_, :][outliers_idx_tr_prime_, :]
        Y_tr_ = Y_tr_[idx_tr_prime_, :][outliers_idx_tr_prime_, :]
        X_ts_ = X_ts_[idx_ts_prime_, :]
        Y_ts_ = Y_ts_[idx_ts_prime_, :]
        print(X_tr_.shape, Y_tr_.shape, X_ts_.shape, Y_ts_.shape)
        data_.append([[X_tr_, Y_tr_], [X_ts_, Y_ts_]])

    # Training and Testing of the Cross-Validate Kernel Learning SVM Model
    e_ts_machine_, Y_ts_hat_, time_, _model = _SVM_traing_and_testing(data_, theta_, kernel, degree)
    print(e_ts_machine_)
    print(time_)

    # Define directory Roor
    root = '/users/terren/solar_forecasting'
    # Save Errors
    name = r'{}/logs/kernel_learning/SVMs/CV-RSVM_v31-1_{}.csv'.format(root, i_label)
    with open(name, 'a', newline = '\n') as f:
        writer = csv.writer(f)
        writer.writerow([i_ker, i_cov, i_sec, i_nor] + time_ + e_val_machine_.tolist() + e_ts_machine_.tolist())
    # Save Models
    name = r'{}/model/kernel_learning/SVMs/CV-RSVM_v31-1_{}{}{}{}-{}.pkl'.format(root, i_ker, i_cov, i_sec, i_nor, i_label)
    with open(name, 'wb') as handle:
        pickle.dump(_model, handle, protocol = pickle.HIGHEST_PROTOCOL)
    # Save Results
    name = r'{}/data/kernel_learning/SVMs/CV-RSVM_v31-1_{}{}{}{}-{}.pkl'.format(root, i_ker, i_cov, i_sec, i_nor, i_label)
    with open(name, 'wb') as handle:
        pickle.dump(Y_ts_hat_, handle, protocol = pickle.HIGHEST_PROTOCOL)
