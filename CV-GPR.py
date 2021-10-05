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
from machine_learning_utils import *

# Do not display warnings in the output file
warnings.filterwarnings('ignore')

# List of Experiments
def _get_experiment(i):
    exp_ = []
    for kernel, degree in zip(['linear', 'RBF', 'RQ', 'poly', 'matern', 'matern', 'matern'], [0, 0, 0, 2, 1./2., 3./2., 5./2.]):
        exp_.append([kernel, degree])
    return exp_[i]

# Gaussian Process Regression model fit...
def _GPR_fit(X_, y_, kernel, degree, num_dim, max_training_iter, early_stop, random_init):
    # Optimize Kernel hyperparameters
    def __optimize(_model, _likel, X_, y_, max_training_iter, early_stop):
        # Storage Variables Initialization
        nmll_ = []
        # Find optimal model hyperparameters
        _model.train()
        # Use the adam optimizer
        _optimizer = torch.optim.Adam(_model.parameters(), lr = .1)  # Includes GaussianLikelihood parameters
        # "Loss" for GPs - the marginal log likelihood
        _mll = gpytorch.mlls.ExactMarginalLogLikelihood(_likel, _model)
        # Begins Iterative Optimization
        for i in range(max_training_iter):
            # Zero gradients from previous iteration
            _optimizer.zero_grad()
            # Output from model
            f_hat_ = _model(X_)
            # Calc loss and backprop gradients
            _nmll = - _mll(f_hat_, y_)
            _nmll.backward()
            _optimizer.step()
            #print(i, np.around(float(_error.detach().numpy()), 2))
            nmll_.append(np.around(float(_nmll.detach().numpy()), 2) )
            if np.isnan(nmll_[-1]):
                return _model, _likel, np.inf
            if i > early_stop:
                if np.unique(nmll_[-early_stop:]).shape[0] == 1:
                    break
        return _model, _likel, nmll_[-1]
    X_tr_ = torch.tensor(X_, dtype = torch.float)
    y_tr_ = torch.tensor(y_, dtype = torch.float)
    # initialize likelihood and model
    _likel = gpytorch.likelihoods.GaussianLikelihood()
    _model = _GPR(X_tr_, y_tr_, _likel, kernel, degree, num_dim, random_init)
    return __optimize(_model, _likel, X_tr_, y_tr_, max_training_iter, early_stop)

# Select the best model using multiple initializations
def _model_selection(X_, y_, kernel, degree, num_dim, n_random_init, max_training_iter, early_stop):
    # Storage Variables Initialization
    model_ = []
    nmll_  = []
    # No Random Initialization
    _GPR, _likel, nmll = _GPR_fit(X_, y_, kernel, degree, num_dim, max_training_iter, early_stop, random_init = False)
    # Get Results
    model_.append([_GPR, _likel])
    nmll_.append(nmll)
    # Perform multiple Random Initializations
    for i in range(n_random_init):
        _GPR, _likel, nmll = _GPR_fit(X_, y_, kernel, degree, num_dim, max_training_iter, early_stop, random_init = True)
        # Get Results
        model_.append([_GPR, _likel])
        nmll_.append(nmll)
    # Best Results of all different Initialization
    _GPR, _likel = model_[np.argmin(nmll_)]
    nmll         = nmll_[np.argmin(nmll_)]
    return _GPR, _likel, nmll

# Calculing prediction for new sample
def _GPR_predict(_model, _likel, X_):
    X_ts_ = torch.tensor(X_, dtype = torch.float)
    _model.eval()
    _likel.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        _f_hat = _likel(_model(X_ts_))
        return _f_hat.mean.numpy(), _f_hat.variance.numpy()

# Gaussian Process Regression
def _get_GPR_prediction(data_, kernel, degree):
    N           = data_[0][1][0].shape[0]
    N_tasks     = len(data_)
    Y_ts_hat_   = np.zeros((N, N_tasks))
    S2p_ts_hat_ = np.zeros((N, N_tasks))
    s2n_ts_hat_ = np.zeros((N_tasks))
    nmll_ts_    = np.zeros((N_tasks))
    t_tr        = 0.
    t_ts        = 0.
    model_      = []
    # Loop over independet outputs
    for i_task in range(N_tasks):
        X_tr_, y_tr_ = data_[i_task][0]
        X_ts_, y_ts_ = data_[i_task][1]
        # Add Bias
        X_tr_ = np.concatenate((X_tr_, np.ones((X_tr_.shape[0], 1))), axis = 1)
        X_ts_ = np.concatenate((X_ts_, np.ones((X_ts_.shape[0], 1))), axis = 1)
        # Training Model
        t_init                         = time.time()
        _GPR, _likel, nmll_ts_[i_task] = _model_selection(X_tr_, y_tr_[:, 0], kernel, degree, num_dim           = X_tr_.shape[1],
                                                                                              n_random_init     = 4,
                                                                                              max_training_iter = 200,
                                                                                              early_stop        = 3)
        t_tr                          += time.time() - t_init
        # Testing Model
        t_init                                       = time.time()
        Y_ts_hat_[:, i_task], S2p_ts_hat_[:, i_task] = _GPR_predict(_GPR, _likel, X_ts_)
        s2n_ts_hat_[i_task]                          = _likel.noise
        t_ts                                        += time.time() - t_init
        model_.append([_GPR, _likel])
    return nmll_ts_, Y_ts_hat_, S2p_ts_hat_, s2n_ts_hat_, [t_tr, t_ts], model_

# Implementation of K-fold cross-validation
def _KFold_CV(data_, kernel, degree, n_kfolds):
    N_tasks = len(data_)
    E_      = np.zeros((n_kfolds, N_tasks))
    NMLL_   = np.zeros((n_kfolds, N_tasks))
    j       = 0
    for idx_val_tr_, idx_val_ts_ in KFold(n_splits     = n_kfolds,
                                          random_state = None,
                                          shuffle      = False).split(data_[0][0]):
        data_val_   = []
        scaler_val_ = []
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
            Y_val_tr_prime_ = Y_val_tr_
            #Y_val_tr_prime_ = _scaler_y.transform(Y_val_tr_)
            X_val_ts_prime_ = _scaler_x.transform(X_val_ts_)
            print(j, i_task, X_val_tr_prime_.shape, Y_val_tr_prime_.shape, X_val_ts_prime_.shape, Y_val_ts_.shape)
            # Save Dataset
            data_val_.append([[X_val_tr_prime_, Y_val_tr_prime_], [X_val_ts_prime_, Y_val_ts_]])
            scaler_val_.append([_scaler_x, _scaler_y])
        #try:
        # Training and Testing GPR
        nmll_ts_, Y_val_ts_hat_prime_ = _get_GPR_prediction(data_val_, kernel, degree)[:2]
        # Undo Normalization of the prediction
        Y_val_ts_hat_ = np.zeros(Y_val_ts_hat_prime_.shape)
        Y_val_ts_     = np.zeros(Y_val_ts_hat_prime_.shape)
        for i_task in range(N_tasks):
            #Y_val_ts_hat_[:, i_task] = scaler_val_[i_task][1].inverse_transform(Y_val_ts_hat_prime_[:, i_task][:, np.newaxis])[:, 0]
            Y_val_ts_hat_[:, i_task] = Y_val_ts_hat_prime_[:, i_task]
            Y_val_ts_[:, i_task]     = data_val_[i_task][1][1][:, 0]
        E_[j, :]    = mean_absolute_percentage_error(Y_val_ts_, Y_val_ts_hat_)
        NMLL_[j, :] = nmll_ts_
        print(E_[j, :])
        print(NMLL_[j, :])
        #Compute Validation error metrics
        # except:
        #     print('ERROR!')
        #     print(theta_)
        #     e_[j] = 1e10
        j += 1
    return np.mean(E_, axis = 0), np.mean(NMLL_, axis = 0)

# GPR K-Fold Cross-Validation of the model Parameters
def _get_GPR_cross_validation(data_, kernel, degree, n_kfolds):
    # Kfold Cross-validation Implementation
    e_val_, nmll_val_ = _KFold_CV(data_, kernel, degree, n_kfolds)
    return e_val_, nmll_val_

# GPR Model validation without kernels
def _meta_GPR_cross_validation(data_, kernel, degree, n_kfolds):
    # RVM Parameters Cross-validation
    e_val_, nmll_val_ = _get_GPR_cross_validation(data_, kernel, degree, n_kfolds)
    return e_val_, nmll_val_

# Model Traning and Testing
def _GPR_traing_and_testing(data_, kernel, degree):
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
        Y_tr_prime_ = Y_tr_
        #Y_tr_prime_ = _scaler_y.transform(Y_tr_)
        X_ts_prime_ = _scaler_x.transform(X_ts_)
        print(i_task, X_tr_prime_.shape, Y_tr_prime_.shape, X_ts_prime_.shape, Y_ts_.shape)
        # Save Dataset
        dataset_.append([[X_tr_prime_, Y_tr_prime_], [X_ts_prime_, Y_ts_]])
        scaler_.append([_scaler_x, _scaler_y])
    #try:
    # Training and testing GPR
    nmll_ts_, Y_ts_hat_prime_, S2p_ts_hat_prime_, s2n_ts_hat_prime_, time_, _GPR = _get_GPR_prediction(dataset_, kernel, degree)
    # Undo Normalization of the prediction
    Y_ts_hat_  = np.zeros(Y_ts_hat_prime_.shape)
    Sp_ts_hat_ = np.zeros(S2p_ts_hat_prime_.shape)
    sn_ts_hat_ = np.zeros(s2n_ts_hat_prime_.shape)
    Y_ts_      = np.zeros(Y_ts_hat_prime_.shape)
    # Loop over Task undoing the stardarization
    for i_task in range(N_tasks):
        Y_ts_hat_[:, i_task]  = Y_ts_hat_prime_[:, i_task]
        Sp_ts_hat_[:, i_task] = S2p_ts_hat_prime_[:, i_task]
        sn_ts_hat_[i_task]    = s2n_ts_hat_prime_[i_task]
        #Y_ts_hat_[:, i_task]  = scaler_[i_task][1].inverse_transform(Y_ts_hat_prime_[:, i_task][:, np.newaxis])[:, 0]
        #Sp_ts_hat_[:, i_task] = np.sqrt(S2p_ts_hat_prime_[:, i_task])*scaler_[i_task][1].scale_
        #sn_ts_hat_[i_task]    = np.sqrt(s2n_ts_hat_prime_[i_task])*scaler_[i_task][1].scale_
        Y_ts_[:, i_task]      = dataset_[i_task][1][1][:, 0]
    e_ts_ = mean_absolute_percentage_error(Y_ts_, Y_ts_hat_)
    #Compute Validation error metrics
    # except:
    #     print('ERROR!')
    #     print(theta_)
    #     e_[j] = 1e10
    return e_ts_, nmll_ts_, Y_ts_hat_, Sp_ts_hat_, sn_ts_hat_, time_, [_GPR, scaler_]

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
    X_, Y_, Z_ = _generate_database(dataset_, cov_idx_ = _get_covariates(i_task, i_cov, i_sec))
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
e_val_machine_, nmll_val_machine_ = _meta_GPR_cross_validation(data_val_, kernel, degree, n_kfolds = 3)
print(e_val_machine_)
print(nmll_val_machine_)

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

# Training and Testing of the Cross-Validate Kernel Learning Model
e_ts_machine_, nmll_ts_machine_, Y_ts_hat_, Sp_ts_hat_, sn_ts_hat_, time_, _model = _GPR_traing_and_testing(data_, kernel, degree)
print(e_ts_machine_)
print(nmll_ts_machine_)

# Define directory Roor
root = '/users/terren/solar_forecasting'
# Save Errors
name = r'{}/logs/kernel_learning/GPRs/CV-GPR_S0_v31-1_{}.csv'.format(root, i_label)
with open(name, 'a', newline = '\n') as f:
    writer = csv.writer(f)
    writer.writerow([i_ker, i_cov, i_sec, i_nor] + time_ + e_val_machine_.tolist() + e_ts_machine_.tolist())
# Save Errors
name = r'{}/logs/kernel_learning/GPRs/NMLL-CV-GPR_S0_v31-1_{}.csv'.format(root, i_label)
with open(name, 'a', newline = '\n') as f:
    writer = csv.writer(f)
    writer.writerow([i_ker, i_cov, i_sec, i_nor] + time_ + nmll_val_machine_.tolist() + nmll_ts_machine_.tolist())
# Save Results
name = r'{}/data/kernel_learning/GPRs/CV-GPR_S0_v31-1_{}{}{}{}-{}.pkl'.format(root, i_ker, i_cov, i_sec, i_nor, i_label)
with open(name, 'wb') as handle:
    pickle.dump([Y_ts_hat_, Sp_ts_hat_, sn_ts_hat_], handle, protocol = pickle.HIGHEST_PROTOCOL)
# Save Models
# name = r'{}/model/kernel_learning/GPRs/CV-GPR_v31-1_{}{}{}{}-{}.pkl'.format(root, i_ker, i_cov, i_sec, i_nor, i_label)
# with open(name, 'wb') as handle:
#     pickle.dump(_model, handle, protocol = pickle.HIGHEST_PROTOCOL)
