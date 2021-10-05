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
            gamma = torch.tensor(gamma, dtype = torch.float)
            # Compute Linear Kernel
            return gamma*(X @ Y.t())
        if kernel is 'poly':
            # Numpy to Torch
            gamma  = torch.tensor(gamma, dtype = torch.float)
            beta   = torch.tensor(beta, dtype = torch.float)
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
            beta   = torch.tensor(beta, dtype = torch.float)
            # Compute Rational Quadratic Kernel
            return (1. + (gamma*euclidean_distances(X, Y)**2)/(2./beta) )**(-beta)
        if kernel is 'matern':
            # Numpy to Torch
            gamma  = torch.tensor(gamma, dtype = torch.float)
            degree = torch.tensor(degree, dtype = torch.float)
            D_ = torch.sqrt(2.*degree)*(gamma*euclidean_distances(X, Y)**2)
            D_[D_ == 0.0] += torch.tensor(np.finfo(float).eps, dtype = torch.float)
            # This function is not implemented in torch
            A_ = ss.kn(degree, D_)
            # Compute Matern Kernel
            return ( (2**(1 - degree) )/ torch.tensor(ss.gamma(degree), dtype = torch.float)) * ( D_**degree ) * A_

    return __K(X_, Y_, kernel, degree, gamma, beta) + 1.


def _RVM_fit(X_, y_, kernel, degree = None, gamma = None, beta = None, scale = None,
             n_init = 1, n_max_iter = 10, tol = 100, pacience = 50, min_n_rv = 15):
    # Update Relevance and Noise parameter
    def __parameters_update(A_, S_, m_, y_, error, N):
        A_prime_ = A_.clone()
        r_ = torch.zeros((A_.shape[0], 1))
        for i in range(A_.shape[0]):
            r_[i]          = 1. - A_prime_[i, i]*S_[i, i]
            A_prime_[i, i] = r_[i]/(m_[i]**2)
        b = (N - torch.sum(r_))/error
        return A_prime_, b
    # Prune Not Relevance Vectors
    def __prune(A_new_, A_old_, K_, S_, m_, idx_relevance_, tau = 1e10):
        alpha_new_ = torch.diagonal(A_new_)
        alpha_old_ = torch.diagonal(A_old_)
        # Find Relevance Vectors
        idx_alpha_ = torch.absolute( alpha_new_ ) < tau
        # Remove no relance vectors
        m_ = m_[idx_alpha_]
        S_ = S_[np.ix_(idx_alpha_, idx_alpha_)]
        K_ = K_[:, idx_alpha_]
        A_new_ = torch.eye(alpha_new_[idx_alpha_].shape[0]) * alpha_new_[idx_alpha_]
        A_old_ = torch.eye(alpha_old_[idx_alpha_].shape[0]) * alpha_old_[idx_alpha_]
        idx_relevance_ = idx_relevance_[idx_alpha_]
        return A_new_, A_old_, K_, S_, m_, idx_relevance_
    # Compute Posterior mean and covariance
    def __posterior(K_, A_, y_, b):
        S_ = _robust_pinverse(A_ + (b * K_.t() @ K_))
        m_ = b*(S_ @ K_.t() @ y_)
        return S_, m_
    # Compute Total Squared Residuals
    def __error(K_, y_, m_):
        return torch.sum( (y_ - K_ @ m_)**2 )
    # Optimize Model
    def __optimize(K_, X_old_, y_, n_max_iter):
        # Constants Initialization
        N     = K_.shape[0]
        e_opt = np.inf
        # Initialize Relevance Vectors
        idx_relevance_old_ = torch.arange(N, dtype = int)
        # Variables Initialization
        #A_old_ = torch.diag(torch.randn(N))*1e-1
        A_old_ = torch.diag(torch.ones(N))*1e-1
        b      = torch.tensor(1e-1, dtype = torch.float)
        # Loop Over number of iterations
        for i in range(n_max_iter):
            #print(K_.shape, A_old_.shape, y_.shape, b.shape)
            # Compute Covanriance and Mean
            S_, m_ = __posterior(K_, A_old_, y_, b)
            # Compute total sum of squared residuals
            error = __error(K_, y_, m_)
            # Compute a new Update Parameter
            A_new_, b = __parameters_update(A_old_, S_, m_, y_, error, N)
            # Save Best Results
            if ((error < e_opt) and (error < e_opt)) or (i == 0):
                e_opt  = error
                i_opt  = i
                S_opt_ = S_.clone()
                m_opt_ = m_.clone()
                b_opt  = b.clone()
                idx_relevance_opt_ = idx_relevance_old_.clone()
            # Prune no relevance vectors
            A_new_, A_old_, K_, S_, m_, idx_relevance_new_ = __prune(A_new_, A_old_, K_, S_, m_, idx_relevance_old_)
            if (A_new_.shape[0] < min_n_rv):
                break
            delta = torch.amax( torch.absolute( torch.diagonal(A_old_) - torch.diagonal(A_new_)) )
            # Stop when patiance or tolerance is over
            if (delta < tol) or (i - i_opt == pacience):
                break
            else:
                A_old_ = A_new_.clone()
                idx_relevance_old_ = idx_relevance_new_.clone()

        return S_opt_, m_opt_, idx_relevance_opt_, b_opt, e_opt
    # Numpy to Torch
    X_ = torch.tensor(X_, dtype = torch.float)
    y_ = torch.tensor(y_, dtype = torch.float)
    # Evaluate Kernel Funcion
    K_ = _kernel(X_, X_, kernel = kernel,
                         degree = degree,
                         beta   = beta,
                         gamma  = gamma)
    # List of Results Definition
    M_, S_, R_, b_, e_ = [], [], [], [], []
    # Loop Over Initialization
    for n in range(n_init):
        # Fit RVM
        s_, m_, idx_relevance_, b, e = __optimize(K_, X_, y_, n_max_iter)
        # Save RVM Optimized Parameters
        S_.append(s_)
        M_.append(m_)
        R_.append(idx_relevance_)
        b_.append(b)
        e_.append(e)
    # Find Best Initialization
    a_star_        = M_[torch.argmin(torch.tensor(e_, dtype = torch.float))]
    idx_relevance_ = R_[torch.argmin(torch.tensor(e_, dtype = torch.float))]
    S_star_        = S_[torch.argmin(torch.tensor(e_, dtype = torch.float))]
    b_star         = b_[torch.argmin(torch.tensor(e_, dtype = torch.float))]
    return [a_star_, S_star_, b_star, idx_relevance_, X_, [kernel, degree, gamma, beta]]

# Calculing prediction for new sample
def _RVM_predict(Y_, model_):
    # Numpy to Torch
    Y_ = torch.tensor(Y_, dtype = torch.float)
    # Model parameters and hyperparameters
    a_star_, S_star_, b_star, idx_relevance_, X_, theta_ = model_
    kernel, degree, gamma, beta                          = theta_
    # Evaluate kernel function for Testing
    K_star_     = _kernel(Y_, X_, kernel, degree, gamma = gamma, beta = beta)[:, idx_relevance_]
    y_star_hat_ = K_star_ @ a_star_
    s_star_hat_ = torch.diagonal((1./b_star) + K_star_ @ S_star_ @ K_star_.t())
    # Torch to Numpy
    return np.squeeze(y_star_hat_.detach().numpy()), np.squeeze(s_star_hat_.detach().numpy())

# Independet Relevance Vector Machine for regression
def _get_RVM_prediction(data_, theta_, kernel, degree):
    N_tasks     = len(data_)
    N           = data_[0][1][0].shape[0]
    t_tr        = 0.
    t_ts        = 0.
    Y_ts_hat_   = np.zeros((N, N_tasks))
    S2p_ts_hat_ = np.zeros((N, N_tasks))
    _model      = []
    # Loop over independet outputs
    for i_task in range(N_tasks):
        x_tr_, y_tr_ = data_[i_task][0]
        x_ts_, y_ts_ = data_[i_task][1]
        # Cross-validation or testing
        t_init = time.time()
        if theta_.shape[0] == N_tasks:
            # Define Model
            _RVM = _RVM_fit(x_tr_, y_tr_, kernel = kernel,
                                          degree = degree,
                                          gamma  = theta_[i_task, 0],
                                          beta   = theta_[i_task, 1],
                                          scale  = theta_[i_task, 2],
                                          n_init = 1, n_max_iter = 9000)
        else:
            # Define Model
            _RVM = _RVM_fit(x_tr_, y_tr_, kernel = kernel,
                                          degree = degree,
                                          gamma  = theta_[0],
                                          beta   = theta_[1],
                                          scale  = theta_[2],
                                          n_init = 1, n_max_iter = 9000)
        t_tr += time.time() - t_init
        # Testing Model
        t_init                                     = time.time()
        Y_ts_hat_[:, i_task], S2p_ts_hat_[:, i_task] = _RVM_predict(x_ts_, _RVM)
        t_ts                                      += time.time() - t_init
        _model.append(_RVM)
    return Y_ts_hat_, S2p_ts_hat_, [t_tr, t_ts], _RVM


# Implementation of K-fold cross-validation
def _KFold_CV(data_, theta_, kernel, degree, n_kfolds):
    N_tasks = len(data_)
    e_      = np.zeros((n_kfolds, N_tasks))
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
            Y_val_tr_prime_ = _scaler_y.transform(Y_val_tr_)
            #Y_val_tr_prime_ = Y_val_tr_
            X_val_ts_prime_ = _scaler_x.transform(X_val_ts_)
            print(j, i_task, X_val_tr_prime_.shape, Y_val_tr_prime_.shape, X_val_ts_prime_.shape, Y_val_ts_.shape)
            # Save Dataset
            data_val_.append([[X_val_tr_prime_, Y_val_tr_prime_], [X_val_ts_prime_, Y_val_ts_]])
            scaler_val_.append([_scaler_x, _scaler_y])
        #try:
        # Training and Testing RVM
        Y_val_ts_hat_prime_ = _get_RVM_prediction(data_val_, theta_, kernel, degree)[0]
        # Undo Normalization of the prediction
        Y_val_ts_hat_ = np.zeros(Y_val_ts_hat_prime_.shape)
        Y_val_ts_     = np.zeros(Y_val_ts_hat_prime_.shape)
        for i_task in range(N_tasks):
            Y_val_ts_hat_[:, i_task] = scaler_val_[i_task][1].inverse_transform(Y_val_ts_hat_prime_[:, i_task][:, np.newaxis])[:, 0]
            #Y_val_ts_hat_[:, i_task] = Y_val_ts_hat_prime_[:, i_task]
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

# RVM K-Fold Cross-Validation of the model Parameters
def _get_RVM_cross_validation(data_, kernel, degree, n_grid, n_kfolds):
    # Define RVM parameters to validate
    gamma_ = np.array((0., ))
    beta_  = np.array((0., ))
    scale_ = np.array((0., ))
    if kernel == 'linear':
        gamma_ = np.logspace(-4, 0, n_grid)
        theta_ = np.meshgrid(gamma_, beta_, scale_)
    if kernel == 'RBF':
        gamma_ = np.logspace(-4, 2, n_grid)
        theta_ = np.meshgrid(gamma_, beta_, scale_)
    if kernel == 'matern':
        gamma_ = np.logspace(-4, 2, n_grid)
        theta_ = np.meshgrid(gamma_, beta_, scale_)
    if kernel == 'poly':
        gamma_ = np.logspace(-4, 2, n_grid)
        beta_  = np.logspace(-2, 2, n_grid)
        theta_ = np.meshgrid(gamma_, beta_, scale_)
    if kernel == 'RQ':
        gamma_ = np.logspace(-4, 2, n_grid)
        beta_  = np.logspace(-1, 2, n_grid)
        theta_ = np.meshgrid(gamma_, beta_, scale_)
    # Constants Initialization
    theta_  = np.array(theta_).T.reshape(-1, len(theta_))
    N, D    = data_[0][0].shape
    N_tasks = len(data_)
    M, P    = theta_.shape
    e_      = np.zeros((M, N_tasks))
    print(N, D, N_tasks, M, P)
    # Loop over combinations of parameters
    for i in np.arange(M, dtype = int):
        # Kfold Cross-validation Implementation
        e_[i, :] = _KFold_CV(data_, theta_[i, :], kernel, degree, n_kfolds)
    # Find Best Model
    i_ = np.argmin(e_, axis = 0)
    return e_[i_, [0, 1, 2, 3, 4, 5]], theta_[i_, :]

# RVM Model validation without kernels
def _meta_RVM_cross_validation(data_, kernel, degree, n_grid, n_kfolds):
    # RVM Parameters Cross-validation
    e_val_, theta_ = _get_RVM_cross_validation(data_, kernel, degree, n_grid, n_kfolds)
    return e_val_, theta_

# Model Traning and Testing
def _RVM_traing_and_testing(data_, theta_, kernel, degree):
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
        #Y_tr_prime_ = Y_tr_
        X_ts_prime_ = _scaler_x.transform(X_ts_)
        print(i_task, X_tr_prime_.shape, Y_tr_prime_.shape, X_ts_prime_.shape, Y_ts_.shape)
        # Save Dataset
        dataset_.append([[X_tr_prime_, Y_tr_prime_], [X_ts_prime_, Y_ts_]])
        scaler_.append([_scaler_x, _scaler_y])
    #try:
    # Training and testing RVM
    Y_ts_hat_prime_, S2p_ts_hat_prime_, time_, _RVM = _get_RVM_prediction(dataset_, theta_, kernel, degree)
    # Undo Normalization of the prediction
    Y_ts_hat_  = np.zeros(Y_ts_hat_prime_.shape)
    Sp_ts_hat_ = np.zeros(Y_ts_hat_prime_.shape)
    Y_ts_      = np.zeros(Y_ts_hat_prime_.shape)
    # Loop over Task undoing the stardarization
    for i_task in range(N_tasks):
        Y_ts_hat_[:, i_task]  = scaler_[i_task][1].inverse_transform(Y_ts_hat_prime_[:, i_task][:, np.newaxis])[:, 0]
        Sp_ts_hat_[:, i_task] = np.sqrt(S2p_ts_hat_prime_[:, i_task])*scaler_[i_task][1].scale_
        #Y_ts_hat_[:, i_task]  = Y_ts_hat_prime_[:, i_task]
        #Sp_ts_hat_[:, i_task] = np.sqrt(S2p_ts_hat_prime_[:, i_task])
        Y_ts_[:, i_task]      = dataset_[i_task][1][1][:, 0]
    e_ts_ = mean_absolute_percentage_error(Y_ts_, Y_ts_hat_)
    #Compute Validation error metrics
    # except:
    #     print('ERROR!')
    #     print(theta_)
    #     e_[j] = 1e10
    return e_ts_, Y_ts_hat_, Sp_ts_hat_, time_, [_RVM, scaler_]

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
e_val_machine_, theta_ = _meta_RVM_cross_validation(data_val_, kernel, degree, n_grid   = 3,
                                                                               n_kfolds = 3)
print(e_val_machine_)
print(theta_)

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
e_ts_machine_, Y_ts_hat_, Sp_ts_hat_, time_, _model = _RVM_traing_and_testing(data_, theta_, kernel, degree)
print(e_ts_machine_)
print(time_)

# Define directory Roor
root = '/users/terren/solar_forecasting'
# Save Errors
name = r'{}/logs/kernel_learning/RVMs/CV-RVM_S1_v31-1_{}.csv'.format(root, i_label)
with open(name, 'a', newline = '\n') as f:
    writer = csv.writer(f)
    writer.writerow([i_ker, i_cov, i_sec, i_nor] + time_ + e_val_machine_.tolist() + e_ts_machine_.tolist())
# # Save Models
# name = r'{}/model/kernel_learning/RVMs/CV-RVM_S0_v31-1_{}{}{}{}-{}.pkl'.format(root, i_ker, i_cov, i_sec, i_nor, i_label)
# with open(name, 'wb') as handle:
#     pickle.dump(_model, handle, protocol = pickle.HIGHEST_PROTOCOL)
# Save Results
name = r'{}/data/kernel_learning/RVMs/CV-RVM_S1_v31-1_{}{}{}{}-{}.pkl'.format(root, i_ker, i_cov, i_sec, i_nor, i_label)
with open(name, 'wb') as handle:
    pickle.dump([Y_ts_hat_, Sp_ts_hat_], handle, protocol = pickle.HIGHEST_PROTOCOL)
