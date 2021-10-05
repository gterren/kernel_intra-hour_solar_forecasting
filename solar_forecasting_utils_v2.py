import pickle, glob, sys, os, csv
import numpy as np

from datetime import datetime
import time

# Training and Testing set
from sklearn.decomposition import PCA, KernelPCA
from scipy.stats import multivariate_normal

# Load Dataset Day-Sample
def _load_dataset(name):
    # Get Files in Directory
    file_names_ = sorted(glob.glob(name))
    dataset_    = []
    for file_ in file_names_:
        with open(file_, 'rb') as file:
            data_ = pickle.load(file)
        if len(data_) > 100:
            dataset_.append(data_)
            print(file_, len(dataset_[-1]))
    return dataset_

def _load_KPCA(name):
    with open(name, 'rb') as file: return pickle.load(file)

# Select Persistent Prediction
def _get_persistent_prediction(x_, idx_pred_horizon_):
    return np.ones(len(idx_pred_horizon_))*x_[0]

# Mean absolute percentage error regression loss
def root_mean_squared_error(y_, y_hat_):
    mape_ = np.sqrt((y_ - y_hat_)**2)
    mape  = np.average(mape_, axis = 0)
    return mape

# MAPE, MAE and RMSE Prediction Errror
def _get_prediction_error(y_, y_hat_):
    MAE  = np.absolute(y_ - y_hat_)
    MAPE = 100.*MAE/y_
    RMSE = np.sqrt((y_ - y_hat_)**2)
    return MAPE

# Mean absolute percentage error regression loss
def mean_absolute_percentage_error(y_, y_hat_):
    mape_ = np.abs(y_hat_ - y_) / np.abs(y_)
    #mape_ = np.sqrt((y_ - y_hat_)**2)
    return np.average(mape_, axis = 0)

# Probabilistic Samples Selection
def _select_training_samples(X_, Y_, Z_, post_, n_samples):
    g_ = post_ - post_.min() + 1e-50
    g_/=g_.sum()
    # Maximum Likelihood
    idx_ = np.argsort(post_)[::-1][:n_samples]
    return X_[idx_, :], Y_[idx_, :], Z_[idx_, :]

# Probabilistic Samples Selection
def _last_training_samples(X_, Y_, Z_, n_samples):
    return X_[-n_samples:, :], Y_[-n_samples:, :], Z_[-n_samples:, :]

# Probabilistic Samples Selection
def _subset_training_samples(X_, Y_, Z_, post_, n_samples):
    g_ = post_ - post_.min() + 1e-50
    g_/=g_.sum()
    # Probabilistic Weighted Samples
    idx_ = np.random.choice(np.arange(g_.shape[0]), n_samples, replace = False, p = g_)
    return X_[idx_, :], Y_[idx_, :], Z_[idx_, :]

# Add Sample to Database
def _add_sample_to_set(X_, Y_, Z_, x_ts_, y_ts_, z_ts_):
    X_ = np.concatenate((X_, x_ts_[np.newaxis, :]), axis = 0)
    Y_ = np.concatenate((Y_, y_ts_[np.newaxis, :]), axis = 0)
    Z_ = np.concatenate((Z_, z_ts_[np.newaxis, :]), axis = 0)
    return X_, Y_, Z_

# Compute and Regularize covariance matrix
def _get_covariance(X_, gamma):
    return np.cov(X_.T) + np.eye(X_.shape[1])*gamma

# Generate Matrix with all database samples
def _generate_database(dataset_, cov_idx_):
    X_, Y_, Z_ = [], [], []
    for i in range(len(dataset_)):
        for j in range(1, len(dataset_[i])):
            x_, y_, z_, flag = _get_sample_features(dataset_[i], cov_idx_, index = j)
            if flag:
                X_.append(x_)
                Y_.append(y_)
                Z_.append(z_)
    return np.stack(X_), np.stack(Y_), np.stack(Z_)

# Get Sample with Selected Features
def _get_sample_features(sample_, cov_idx_, index):
    flag = True
    # Unpack Selected Features
    ref_, K_, pred_, angl_, auto_, feat_ = sample_[index]
    M_f_, S_f_, G_f_, K_f_, M_h_, S_h_, G_h_, K_h_, M_t_, S_t_, G_t_, K_t_ = feat_
    # Features Index for experiment
    idx_pred, idx_pred_horizon_, idx_auto_, idx_angl_, idx_temp, idx_heig, idx_stats_, idx_cov_horizon_, idx_cov_ = cov_idx_
    # Select Autoregressive Covariates
    x_ = auto_[idx_auto_, idx_pred]
    w_ = angl_[idx_angl_]
    # Select Temperature
    t_ = np.concatenate((M_t_[idx_temp, idx_cov_horizon_][:, np.newaxis],
                         S_t_[idx_temp, idx_cov_horizon_][:, np.newaxis],
                         G_t_[idx_temp, idx_cov_horizon_][:, np.newaxis],
                         K_t_[idx_temp, idx_cov_horizon_][:, np.newaxis]),
                         axis = 1)[:, idx_stats_].flatten()
    # Select Height
    h_ = np.concatenate((M_h_[idx_heig, idx_cov_horizon_][:, np.newaxis],
                         S_h_[idx_heig, idx_cov_horizon_][:, np.newaxis],
                         G_h_[idx_heig, idx_cov_horizon_][:, np.newaxis],
                         K_h_[idx_heig, idx_cov_horizon_][:, np.newaxis]),
                         axis = 1)[:, idx_stats_].flatten()/1000.
    # Select Features
    m_ = np.concatenate((M_f_[0, idx_cov_horizon_][:, np.newaxis],
                         S_f_[0, idx_cov_horizon_][:, np.newaxis],
                         G_f_[0, idx_cov_horizon_][:, np.newaxis],
                         K_f_[0, idx_cov_horizon_][:, np.newaxis]),
                         axis = 1)[:, idx_stats_].flatten()
    d_ = np.concatenate((M_f_[1, idx_cov_horizon_][:, np.newaxis],
                         S_f_[1, idx_cov_horizon_][:, np.newaxis],
                         G_f_[1, idx_cov_horizon_][:, np.newaxis],
                         K_f_[1, idx_cov_horizon_][:, np.newaxis]),
                         axis = 1)[:, idx_stats_].flatten()
    v_ = np.concatenate((M_f_[2, idx_cov_horizon_][:, np.newaxis],
                         S_f_[2, idx_cov_horizon_][:, np.newaxis],
                         G_f_[2, idx_cov_horizon_][:, np.newaxis],
                         K_f_[2, idx_cov_horizon_][:, np.newaxis]),
                         axis = 1)[:, idx_stats_].flatten()
    z_ = np.stack([t_, h_, m_, d_, v_])[idx_cov_, :].flatten()
    # Generate Feature Vector
    x_ = np.concatenate((x_, w_, z_), axis = 0)
    if np.isnan(x_).any() or np.isinf(x_).any():
        print(x_)
        flag = False
    # Select Predictor
    y_ = pred_[idx_pred_horizon_, idx_pred]
    # Get Tag
    z_ = np.array(ref_[1])[np.newaxis]
    return x_, y_, z_, flag

# Training and testing without shuffling the samples
def _split_dataset(X_, Y_, Z_, percentage):
    # No-Shaffle the data
    N    = X_.shape[0]
    N_tr = int(N * percentage)
    N_ts = int(N - N_tr)
    X_tr_ = X_[:N_tr, :]
    Y_tr_ = Y_[:N_tr, :]
    Z_tr_ = Z_[:N_tr, :]
    X_ts_ = X_[-N_ts:, :]
    Y_ts_ = Y_[-N_ts:, :]
    Z_ts_ = Z_[-N_ts:, :]
    return X_tr_, Y_tr_, Z_tr_, X_ts_, Y_ts_, Z_ts_

# # Compute Probabilities of Selecting a Sample from database
# def _get_prob(X_, mu_, sigma_):
#     W_ = X_ - np.tile(mu_, (X_.shape[0], 1))
#     p_ = mu_.shape[0]*np.log(2*np.pi) + np.log(np.linalg.det(sigma_)) + np.diagonal(W_ @ np.linalg.inv(sigma_) @ W_.T)
#     return - p_/2.

# def _get_prob(X_, mu_, sigma_):
#     return multivariate_normal(mu_, sigma_).logpdf(X_)

def _get_prob(X_, mu_, sigma_, verbose = False):

    for gamma in np.logspace(-12, 12, 100):
        try:
            return multivariate_normal(mu_, sigma_ + np.eye(sigma_.shape[0])*gamma).logpdf(X_)
        except:
            if verbose: print('increasing regularization', gamma)
            continue

# Beyesian Sample Selection
def _get_sampling_covariance(X_):
    # Constants definiton
    N = X_.shape[0]
    # Prior Selection Selection
    A_ = np.zeros((X_.shape[1], X_.shape[1]))
    for i in range(N):
        x_  = np.tile(X_[i, :], (X_.shape[0] - 1, 1))
        Z_  = np.delete(X_, i, axis = 0)
        A_ += (Z_ - x_).T @ (Z_ - x_)
    return A_/(N*N)

# Update Sampling Covariance Matrix with last Sample
def _update_sampling_covariance(x_ts_, X_tr_, A_):
    N  = X_tr_.shape[0]
    x_ = np.tile(x_ts_, (X_tr_.shape[0], 1))
    A_prime_ = (X_tr_ - x_).T @ (X_tr_ - x_)
    return ( ( (N - 1.)/(N + 1.) )*A_ ) + ( ( (1.)/(N + 1.) )*A_prime_ )

# Make Sure that the Features were extracted correcty
def _is_good_sample(x_):
    if np.isnan(x_).any() or np.isinf(x_).any():
        return False
    else:
        return True

# Make Sure there is not Errors in the regression
def _robust_regression(y_ts_hat_persistence_, y_ts_hat_machine_):

    if y_ts_hat_machine_ is None:
        return y_ts_hat_persistence_
    else:
        # Check If thre is Nan
        idx_nan_ = np.isnan(y_ts_hat_machine_)
        y_ts_hat_machine_[idx_nan_] = y_ts_hat_persistence_[idx_nan_]
        # Check if out of limits
        idx_0_ = y_ts_hat_machine_ < 0.
        idx_1_ = y_ts_hat_machine_ > 2.
        y_ts_hat_machine_[idx_0_] = y_ts_hat_persistence_[idx_0_]
        y_ts_hat_machine_[idx_1_] = y_ts_hat_persistence_[idx_1_]
        return y_ts_hat_machine_

__all__ = ['_generate_database', '_get_sample_features', '_get_covariance', '_add_sample_to_set',
           'mean_absolute_percentage_error', '_get_prediction_error', '_get_persistent_prediction',
           '_load_KPCA', '_load_dataset', '_split_dataset', '_get_prob', '_is_good_sample',
           '_select_training_samples', '_get_sampling_covariance', '_update_sampling_covariance',
           '_robust_regression', '_subset_training_samples', '_last_training_samples', 'root_mean_squared_error']
