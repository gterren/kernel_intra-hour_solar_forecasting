# Kernel Learning for Intra-hour Solar Forecasting

The codes in this repository required auxiliar functions to construct the dataset and the running the cross-validation of the kernels' hyperparameters and models' parameters. Thes functions necessaries are included in these two files: feature_extraction_utils.py and solar_forecasting_utils_v2.py.

## Multi-Ouput Kernel Regression

This repository includes codes for multi-output kernel lernarning models. The methods implemented to develop multi-output models are:

* Independet regressors. An independet model for each forecasting horizon.

* Chain of regressors (i.e., recursive regressors). Similar to the architecture of recursive networks, the regression chain concatenates the output of previous models with the covariates in the next one.

* Multi-task regressor. A single model for all forecasting horizon.

## Dense Kernel Methods

Kernel learning methods that use all the samples to define the covariance matrix (i.e., Gram matrix).

### Kernel Ridge Regression

Implementation of KRR in pytorch for GPUs and CPUs paralellization support. Respectivaly, the codes are CV-KRR.py, CV-RKRR.py, and CV-MTKRR.py for independent KRRs, chain of KRRs and multi-task KRR.

### Gaussian Process for Regression

The multi-output Gaussian processes were implemented using GPytorch library. As this library is implemented using pytorch, the traning support parallelization. The codes are CV-GPR.py, CV-RGPR.py, and CV-MTGPR.py, for the independent GPRs, the chain of GPRs and the multi-task GPR respectivaly. The models implementation in GPytorch are defined in this file machine_learning_utils.py.

## Sparse Kernel Methods

Kerner learning methods that select the basis functions that convey information to the model. Therefore, the covariance matrix (i.e., Gram matrix) dimensions is smaller to the number of samples.

### Support Vector Machine for Regression

The paralellization of the SVM was performed using MPI, and the algorithm used from Sklearn. The codes are CV-SVM-MPI.pym CV-RSVM-MPI.py, and CV-MTSVM-MPI.py for the independent SVMs, chain of SVMs, and multi-task SVM. Notice that in the case of MTSVM, the kernel have to be precomputed and the inputs extended, so it can be solved with sklearn.

### Relevance Vector Machine for Regression

Implementation of RVM in pytorch for GPUs and CPUs paralellization support. Respectivaly, the codes are CV-RVM.py, CV-RRVM.py, and CV-MTRVM.py for independent RVMs, chain of RVMs and multi-task RVM.
