# Kernel Learning for Intra-hour Solar Forecasting

feature_extraction_utils.py

solar_forecasting_utils_v2.py

## Multi-Ouput Kernel Regression

This repository includes codes for multi-output kernel lernarning models. The methods implemented to develop multi-output models are:

* independet Regressors

* Regressors Chain 

* Multi-Task Regressor

## Dense Kernel Methods

### Kernel Ridge Regression

Implementation of KRR in pytorch for GPUs and CPUs paralellization support. Respectivaly, the codes are CV-KRR.py, CV-RKRR.py, and CV-MTKRR.py for independent KRRs, chain of KRRs and Multi-Task KRR.

### Gaussian Process for Regression

CV-GPR.py, CV-RGPR.py, CV-MTGPR.py


machine_learning_utils.py

## Sparse Kernel Methods

### Support Vector Machine for Regression

The paralellization of the SVM was performed using MPI, and the algorithm used from Sklearn. The codes are CV-SVM-MPI.pym CV-RSVM-MPI.py, and CV-MTSVM-MPI.py for the independent SVMs, chain of SVMs, and Multi-Task SVM. Notice that in the case of MTSVM, the kernel have to be precomputed and the inputs extended, so it can be solved with sklearn.

### Relevance Vector Machine for Regression

Implementation of SVM in pytorch for GPUs and CPUs paralellization support. Respectivaly, the codes are CV-RVM.py, CV-RRVM.py, and CV-MTRVM.py for independent RVMs, chain of RVMs and Multi-Task RVM.
