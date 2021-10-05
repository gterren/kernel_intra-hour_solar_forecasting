import pickle, glob, sys, os, warnings, csv, gpytorch, torch
import numpy as np

# # Gaussian Process Regression model fit...
# def _GPR_fit(X_, y_, kernel, degree, verbose = False):
#     def __optimize(_model, _likel, X_, y_, training_iter):
#         # Find optimal model hyperparameters
#         _model.train()
#         # Use the adam optimizer
#         _optimizer = torch.optim.Adam(_model.parameters(), lr = 0.1)  # Includes GaussianLikelihood parameters
#         # "Loss" for GPs - the marginal log likelihood
#         _mll = gpytorch.mlls.ExactMarginalLogLikelihood(_likel, _model)
#         for i in range(training_iter):
#             # Zero gradients from previous iteration
#             _optimizer.zero_grad()
#             # Output from model
#             f_hat_tr_ = _model(X_tr_)
#             # Calc loss and backprop gradients
#             _error = - _mll(f_hat_tr_, y_tr_)
#             _error.backward()
#             if verbose:
#                 print('Iter %d/%d - Loss: %.3f noise: %.3f' % (i + 1, training_iter, _error.item(), _model.likelihood.noise.item()))
#             _optimizer.step()
#         return _model, _likel
#
#     X_tr_ = torch.tensor(X_, dtype = torch.float)
#     y_tr_ = torch.tensor(y_, dtype = torch.float)
#     # initialize likelihood and model
#     _likel = gpytorch.likelihoods.GaussianLikelihood()
#     _model = ExactGPModel(X_tr_, y_tr_, _likel, kernel, degree)
#     return __optimize(_model, _likel, X_tr_, y_tr_, training_iter = 50)


# # Gaussian Process Regression model fit...
# def _GPR_fit(X_, y_, kernel_, degree_, max_training_iter, early_stop, random_init):
#     # Optimize Kernel hyperparameters
#     def __optimize(_model, _likel, X_, y_, max_training_iter, early_stop):
#         # Storage Variables Initialization
#         error_ = []
#         # Find optimal model hyperparameters
#         _model.train()
#         # Use the adam optimizer
#         _optimizer = torch.optim.Adam(_model.parameters(), lr = .1)  # Includes GaussianLikelihood parameters
#         # "Loss" for GPs - the marginal log likelihood
#         _mll = gpytorch.mlls.ExactMarginalLogLikelihood(_likel, _model)
#         # Begins Iterative Optimization
#         for i in range(max_training_iter):
#             # Zero gradients from previous iteration
#             _optimizer.zero_grad()
#             # Output from model
#             f_hat_tr_ = _model(X_tr_)
#             # Calc loss and backprop gradients
#             _error = - _mll(f_hat_tr_, y_tr_)
#             _error.backward()
#             _optimizer.step()
#             # Optimization Early stopping
#             error_.append(np.around(float(_error.detach().numpy()), 3) )
#             if  i > early_stop:
#                 if np.unique(error_[-early_stop:]).shape[0] == 1:
#                     break
#         return _model, _likel, error_[-1]
#
#     X_tr_ = torch.tensor(X_, dtype = torch.float)
#     y_tr_ = torch.tensor(y_, dtype = torch.float)
#     # initialize likelihood and model
#     _likel = gpytorch.likelihoods.GaussianLikelihood()
#     #_likel = gpytorch.likelihoods.GaussianLikelihood(noise_constraint = gpytorch.constraints.GreaterThan(1e-10))
#     _model = _GPR(X_tr_, y_tr_, _likel, kernel, degree, random_init)
#     return __optimize(_model, _likel, X_tr_, y_tr_, max_training_iter, early_stop)

# # Gaussian Process Regression model fit...
# def _GPR_fit(X_, y_, kernel, degree, verbose = False):
#     def __optimize(_model, _likel, X_, y_, training_iter):
#         # Find optimal model hyperparameters
#         _model.train()
#         # Use the adam optimizer
#         _optimizer = torch.optim.Adam(_model.parameters(), lr = 0.1)  # Includes GaussianLikelihood parameters
#         # "Loss" for GPs - the marginal log likelihood
#         _mll = gpytorch.mlls.ExactMarginalLogLikelihood(_likel, _model)
#         for i in range(training_iter):
#             # Zero gradients from previous iteration
#             _optimizer.zero_grad()
#             # Output from model
#             f_hat_tr_ = _model(X_tr_)
#             # Calc loss and backprop gradients
#             _error = - _mll(f_hat_tr_, y_tr_)
#             _error.backward()
#             if verbose:
#                 print('Iter %d/%d - Loss: %.3f noise: %.3f' % (i + 1, training_iter, _error.item(), _model.likelihood.noise.item()))
#             _optimizer.step()
#         return _model, _likel
#
#     X_tr_ = torch.tensor(X_, dtype = torch.float)
#     y_tr_ = torch.tensor(y_, dtype = torch.float)
#     # initialize likelihood and model
#     _likel = gpytorch.likelihoods.GaussianLikelihood()
#     _model = ExactGPModel(X_tr_, y_tr_, _likel, kernel, degree)

def _random(low = -2.5, high = 2.5):
    return torch.tensor(np.exp(float(np.random.uniform(low, high, size = 1)[0])))
#
# # Gaussian Process for Regression
# class ExactGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood, kernel, degree):
#         super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module  = gpytorch.means.ConstantMean()
#         if kernel is 'linear': self.covar_module = gpytorch.kernels.LinearKernel()
#         if kernel is 'RBF':    self.covar_module = gpytorch.kernels.RBFKernel()
#         if kernel is 'poly':   self.covar_module = gpytorch.kernels.PolynomialKernel(power = degree)
#         if kernel is 'matern': self.covar_module = gpytorch.kernels.MaternKernel(nu = degree)
#         if kernel is 'RQ':     self.covar_module = gpytorch.kernels.RQKernel()
#
#     def forward(self, x):
#         mean_x  = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Gaussian Process for Regression
class _GPR(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, degree, num_dim, random_init            = True,
                                                                              multiple_length_scales = False):
        super(_GPR, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # Random Parameters Initialization
        self.random_init            = random_init
        self.multiple_length_scales = multiple_length_scales
        idx_dim_                    = torch.linspace(1, num_dim, num_dim, dtype = int) - 1
        # Define Covariates Kernel
        _K                = self.__define_kernel(kernel, degree, idx_dim_ = idx_dim_[:-1])
        # Define bias Kernel
        _K_b              = self.__define_kernel(kernel   = 'linear', 
                                                 degree   = 0,
                                                 idx_dim_ = idx_dim_[-1])
        self.covar_module = _K + _K_b
    # Define a kernel
    def __define_kernel(self, kernel, degree, idx_dim_ = None):
        if self.multiple_length_scales:
            dim = int(idx_dim_.shape[0])
        else:
            dim = None
        # Random Initialization Covariance Matrix
        if self.random_init: self.likelihood.noise_covar.raw_noise.data.fill_(self.likelihood.noise_covar.raw_noise_constraint.inverse_transform(_random()))
        if self.random_init: 
            self.mean_module.constant.data.fill_(_random())
        # Linear Kernel
        if kernel is 'linear':
            _K = gpytorch.kernels.LinearKernel(active_dims = idx_dim_)
            # Linear Kernel hyperparameters
            if self.random_init: _K.raw_variance.data.fill_(_K.raw_variance_constraint.inverse_transform(_random()))
            return _K
        # Radian Basis Function Kernel
        if kernel is 'RBF':
            _K = gpytorch.kernels.RBFKernel(active_dims = idx_dim_, ard_num_dims = dim)
            # RBF Kernel hyperparameters
            if self.random_init:
                for i in range(_K.raw_lengthscale.shape[1]): _K.raw_lengthscale[0, i].data.fill_(_K.raw_lengthscale_constraint.inverse_transform(_random()))
        # Polynomial Expansion Kernel
        if kernel is 'poly':
            _K = gpytorch.kernels.PolynomialKernel(power = degree, active_dims = idx_dim_)
            # Polynomial Kernel hyperparameters
            if self.random_init: _K.raw_offset.data.fill_(_K.raw_offset_constraint.inverse_transform(_random()))
        # Matern Kernel
        if kernel is 'matern':
            _K = gpytorch.kernels.MaternKernel(nu = degree, active_dims = idx_dim_, ard_num_dims = dim)
            # Matern Kernel hyperparameters
            if self.random_init:
                for i in range(_K.raw_lengthscale.shape[1]): _K.raw_lengthscale[0, i].data.fill_(_K.raw_lengthscale_constraint.inverse_transform(_random()))
        # Rational Quadratic Kernel
        if kernel is 'RQ':
            _K = gpytorch.kernels.RQKernel(active_dims = idx_dim_, ard_num_dims = dim)
            # RQ Kernel hyperparameters
            if self.random_init: _K.raw_alpha.data.fill_(_K.raw_alpha_constraint.inverse_transform(_random()))
            if self.random_init:
                for i in range(_K.raw_lengthscale.shape[1]): _K.raw_lengthscale[0, i].data.fill_(_K.raw_lengthscale_constraint.inverse_transform(_random()))
        return _K
        #_K = gpytorch.kernels.ScaleKernel(_K)
        # Amplitude Coefficient hyperparameters
        #if self.random_init: _K.raw_outputscale.data.fill_(_K.raw_outputscale_constraint.inverse_transform(_random()))

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class _MTGPR(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, degree, num_dim, num_tasks, random_init            = True, 
                                                                                         multiple_length_scales = False):
        super(_MTGPR, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks = num_tasks)
        # Random Parameters Initialization
        self.random_init            = random_init
        self.multiple_length_scales = multiple_length_scales
        idx_dim_                    = torch.linspace(1, num_dim, num_dim, dtype = int) - 1
        # Define bias Kernel
        _K_b              = self.__define_kernel(kernel    = 'linear', 
                                                 degree    = 0, 
                                                 num_tasks = num_tasks,
                                                 idx_dim_  = idx_dim_[-1])
        # Define Covariates Kernel
        _K                = self.__define_kernel(kernel, degree, num_tasks = num_tasks,
                                                                 idx_dim_  = idx_dim_[:-1])
        # Add Kernels
        self.covar_module = gpytorch.kernels.MultitaskKernel(_K + _K_b, num_tasks = num_tasks, 
                                                                        rank      = 1)
    # Define a kernel
    def __define_kernel(self, kernel, degree, num_tasks, idx_dim_ = None):
        if self.multiple_length_scales:
            dim = int(idx_dim_.shape[0])
        else:
            dim = None
        # Random Initialization Covariance Matrix
        if self.random_init:
            # Kernel noise parameter
            self.likelihood.raw_noise.data.fill_(self.likelihood.raw_noise_constraint.inverse_transform(_random()))
            # Task noise parameter
            for i in range(num_tasks):
                self.likelihood.raw_task_noises.data[i].fill_(
                    self.likelihood.raw_task_noises_constraint.inverse_transform(_random()))
        # Linear Kernel
        if kernel is 'linear':
            _K = gpytorch.kernels.LinearKernel(active_dims = idx_dim_)
            # Linear Kernel hyperparameters
            if self.random_init: 
                _K.raw_variance.data.fill_(_K.raw_variance_constraint.inverse_transform(_random()))
            return _K
        # Radian Basis Function Kernel
        if kernel is 'RBF':
            _K = gpytorch.kernels.RBFKernel(active_dims = idx_dim_, ard_num_dims = dim)
            # RBF Kernel hyperparameters
            if self.random_init:
                for i in range(_K.raw_lengthscale.shape[1]): 
                    _K.raw_lengthscale[0, i].data.fill_(_K.raw_lengthscale_constraint.inverse_transform(_random()))
            return _K
        # Polynomial Expansion Kernel
        if kernel is 'poly':
            _K = gpytorch.kernels.PolynomialKernel(power = degree, active_dims = idx_dim_)
            # Polynomial Kernel hyperparameters
            if self.random_init: 
                _K.raw_offset.data.fill_(_K.raw_offset_constraint.inverse_transform(_random()))
            return _K
        # Matern Kernel
        if kernel is 'matern':
            _K = gpytorch.kernels.MaternKernel(nu = degree, active_dims = idx_dim_, ard_num_dims = dim)
            # Matern Kernel hyperparameters
            if self.random_init:
                for i in range(_K.raw_lengthscale.shape[1]): 
                    _K.raw_lengthscale[0, i].data.fill_(_K.raw_lengthscale_constraint.inverse_transform(_random()))
            return _K
        # Rational Quadratic Kernel
        if kernel is 'RQ':
            _K = gpytorch.kernels.RQKernel(active_dims = idx_dim_, ard_num_dims = dim)
            # RQ Kernel hyperparameters
            if self.random_init: 
                _K.raw_alpha.data.fill_(_K.raw_alpha_constraint.inverse_transform(_random()))
            if self.random_init:
                for i in range(_K.raw_lengthscale.shape[1]): 
                    _K.raw_lengthscale[0, i].data.fill_(_K.raw_lengthscale_constraint.inverse_transform(_random()))
            return _K
        #_K = gpytorch.kernels.ScaleKernel(_K)
        # Amplitude Coefficient hyperparameters
        #if self.random_init: _K.raw_outputscale.data.fill_(_K.raw_outputscale_constraint.inverse_transform(_random()))

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

# class MultitaskGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood, kernel, degree, num_tasks):
#         super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module  = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks = num_tasks)
#         if kernel is 'linear': self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.LinearKernel(), num_tasks = num_tasks, rank = 1)
#         if kernel is 'rbf':    self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.RBFKernel(), num_tasks = num_tasks, rank = 1)
#         if kernel is 'poly':   self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.PolynomialKernel(power = degree), num_tasks = num_tasks, rank = 1)
#         if kernel is 'matern': self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.MaternKernel(nu = degree), num_tasks = num_tasks, rank = 1)
#         if kernel is 'RQ':     self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.RQKernel(), num_tasks = num_tasks, rank = 1)
#
#     def forward(self, x):
#         mean_x  = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def _display_gp_parameters(_model):
    for param_name, param in _model.named_parameters():
        print(param_name)
        print(param)
        #print(f'Parameter name: {param_name:42} value = {param.item()}')

# Gaussian Process MultiKernel for Regression
class _GPR_MK(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_, degree_, idx_dim_,
                 random_init = True, num_dim = None, multiple_length_scales = True):
        super(_GPR_MK, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # Random Parameters Initialization
        self.random_init            = random_init
        self.multiple_length_scales = multiple_length_scales
        # Define Recursive Dimensions for Multiple Source Kernel
        dim    = list(train_x.size())[1]
        extra_ = torch.linspace(num_dim, dim - 2, steps =  dim - num_dim - 1, dtype = int)
        # Define bias Kernel
        self.covar_module = self.__define_kernel('linear', 0, idx_dim_ = torch.tensor([dim - 1]))
        # Define Recursive Kernel
        self.covar_module = self.__define_kernel('RQ', 0, idx_dim_ = extra_)
        # Define Features Kernels
        for kernel, degree, idx_dim in zip(kernel_, degree_, idx_dim_):
            self.covar_module += self.__define_kernel(kernel, degree, idx_dim)

    # Define a kernel
    def __define_kernel(self, kernel, degree, idx_dim_ = None):
        if self.multiple_length_scales:
            dim = int(idx_dim_.shape[0])
        else:
            dim = None
        print(kernel, degree, dim)
        # Random Initialization Covariance Matrix
        if self.random_init:
            self.likelihood.noise_covar.raw_noise.data.fill_(self.likelihood.noise_covar.raw_noise_constraint.inverse_transform(_random()))
        if self.random_init:
            self.mean_module.constant.data.fill_(_random())
        # Linear Kernel
        if kernel is 'linear':
            _K = gpytorch.kernels.LinearKernel(active_dims = idx_dim_)
            # Linear Kernel hyperparameters
            if self.random_init:
                _K.raw_variance.data.fill_(_K.raw_variance_constraint.inverse_transform(_random()))
            return _K
        # Radian Basis Function Kernel
        if kernel is 'RBF':
            _K = gpytorch.kernels.RBFKernel(active_dims = idx_dim_, ard_num_dims = dim)
            # RBF Kernel hyperparameters
            if self.random_init:
                for i in range(_K.raw_lengthscale.shape[1]): _K.raw_lengthscale[0, i].data.fill_(_K.raw_lengthscale_constraint.inverse_transform(_random()))
            _K = gpytorch.kernels.ScaleKernel(_K)
            if self.random_init:
                _K.raw_outputscale.data.fill_(_K.raw_outputscale_constraint.inverse_transform(_random()))
            return _K
        # Polynomial Expansion Kernel
        if kernel is 'poly':
            _K = gpytorch.kernels.PolynomialKernel(power = degree, active_dims = idx_dim_)
            # Polynomial Kernel hyperparameters
            if self.random_init:
                _K.raw_offset.data.fill_(_K.raw_offset_constraint.inverse_transform(_random()))
            _K = gpytorch.kernels.ScaleKernel(_K)
            if self.random_init:
                _K.raw_outputscale.data.fill_(_K.raw_outputscale_constraint.inverse_transform(_random()))
            return _K
        # Matern Kernel
        if kernel is 'matern':
            _K = gpytorch.kernels.MaternKernel(nu = degree, active_dims = idx_dim_, ard_num_dims = dim)
            # Matern Kernel hyperparameters
            if self.random_init:
                for i in range(_K.raw_lengthscale.shape[1]): _K.raw_lengthscale[0, i].data.fill_(_K.raw_lengthscale_constraint.inverse_transform(_random()))
            _K = gpytorch.kernels.ScaleKernel(_K)
            if self.random_init:
                _K.raw_outputscale.data.fill_(_K.raw_outputscale_constraint.inverse_transform(_random()))
            return _K
        # Rational Quadratic Kernel
        if kernel is 'RQ':
            _K = gpytorch.kernels.RQKernel(active_dims = idx_dim_, ard_num_dims = dim)
            # RQ Kernel hyperparameters
            if self.random_init:
                _K.raw_alpha.data.fill_(_K.raw_alpha_constraint.inverse_transform(_random()))
            if self.random_init:
                for i in range(_K.raw_lengthscale.shape[1]): _K.raw_lengthscale[0, i].data.fill_(_K.raw_lengthscale_constraint.inverse_transform(_random()))
            _K = gpytorch.kernels.ScaleKernel(_K)
            if self.random_init:
                _K.raw_outputscale.data.fill_(_K.raw_outputscale_constraint.inverse_transform(_random()))
            return _K
        #_K = gpytorch.kernels.ScaleKernel(_K)
        # Amplitude Coefficient hyperparameters
        #if self.random_init: _K.raw_outputscale.data.fill_(_K.raw_outputscale_constraint.inverse_transform(_random()))

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

__all__ = ['_random', '_GPR', '_MTGPR', '_GPR_MK', '_display_gp_parameters']
