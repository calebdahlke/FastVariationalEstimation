import numpy as np
from math import isinf
import time
import platform

# This is needed to ignore scikit-learn's forced warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Import proper library for logistic regression fits
if platform.system() == "Linux":
    import glmnet
else: 
    from sklearn import linear_model  

class Inference:

    """
    Base class for Ratio Estimation Inference for a particular simulator model.
    """

    def __init__(self, simobj):

        """
        Parameters
        ----------
        simobj: simulator object
            Object of the implicit simulator model.
        """

        self.simobj = simobj


class MOMENT_MATCH(Inference):

    """
    Class that implements Likelihood-Free Inference by Ratio Estimation (LFIRE)
    for a simulator models.

    Attributes
    ----------
    d: np.ndarray
        The experimental design at which to calculate the LFIRE ratios.
    prior_samples: np.ndarray
        The samples from the prior distribution used in LFIRE.
    weights: np.ndarray
        Weights corresponding to the prior samples.
    simobj: simulator object
        Object of the implicit simulator model.
    psi: function
        Function that returns the summary statistics of data. This can be
        provided seperately but will by default be from 'simobj'.
    kld: list
        List of LFIRE log ratios.
    betas: list
        List of LFIRE classification coefficients for each prior sample.

    Methods
    -------
    eval:
        Returns a set of log-ratios and classification coefficients
        corresponding to the prior samples.

    """

    def __init__(self, d, prior_samples, weights, simobj, psi=None):

        """
        Parameters
        ----------
        d: np.ndarray
            The experimental design at which to calculate the LFIRE ratios.
        prior_samples: np.ndarray
            The samples from the prior distribution used in LFIRE.
        weights: np.ndarray
            Weights corresponding to the prior samples.
        simobj: simulator object
            Object of the implicit simulator model.
        psi: Boolean or Function
            Function that returns the summary statistics of data. If 'None' it
            uses the summary statistics contained in the simulator object.
            (default is None)
        """

        super(MOMENT_MATCH, self).__init__(simobj)
        self.d = d
        self.prior_samples = prior_samples
        self.weights = weights
        self.simobj = simobj
        # summary statistics
        if psi is None:
            self.psi = self.simobj.summary
        else:
            if callable(psi):
                self.psi = psi
            else:
                raise TypeError("Summary statistics needs to be a function.")


    def eval(self, numsamp=10000):
        """
        Evaluates MI using Gaussian moment matching solution.

        numsamp: int
            Number of samples to use. Prior samples at which to compute
            the log ratios are sampled according to the provided 'weights'.
        """

        # Normalise weights
        ws_norm = self.weights / np.sum(self.weights)

        # # Prepare lookup dictionary
        # lookup = dict()
        # for i in range(self.prior_samples.shape[0]):
        #     lookup[i] = np.array([0])

        # resample parameters        
        cat = np.random.choice(range(len(ws_norm)), p=ws_norm, size=numsamp)
        this_prior_samples = self.prior_samples[cat]
        y = [ self.simobj.sample_data(self.d, samp, num=1)[0] for samp in this_prior_samples ]

        # compute covariance 
        all_samples = np.concatenate((this_prior_samples, y), axis=1)
        all_samples = all_samples[:,0:-1]  # Drop (R)ecovered since it is deterministic
        cov_joint = np.cov(all_samples, rowvar=False)
        cov_joint += 1e-5 * np.eye( 4 )  # Add fudge factor to diagonal to prevent singular covariance
        
        # compute log-determinants
        cov_prior = cov_joint[0:2, 0:2]
        s_prior, logdet_prior = np.linalg.slogdet(cov_prior)
        assert s_prior > 0.0
        cov_like = cov_joint[2:, 2:]
        s_like, logdet_like = np.linalg.slogdet(cov_like)
        assert s_like > 0.0
        s_joint, logdet_joint = np.linalg.slogdet(cov_joint)
        assert s_joint > 0.0
        
        # return MI
        MI = 0.5 * ( logdet_prior + logdet_like - logdet_joint )
        return MI
        

class LFIRE(Inference):

    """
    Class that implements Likelihood-Free Inference by Ratio Estimation (LFIRE)
    for a simulator models.

    Attributes
    ----------
    d: np.ndarray
        The experimental design at which to calculate the LFIRE ratios.
    prior_samples: np.ndarray
        The samples from the prior distribution used in LFIRE.
    weights: np.ndarray
        Weights corresponding to the prior samples.
    simobj: simulator object
        Object of the implicit simulator model.
    psi: function
        Function that returns the summary statistics of data. This can be
        provided seperately but will by default be from 'simobj'.
    kld: list
        List of LFIRE log ratios.
    betas: list
        List of LFIRE classification coefficients for each prior sample.

    Methods
    -------
    ratios:
        Returns a set of log-ratios and classification coefficients
        corresponding to the prior samples.

    """

    def __init__(self, d, prior_samples, weights, simobj, psi=None):

        """
        Parameters
        ----------
        d: np.ndarray
            The experimental design at which to calculate the LFIRE ratios.
        prior_samples: np.ndarray
            The samples from the prior distribution used in LFIRE.
        weights: np.ndarray
            Weights corresponding to the prior samples.
        simobj: simulator object
            Object of the implicit simulator model.
        psi: Boolean or Function
            Function that returns the summary statistics of data. If 'None' it
            uses the summary statistics contained in the simulator object.
            (default is None)
        """

        super(LFIRE, self).__init__(simobj)
        self.d = d
        self.prior_samples = prior_samples
        self.weights = weights
        self.simobj = simobj
        # summary statistics
        if psi is None:
            self.psi = self.simobj.summary
        else:
            if callable(psi):
                self.psi = psi
            else:
                raise TypeError("Summary statistics needs to be a function.")

    def _logistic_regression(self, theta, K=10):

        """
        Computes logistic regression coefficients for data sampled from the
        likelihood and from the marginal.

        theta: np.ndarray
            Parameter sample for which to compute the coefficients.
        K: int
            The number of K-fold cross-validations to be used.
        """

        # Select model params according to weights
        ws_norm = self.weights / np.sum(self.weights)
        p_selec = list()
        idx_selec = list()
        for _ in range(self.prior_samples.shape[0]):
            cat = np.random.choice(range(len(ws_norm)), p=ws_norm)
            p_selec.append(self.prior_samples[cat])
            idx_selec.append(cat)

        # Simulate from marginal using selected model params
        y_m = self.simobj.sample_data(self.d, p_selec)
        y_m = self.simobj.summary(y_m)

        # Simulate from likelihood
        y_t = self.simobj.sample_data(self.d, theta, num=len(ws_norm))
        y_t = self.simobj.summary(y_t)

        # Prepare targets
        t_t = np.ones(y_t.shape[0])
        t_m = np.zeros(y_m.shape[0])

        # Concatenate data
        Y = np.concatenate((y_t, y_m), axis=0)
        T = np.concatenate((t_t, t_m))

        # Define glmnet model
        if platform.system() == "Linux":
            model = glmnet.LogitNet(
                n_splits=K, verbose=False, n_jobs=1, scoring="log_loss")        
            model.fit(Y, T)

            # collect coefficients and intercept
            cf_choice = model.coef_path_[..., model.lambda_max_inx_].T.reshape(-1)
            inter = model.intercept_path_[..., model.lambda_max_inx_]
        else:   
            model = linear_model.LogisticRegressionCV(
                cv=K, verbose=False, n_jobs=1, scoring="neg_log_loss")
            
            # collect coefficients and intercept
            cf_choice = model.coef_.T.reshape(-1)
            inter = model.intercept_

        cf = np.array(list(inter) + list(cf_choice)).reshape(-1, 1)        
        return cf

    def ratios(self, numsamp=10000):

        """
        Returns a set of LFIRE log ratios of size 'numsamp', as well as
        coefficients for each model parameter.

        numsamp: int
            Number of desired LFIRE ratios. Prior samples at which to compute
            the log ratios are sampled according to the provided 'weights'.
        """

        # Normalise weights
        ws_norm = self.weights / np.sum(self.weights)

        # Prepare lookup dictionary
        lookup = dict()
        for i in range(self.prior_samples.shape[0]):
            lookup[i] = np.array([0])

        # For each sample from the prior distribution, compute ratios
        self.kld = list()
        tdelta = 0.0
        for _ in range(numsamp):

            # sample data from selected prior sample
            cat = np.random.choice(range(len(ws_norm)), p=ws_norm)
            y = self.simobj.sample_data(self.d, self.prior_samples[cat], num=1)

            # compute summary statistics of data
            if len(y.shape) > 2:
                psi_y = self.simobj.summary(y)
            else:
                psi_y = self.simobj.summary(y.reshape(1, -1))

            # check if we haven't already computed logistic regression cfs
            if (lookup[cat] == 0).all():
                t1 = time.time()
                b = self._logistic_regression(self.prior_samples[cat], K=3)
                t2 = time.time()
                lookup[cat] = b
                tdelta += t2-t1
            else:
                b = lookup[cat]

            # compute the log ratio of the data y with the coefficients
            logr = psi_y.reshape(1, -1) @ b[1:] + b[0]
            logr = logr[0][0]

            # store the log ratio
            if not isinf(logr):
                self.kld.append(logr)

        # Store the coefficients for each parameter sample in a list
        self.betas = list(lookup.values())

        # print('Total Regression Time {0:.2f}s for design {1:.3f} '.format(tdelta, self.d[0]))

        return self.kld, self.betas


class LFIRE_SMC(Inference):

    """
    Class that implements Likelihood-Free Inference by Ratio Estimation (LFIRE)
    for a simulator models. The difference to the 'LFIRE' class is that this
    class provides one log-ratio for each prior parameter, instead of
    potentially several.

    Attributes
    ----------
    d: np.ndarray
        The experimental design at which to calculate the LFIRE ratios.
    prior_samples: np.ndarray
        The samples from the prior distribution used in LFIRE.
    weights: np.ndarray
        Weights corresponding to the prior samples.
    simobj: simulator object
        Object of the implicit simulator model.
    psi: function
        Function that returns the summary statistics of data. This can be
        provided seperately but will by default be from 'simobj'.
    kld: list
        List of LFIRE log ratios.
    betas: list
        List of LFIRE classification coefficients for each prior sample.

    Methods
    -------
    ratios:
        Returns a set of log-ratios and classification coefficients
        corresponding to the prior samples.

    """

    def __init__(self, d, prior_samples, weights, simobj, psi=None):

        """
        Parameters
        ----------
        d: np.ndarray
            The experimental design at which to calculate the LFIRE ratios.
        prior_samples: np.ndarray
            The samples from the prior distribution used in LFIRE.
        weights: np.ndarray
            Weights corresponding to the prior samples.
        simobj: simulator object
            Object of the implicit simulator model.
        psi: Boolean or Function
            Function that returns the summary statistics of data. If 'None' it
            uses the summary statistics contained in the simulator object.
            (default is None)
        """

        super(LFIRE_SMC, self).__init__(simobj)
        self.d = d
        self.prior_samples = prior_samples
        self.weights = weights
        self.simobj = simobj
        # summary statistics
        if psi is None:
            self.psi = self.simobj.summary
        else:
            if callable(psi):
                self.psi = psi
            else:
                raise TypeError("Summary statistics needs to be a function.")

    def _logistic_regression(self, theta, K=10):

        """
        Computes logistic regression coefficients for data sampled from the
        likelihood and from the marginal.

        theta: np.ndarray
            Parameter sample for which to compute the coefficients.
        K: int
            The number of K-fold cross-validations to be used.
        """

        # Create weights for logistic regression
        ws_norm = self.weights / np.sum(self.weights)
        ws_unweighted = np.ones(len(ws_norm)) / len(ws_norm)
        weights_concat = np.concatenate((ws_unweighted, ws_norm))

        # Simulate from marginal using selected model params
        y_m = self.simobj.sample_data(self.d, self.prior_samples)
        y_m = self.simobj.summary(y_m)

        # Simulate from likelihood
        y_t = self.simobj.sample_data(self.d, theta, num=len(ws_norm))
        y_t = self.simobj.summary(y_t)

        # Prepare targets
        t_t = np.ones(y_t.shape[0])
        t_m = np.zeros(y_m.shape[0])

        # Concatenate data
        Y = np.concatenate((y_t, y_m), axis=0)
        T = np.concatenate((t_t, t_m))

        # Define glmnet model
        # model = glmnet.LogitNet(
        #     n_splits=K, verbose=False, n_jobs=1, scoring="log_loss")
        model = sklearn.linear_model.LogisticRegressionCV(
            cv=K, verbose=False, n_jobs=1, scoring="neg_log_loss")
        model.fit(Y, T, sample_weight=weights_concat)

        # collect coefficients and intercept
        # cf_choice = model.coef_path_[..., model.lambda_max_inx_].T.reshape(-1)
        # inter = model.intercept_path_[..., model.lambda_max_inx_]
        cf_choice = model.coef_.T.reshape(-1)
        inter = model.intercept_
        coef = np.array(list(inter) + list(cf_choice)).reshape(-1, 1)

        return cf

    def ratios(self):

        """
        Returns a set of LFIRE log ratios of the size of the prior samples.
        """

        # Normalise weights
        ws_norm = self.weights / np.sum(self.weights)

        # For each sample from the prior distribution, compute ratios
        self.kld = list()
        self.betas = list()
        for p in tqdm(self.prior_samples, disable=not bar):

            # sample data from selected prior sample
            y = self.simobj.sample_data(self.d, p, num=1)

            # compute summary statistics of data
            if len(y.shape) > 2:
                psi_y = self.simobj.summary(y)
            else:
                psi_y = self.simobj.summary(y.reshape(1, -1))

            # compute ratio for all prior samples
            b = self.logistic_regression_lowvar(
                self.d, p, self.simobj, self.prior_samples, self.weights, K=10
            )
            self.betas.append(b)

            # compute the log ratio of the data y with the coefficients
            logr = psi_y.reshape(1, -1) @ b[1:] + b[0]
            logr = logr[0][0]

            # store the log ratio
            if not isinf(logr):
                self.kld.append(logr)
            else:
                self.kld.append(0)

        return self.kld, self.betas

if __name__ == '__main__':
    print('Main called in inference.py!')
