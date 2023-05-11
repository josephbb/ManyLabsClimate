import pytensor.tensor as at  
from pymc.distributions.continuous import Continuous
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.shape_utils import rv_size_is_none
from pymc.pytensorf import floatX, intX
from pytensor.tensor.random.basic import (
    ScipyRandomVariable,
)
import scipy as sp
from pymc.distributions import Bernoulli
import numpy as np
import pymc as pm
from pymc.distributions import Beta, DiracDelta, Mixture, LogitNormal
from aesara.tensor.random.op import RandomVariable
from typing import List, Tuple
from pytensor.tensor.var import TensorVariable
# Create your own `RandomVariable`...
from pymc import Distribution 
from scipy.special import expit 
from scipy import stats
from scipy import special

import pytensor.tensor as at  
from pymc.pytensorf import floatX, intX


class ZOIBProportionRV(ScipyRandomVariable):
    name = "ZOIB"
    ndim_supp = 0
    ndims_params = [0, 0, 0, 0]
    dtype = 'floatX'
    # A pretty text and LaTeX representation for the RV
    _print_name = ("ZOIB", "\\operatorname{ZOIB}")

    @classmethod
    def rng_fn_scipy(cls, rng, mu, kappa, theta, tau, size) -> np.ndarray: 
        alpha = (kappa * mu)
        beta = (kappa * (1 - mu))
        thetas = sp.stats.uniform.rvs(0, 1, size=size)
        taus =  sp.stats.uniform.rvs(0, 1, size=size)
        betas = sp.stats.beta.rvs(alpha, beta, size=size)
        betas[np.where(np.bitwise_and(taus > tau, thetas < theta))] = 0
        betas[np.where(np.bitwise_and(taus < tau, thetas < theta))] = 1
        return np.asarray(betas) 
    
zoibrv = ZOIBProportionRV()

class ZOIBProportion(Continuous):
    
    rv_op = zoibrv
    
    @classmethod
    def dist(cls, mu=None, kappa=None, theta=None, tau=None, *args, **kwargs):
        mu = at.as_tensor_variable(floatX(mu))
        kappa = at.as_tensor_variable(floatX(kappa))
        theta = at.as_tensor_variable(floatX(theta))
        tau = at.as_tensor_variable(floatX(tau))
        return super().dist([mu, kappa, theta, tau], *args, **kwargs)

    def moment(rv, size, mu, kappa, theta, tau):
        mean = tau * theta + (1-theta) * mu
        if not rv_size_is_none(size):
            mean = at.full(size, mean)
        return mean

    def logp(value, mu, kappa, theta, tau):
        # Return zero when mu and value are both zero
        alpha = (kappa * mu)
        beta = (kappa * (1 - mu))


        # res = at.switch(at.or_(at.gt(value, 0.9998), at.lt(value, 0.0002)), 
        #                                pm.logp(Bernoulli.dist(p=theta), 1.0) + pm.logp(Bernoulli.dist(p=tau), value), 
        #                                pm.logp(Bernoulli.dist(p=theta), 0.0) + pm.logp(Beta.dist(alpha=alpha, beta=beta), value))

        #This is really hacky but necessary to get the GPU sampling to work. 
        #Basically, the GPU sampler cannot take a value 1 or 0 as an input 
        #if it is valuating the logp of a Bernoulli distribution, even if
        #the switch statement is used to ensure avoiding this.
        #The solution is to flag in the data any values that are 1 or 0 and
        #add or substract a small constant such that they are no longer 1 or 0.
        #In the hours spent figuring this out, I could have written a paper.
        res = pm.logp(Bernoulli.dist(p=theta), 0.0) + pm.logp(Beta.dist(alpha=alpha, beta=beta), value)
        res = at.switch(at.gt(value, 0.99998), pm.logp(Bernoulli.dist(p=theta), 1.0) + pm.logp(Bernoulli.dist(p=tau), 1.0), res)
        res = at.switch(at.lt(value, 0.00002), pm.logp(Bernoulli.dist(p=theta), 1.0) + pm.logp(Bernoulli.dist(p=tau), 0.0), res)

        res = at.switch(at.or_(at.lt(value, 0), at.gt(value, 1)), -np.inf, res)

        return check_parameters(
            res,
            msg="mu > 0, mu < 1, kappa > 0, 0 <= theta <= 1, 0 <= tau <= 1",
        )
    


def _upper_inflated_mixture_truncated(*, name, theta, upper, nonzero_dist, **kwargs):
    """Helper function to create an upper-inflated mixture  """
    theta = at.as_tensor_variable(floatX(theta))
    upper = at.as_tensor_variable(intX(upper))

    weights = at.stack([theta, 1-theta], axis=-1)
    comp_dists = [
        pm.DiracDelta.dist(upper),
       nonzero_dist,
    ]
    if name is not None:
        return pm.Mixture(name, weights, comp_dists, **kwargs)
    else:
        return pm.Mixture.dist(weights, comp_dists, **kwargs)


class UpperInflatedTruncatedGeom:
    def __new__(cls, name, p, theta,  upper, **kwargs):
        return _upper_inflated_mixture_truncated(
            name=name, 
            theta=theta,
            upper=upper,
            nonzero_dist=pm.Truncated.dist(pm.Geometric.dist(p=p), lower=1, upper=upper-1),
            **kwargs
        )

    @classmethod
    def dist(cls, p, theta, upper, **kwargs):
        return _upper_inflated_mixture_truncated(
            name=None, 
            theta=theta,
            upper=upper,
            nonzero_dist=pm.Truncated.dist(pm.Geometric.dist(p=p), lower=1, upper=upper-1),
            **kwargs
        )
