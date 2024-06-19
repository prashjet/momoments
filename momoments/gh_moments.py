import numpy as np
from scipy import stats 
from scipy.special import hermite


def fit_gh_coefficients(true_losvd):
    gh = dyn.kinematics.GaussHermite()
    def obj_func(pars):
        V, sigma, h3, h4 = pars
        h0, h1, h2 = 1., 0., 0.
        h = np.array([[[h0, h1, h2, h3, h4]]])
        gh_losvd = gh.evaluate_losvd(v, np.array([V]), np.array([sigma]), h)
        gh_losvd = np.squeeze(gh_losvd)
        mse = np.mean((gh_losvd - true_losvd)**2)
        return mse
    no_bounds = (None,None)
    r = minimize(obj_func, (0., 1., 0.,0.), bounds=(no_bounds,(0,None),no_bounds,no_bounds))
    return r


def get_discrete_gh(w):
    nrm = stats.norm()
    gh = dyn.kinematics.GaussHermite()
    hpoly_coefs = gh.get_hermite_polynomial_coeffients(4)
    nrm_w_rootfourpi = np.sqrt(4*np.pi)*nrm.pdf(w)
    h3_poly = lambda w: np.polyval(hpoly_coefs[3][::-1], w)
    h4_poly = lambda w: np.polyval(hpoly_coefs[4][::-1], w)
    h3 = np.mean(nrm_w_rootfourpi * h3_poly(w))
    h4 = np.mean(nrm_w_rootfourpi * h4_poly(w))
    return np.array([h3, h4])


def get_gh_expansion_coeffs(f, varr, dv):
    n = f.shape[0]
    gh = dyn.kinematics.GaussHermite()
    vedg = np.concatenate([varr-dv/2., [varr[-1]+dv/2.]])    
    vel_hist = dyn.kinematics.Histogram(xedg=vedg, y=f[:,:,np.newaxis])
    gh_expansion_coefficients = gh.get_gh_expansion_coefficients(
        v_mu = np.zeros(n),
        v_sig = np.ones(n),
        vel_hist = vel_hist,
        max_order=4)
    h3 = gh_expansion_coefficients[:,0,3]
    h4 = gh_expansion_coefficients[:,0,4]
    return np.vstack((h4, h3))