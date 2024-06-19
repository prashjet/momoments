import numpy as np
from scipy import stats 
from scipy.special import hermite
from scipy.special import hermitenorm
from scipy import interpolate

def get_hl_moments(f, varr, dv):
    nrm = stats.norm()
    F = np.cumsum(f*dv, -1)
    u_edg = np.linspace(0, 1, 10000)
    du = u_edg[1]
    u = (u_edg[:-1] + u_edg[1:])/2.
    etas = []
    for F0 in F:
        Finv = interpolate.interp1d(F0, varr, bounds_error=False, fill_value=(varr[0], varr[-1]))
        eta2 = np.sum(Finv(u) * hermitenorm(1)(nrm.ppf(u)) * du)
        eta3 = np.sum(Finv(u) * hermitenorm(2)(nrm.ppf(u)) * du)
        eta4 = np.sum(Finv(u) * hermitenorm(3)(nrm.ppf(u)) * du)
        etas += [[eta2, eta3, eta4]]
    etas = np.array(etas)
    tau3 = etas[:,1]/etas[:,0]
    tau4 = etas[:,2]/etas[:,0]
    return np.vstack((tau4, tau3))


def get_hl_moments(f, varr, dv):
    nrm = stats.norm()
    F = np.cumsum(f*dv, -1)
    Phi_inv_F = nrm.ppf(F)
    H1_Phi_inv_F = np.polyval(np.array([ 1., 0.]), Phi_inv_F)
    H1_Phi_inv_F = np.polyval(np.array([ 1., 0., -1.]), Phi_inv_F)
    H3_Phi_inv_F = np.polyval(np.array([ 1., 0., -3., 0.]), Phi_inv_F)
    lambda2 = np.sum(varr * f * H1_Phi_inv_F * dv, -1)
    lambda3 = np.sum(varr * f * H1_Phi_inv_F * dv, -1)
    lambda4 = np.sum(varr * f * H3_Phi_inv_F * dv, -1)
    tau3 = lambda3/lambda2
    tau4 = lambda4/lambda2
    return Phi_inv_F, np.vstack((tau4, tau3))

def get_sample_hlmoment3(x):
    nrm = stats.norm()
    H1 = hermite(1)
    H2 = hermite(2)
    n = len(x)
    x_i_n = np.sort(x)
    z_i_n = nrm.ppf(np.linspace(0, 1, n+2)[1:-1])
    eta2 = np.mean(H1(z_i_n) * x_i_n)
    eta3 = np.mean(H2(z_i_n) * x_i_n)
    return -eta3/eta2