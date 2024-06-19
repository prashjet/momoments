import numpy as np
from scipy.special import comb
from scipy import stats

def sample_l_moments(x):
    """_summary_

    Vectorised version of `_samlmusmall` from the `lmoments3` library. This 
    also required dropping the `exact` keyword from the `comb` function.
    
    """
    n = len(x)
    if n < 3:
        raise ValueError("Insufficient length of data")
    i = np.arange(n)
    x = np.sort(x)
    comb1 = np.arange(n)
    coefl2 = 0.5 / comb(n, 2)
    sum_xtrans = np.sum((comb1 - comb1[::-1]) * x)
    l2 = coefl2 * sum_xtrans
    comb3 = comb(i, 2)
    coefl3 = 1.0 / 3.0 / comb(n, 3)
    sum_xtrans = np.sum((comb3 - 2.*comb1*comb1[::-1] + comb3[::-1]) * x)
    l3 = coefl3 * sum_xtrans / l2
    comb5 = comb(i, 3)
    coefl4 = 0.25 / comb(n, 4)
    sum_xtrans = np.sum(
        (comb5 
         - 3 * comb3 * comb1[::-1] 
         + 3 * comb1 * comb3[::-1] 
         - comb5[::-1])
         * x
         )
    l4 = coefl4 * sum_xtrans / l2
    return np.vstack((l3, l4))

def sample_rl_moments(x):
    l3, l4 = sample_l_moments(x)
    # delta_1_2_2 = get_delta_i_j_k_Phi(1, 2, 2)
    # delta_1_2_3 = get_delta_i_j_k_Phi(1, 2, 3)
    # delta_2_3_4 = get_delta_i_j_k_Phi(2, 3, 4)
    # delta_3_4_4 = get_delta_i_j_k_Phi(3, 4, 4)
    # THESE VALUES TAKEN FROM HARTER 1961, TABULATED VALUES UP TO 5 DP.
    # https://www.jstor.org/stable/2333139?seq=1
    delta_1_2_2 = 2. * 0.56419
    delta_1_2_3 = 0.84628
    delta_2_3_4 = 2. * 0.29701
    delta_3_4_4 = 1.02938 - 0.29701
    l3 = delta_1_2_2/delta_1_2_3 * l3
    l4 = delta_1_2_2/5. * (
        (3./delta_2_3_4 + 2./delta_3_4_4) * l4
        - 3. * (1./delta_2_3_4 - 1./delta_3_4_4)
    )
    return np.vstack((l3, l4))

def integral_rl_moments(f, x_edg):
    """Integral estimator of RL 

    Parameters
    ----------
    f : array, shape (..., nv), 
        Evaluated probability density function. The trailing dimension should
        correspond to the velocity dimension.
    x_edg : array, shape (nv+1,)
        velocity bin edges

    Returns
    -------
    array
        shape (2, ...), where 0'th entry is RL skewness, 1'st is RL kurtosis
    """    
    dx = x_edg[1] - x_edg[0]
    x_c = (x_edg[:-1] + x_edg[1:])/2.
    F = np.cumsum(f*dx, -1)
    R1F = np.polyval(np.array([ 1.7724, -0.8862]), F)
    R2F = np.polyval(np.array([ 7.0896, -7.0896,  1.1816]), F)
    R3F = np.polyval(np.array([ 31.11946096, -46.67919144,  18.29013048,  -1.3652    ]), F)
    lambda2 = np.sum(x_c * f * R1F * dx, -1)
    lambda3 = np.sum(x_c * f * R2F * dx, -1)
    lambda4 = np.sum(x_c * f * R3F * dx, -1)
    tau3 = lambda3/lambda2
    tau4 = lambda4/lambda2
    return np.vstack((tau3, tau4))

def get_lmoment_integral_DF(f, x_edg):
    dx = x_edg[1] - x_edg[0]
    x_c = (x_edg[:-1] + x_edg[1:])/2.
    F = np.cumsum(f*dx, -1)
    F = np.vstack((np.zeros(F.shape[0]), F.T)).T
    dF = F[:,1:] - F[:,:-1]
    Fc = (F[:,:-1] + F[:,1:])/2.
    lmd2 = np.sum(x_c * (2.*Fc - 1) * dF, 1)
    lmd3 = np.sum(x_c * (6.*Fc**2 - 6*Fc + 1.) * dF, 1)
    lmd4 = np.sum(x_c * (20.*Fc**3 - 30*Fc**2 + 12*Fc - 1.) * dF, 1)
    tau3 = lmd3/lmd2
    tau4 = lmd4/lmd2
    return np.vstack((tau3, tau4))

def get_rlmoment_from_lmoment(f, x_edg):
    nrm = stats.norm()
    tau3, tau4 = get_lmoment_integral_DF(f, x_edg)
    # delta_1_2_2 = get_delta_i_j_k_Phi(1, 2, 2)
    # delta_1_2_3 = get_delta_i_j_k_Phi(1, 2, 3)
    # delta_2_3_4 = get_delta_i_j_k_Phi(2, 3, 4)
    # delta_3_4_4 = get_delta_i_j_k_Phi(3, 4, 4)
    # THESE VALUES TAKEN FROM HARTER 1961, TABULATED VALUES UP TO 5 DP.
    # https://www.jstor.org/stable/2333139?seq=1
    delta_1_2_2 = 2. * 0.56419
    delta_1_2_3 = 0.84628
    delta_2_3_4 = 2. * 0.29701
    delta_3_4_4 = 1.02938 - 0.29701
    tau3 = delta_1_2_2/delta_1_2_3 * tau3
    tau4 = delta_1_2_2/5. * (
        (3./delta_2_3_4 + 2./delta_3_4_4) * tau4
        - 3. * (1./delta_2_3_4 - 1./delta_3_4_4)
    )
    return np.vstack((tau3, tau4))



