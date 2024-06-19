import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy import stats

class MyCosine(stats.rv_continuous):
    "Cosine distribution"
    def __init__(self, *args, scale=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale

    def _pdf(self, x):
        return np.pi/2./self.scale*np.cos(np.pi*x/2./self.scale)

    def _cdf(self, x):
        return np.sin(np.pi*x/2./self.scale)

    def _get_support(self):
        return 0, self.scale

    def _ppf(self, q):
        x = np.arcsin(q) * 2.*self.scale/np.pi
        return x
    
class FlippedDistribution(stats.rv_continuous):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        
    def _pdf(self, x):
        pdf = self.model.pdf(-x)
        return pdf
            
    def _sf(self, x):
        sf = 1. - self.model.sf(-x)
        return sf

    def _cdf(self, x):
        cdf = 1.-self.model.cdf(-x)
        return cdf

    def rvs(self, size):
        samples = self.model.rvs(size=size)
        samples = -samples
        return samples

class MixtureModel(stats.rv_continuous):
    """ Taken from https://stackoverflow.com/a/72315113/11231128
    """
    def __init__(self, submodels, *args, weights = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        if weights is None:
            weights = [1 for _ in submodels]
        if len(weights) != len(submodels):
            raise(ValueError(f'There are {len(submodels)} submodels and {len(weights)} weights, but they must be equal.'))
        self.weights = [w / sum(weights) for w in weights]
        
    def _pdf(self, x):
        pdf = self.submodels[0].pdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            pdf += submodel.pdf(x)  * weight
        return pdf
            
    def _sf(self, x):
        sf = self.submodels[0].sf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            sf += submodel.sf(x)  * weight
        return sf

    def _cdf(self, x):
        cdf = self.submodels[0].cdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            cdf += submodel.cdf(x)  * weight
        return cdf

    def rvs(self, size):
        submodel_choices = np.random.choice(len(self.submodels), size=size, p = self.weights)
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs
    

@dataclass

class BaseKernel(ABC):
    """A generic base kernel"""
    alpha: float
    beta: float
    gamma: float

    def pdf(self,
            a_minus: float,
            a_plus: float,
            w: np.array) -> np.array:
        kernel = self.constuct_kernel(a_minus, a_plus)
        pdf = kernel._pdf(w)
        return pdf

    def rvs(self,
            a_minus,
            a_plus,
            N):
        kernel = self.constuct_kernel(a_minus, a_plus)
        return kernel.rvs(N)
    
    def delta_max_two_sided(self) -> float:
        return self.alpha/np.sqrt(self.beta + self.gamma)

    def delta_max_real(self) -> float:
        """This would allow kernels not 'stiched at w=0' but overlaid one one-side of w=0"""
        return self.alpha/np.sqrt(self.gamma)
    
    def get_half_kernel_scales(self, 
                               sigmaK: float, 
                               delta: float) -> tuple[float, float]:
        Delta = sigmaK*delta/self.alpha
        a = np.sqrt((sigmaK**2. - self.gamma*Delta**2.)/self.beta)
        a_minus = a - Delta
        a_plus = a + Delta
        return a_minus, a_plus

class UniformKernel(BaseKernel):
    
    def __init__(self):
        super().__init__(alpha=0.5, beta=1/3., gamma=1/12.)

    def constuct_kernel(self, a_minus, a_plus):
        kernel = MixtureModel(
            [stats.uniform(loc=-a_minus, scale=a_minus),
             stats.uniform(loc=0., scale=a_plus)],
            weights=[0.5, 0.5]
            )
        return kernel
    
    def evaluate_fft(self,
                     a_minus: float,
                     a_plus: float,
                     u: np.array) -> np.array:
        iu = 1j*u
        iau_min = -a_minus*iu
        iau_pls = a_plus*iu
        Fk = 0.5*(np.exp(iau_min)-1.)/iau_min 
        Fk += 0.5*(np.exp(iau_pls)-1.)/iau_pls
        return Fk
    
class CosineKernel(BaseKernel):
    
    def __init__(self):
        super().__init__(
            alpha=1.-2/np.pi, 
            beta=1.-8/np.pi**2, 
            gamma=4/np.pi*(1-3/np.pi)
        )

    def constuct_kernel(self, a_minus, a_plus):
        kernel = MixtureModel(
            [
                FlippedDistribution(MyCosine(scale=a_minus)),
                MyCosine(scale=a_plus)
            ],
            weights=[0.5, 0.5]
            )
        return kernel

    def evaluate(self, 
                 a_minus: float, 
                 a_plus: float, 
                 w: np.array) -> np.array:
        w = np.atleast_1d(w)
        pi_on_2am = 0.5*np.pi/a_minus
        pi_on_2ap = 0.5*np.pi/a_plus
        k = np.where(
            w<=0., 
            0.5*pi_on_2am*np.cos(pi_on_2am*w), 
            0.5*pi_on_2ap*np.cos(pi_on_2ap*w)
        )
        k[w<-a_minus] = 0.
        k[w>a_plus] = 0.
        return k
    
    def evaluate_fft(self,
                     a_minus: float,
                     a_plus: float,
                     u: np.array) -> np.array:
        iu = 1j*u
        iau_min = -a_minus*iu
        iau_pls = a_plus*iu
        Fk = (np.pi*np.exp(iau_min)-2.*iau_min)/(np.pi**2-4*(a_minus*u)**2)
        Fk += (np.pi*np.exp(iau_pls)-2.*iau_pls)/(np.pi**2-4*(a_plus*u)**2)
        Fk *= np.pi/2.
        return Fk

class LaplaceKernel(BaseKernel):
    
    def __init__(self):
        super().__init__(alpha=1., beta=2., gamma=1.)

    def constuct_kernel(self, a_minus, a_plus):
        kernel = MixtureModel(
            [
                FlippedDistribution(stats.expon(scale=a_minus)),
                stats.expon(scale=a_plus)
            ],
            weights=[0.5, 0.5]
            )
        return kernel

    def evaluate_fft(self,
                     a_minus: float,
                     a_plus: float,
                     u: np.array) -> np.array:
        iu = 1j*u
        Fk = 0.5/(1.+a_minus*iu) + 0.5/(1.-a_plus*iu)
        return Fk

class GaussianKernel(BaseKernel):
    
    def __init__(self):
        super().__init__(
            alpha=(2/np.pi)**0.5, 
            beta=1., 
            gamma=1.-2/np.pi)

    def constuct_kernel(self, a_minus, a_plus):
        kernel = MixtureModel(
            [
                FlippedDistribution(stats.halfnorm(scale=a_minus)),
                stats.halfnorm(scale=a_plus)
            ],
            weights=[0.5, 0.5]
            )
        return kernel
    
    def evaluate(self, 
                 a_minus: float, 
                 a_plus: float, 
                 w: np.array) -> np.array:
        k = np.where(w<=0.,
                     1./a_minus/(2*np.pi)**0.5 * np.exp(-0.5*w**2/a_minus**2),
                     1./a_plus/(2*np.pi)**0.5 * np.exp(-0.5*w**2/a_plus**2))
        return k
    
    def evaluate_fft(self,
                     a_minus: float,
                     a_plus: float,
                     u: np.array) -> np.array:
        au_min = u*a_minus
        au_pls = u*a_plus
        nrm = stats.norm(0,1)
        Fk = np.exp(-0.5*(au_min)**2.)*nrm.cdf(-1j*au_min)
        Fk += np.exp(-0.5*(au_pls)**2.)*nrm.cdf(1j*au_pls)
        Fk[Fk!=Fk]=0. # a hack to remove nans - FT for large u should --> 0?
        return Fk

class Kernel(object):
    
    def __init__(self, 
                 sigmaK=0., 
                 delta=0., 
                 kappa=0., 
                 pkernel_type='uniform',
                 lkernel_type='laplace',
                 which_delta_limit='two_sided'
                 ):
        assert sigmaK >= 0.
        self.sigmaK = sigmaK
        self.pkernel_type = pkernel_type
        self.lkernel_type = lkernel_type
        self.which_delta_limit = which_delta_limit
        self.set_half_kernels()
        assert np.abs(delta) <= self.delta_max
        self.delta = -delta
        self.muK = self.sigmaK*self.delta
        self.pkscales = self.pkernel.get_half_kernel_scales(self.sigmaK, self.delta)
        self.lkscales = self.lkernel.get_half_kernel_scales(self.sigmaK, self.delta)
        self.kappa_weight = (kappa+1)/2.

    def constuct_kernel(self):
        kernel = MixtureModel(
            [
                self.pkernel.constuct_kernel(*self.pkscales),
                self.lkernel.constuct_kernel(*self.lkscales),
            ],
            weights=[1.-self.kappa_weight, self.kappa_weight]
            )
        return kernel
    
    def pdf(self, w: np.array) -> np.array:
        kernel = self.constuct_kernel()
        pdf = kernel._pdf(w)
        return pdf

    def rvs(self, N):
        kernel = self.constuct_kernel()
        return kernel.rvs(N)
    
    def set_half_kernels(self):
        if self.pkernel_type=='uniform':
            self.pkernel = UniformKernel()
        elif self.pkernel_type=='cosine':
            self.pkernel = CosineKernel()
        else:
            raise ValueError('Unknown platykurtic_kernel')
        if self.lkernel_type=='laplace':
            self.lkernel = LaplaceKernel()
        elif self.lkernel_type=='gaussian':
            self.lkernel = GaussianKernel()
        else:
            raise ValueError('Unknown leptokurtic_kernel')
        if self.which_delta_limit == 'real':
            self.delta_max = np.min([
                self.pkernel.delta_max_real(),
                self.lkernel.delta_max_real()
            ])
        elif self.which_delta_limit == 'two_sided':
            self.delta_max = np.min([
                self.pkernel.delta_max_two_sided(),
                self.lkernel.delta_max_two_sided()
            ])
        else:
            raise ValueError('which_delta_limit should be `real` or two_sided`')

    def evaluate(self, w):
        w = np.atleast_1d(w)
        kp = self.pkernel.evaluate(*self.pkscales, w)
        kl = self.lkernel.evaluate(*self.lkscales, w)
        k = (1-self.kappa_weight)*kp + self.kappa_weight*kl
        return k[::-1]

    def evaluate_fft(self, u):
        u = np.atleast_1d(u)
        Fkp = self.pkernel.evaluate_fft(*self.pkscales, u)
        Fkl = self.lkernel.evaluate_fft(*self.lkscales, u)
        Fk = (1.-self.kappa_weight)*Fkp + self.kappa_weight*Fkl
        Fk[u==0.]=1.
        return Fk

class LOSVD(object):
    
    def __init__(self, V: float, sigma: float, kernel: Kernel):
        assert sigma > 0.
        self.kernel = kernel
        self.V = V
        self.sigma = sigma
        self.sigma_rescale = self.sigma/(1.+kernel.sigmaK**2)**0.5
        self.deltaV = kernel.muK*self.sigma_rescale + V

    def rvs(self, N):
        kernel = self.kernel.constuct_kernel()
        y = kernel.rvs(N)
        nrm = stats.norm()
        z = nrm.rvs(N)
        x = y+z
        x *= self.sigma_rescale
        x -= self.deltaV
        return -x
    
    def evaluate_fft(self, u, shift_and_rescale=True):
        def evaluate_fft_base(u):
            if self.kernel.sigmaK>0:
                Fk = self.kernel.evaluate_fft(u)
            else:
                Fk = np.ones_like(u, dtype=np.complex128)
            # use the convention for FT consistent with
            # characteristic function, which is what is used 
            # for the kernel
            # don't forget that scaling the PDF by sigma
            # involves scaling the argument but also the prefactor!
            FTN = np.exp(-0.5*u**2)
            return FTN*Fk
        if shift_and_rescale:
            fft = evaluate_fft_base(u*self.sigma_rescale)
            fft *= np.exp(-1j*self.deltaV*u)
        else:
            fft = evaluate_fft_base(u)
        return fft

    def evaluate_via_fft(self, vmax=10., nv=401, shift_and_rescale=True):
        assert nv%2==1
        v = np.linspace(-vmax, vmax, nv)
        dv = v[1]-v[0]
        n = int((nv-1)/2)
        u = np.linspace(0., np.pi, n)/dv
        Flosvd = self.evaluate_fft(u, shift_and_rescale=shift_and_rescale)
        losvd = np.fft.irfft(Flosvd, v.size)
        losvd = np.roll(losvd, n)/dv
        return v, losvd