import numpy as np 
import torch 
from scipy import stats
import sys
sys.path.append('../../')
from dgminv.utils.moments import moment

def kurtosis(a, axis=0, fisher=True, bias=True, nan_policy='propagate'):
    """
    Ported from scipy stats
    Compute the kurtosis (Fisher or Pearson) of a dataset.

    Kurtosis is the fourth central moment divided by the square of the
    variance. If Fisher's definition is used, then 3.0 is subtracted from
    the result to give 0.0 for a normal distribution.

    If bias is False then the kurtosis is calculated using k statistics to
    eliminate bias coming from biased moment estimators

    Use `kurtosistest` to see if result is close enough to normal.

    Parameters
    ----------
    a : array
        Data for which the kurtosis is calculated.
    axis : int or None, optional
        Axis along which the kurtosis is calculated. Default is 0.
        If None, compute over the whole array `a`.
    fisher : bool, optional
        If True, Fisher's definition is used (normal ==> 0.0). If False,
        Pearson's definition is used (normal ==> 3.0).
    bias : bool, optional
        If False, then the calculations are corrected for statistical bias.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default is 'propagate'.

    Returns
    -------
    kurtosis : array
        The kurtosis of values along an axis. If all values are equal,
        return -3 for Fisher's definition and 0 for Pearson's definition.

    References
    ----------
    .. [1] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
       Probability and Statistics Tables and Formulae. Chapman & Hall: New
       York. 2000.

    Examples
    --------
    In Fisher's definiton, the kurtosis of the normal distribution is zero.
    In the following example, the kurtosis is close to zero, because it was
    calculated from the dataset, not from the continuous distribution.

    >>> from scipy.stats import norm, kurtosis
    >>> data = norm.rvs(size=1000, random_state=3)
    >>> kurtosis(data)
    -0.06928694200380558

    The distribution with a higher kurtosis has a heavier tail.
    The zero valued kurtosis of the normal distribution in Fisher's definition
    can serve as a reference point.

    >>> import matplotlib.pyplot as plt
    >>> import scipy.stats as stats
    >>> from scipy.stats import kurtosis

    >>> x = np.linspace(-5, 5, 100)
    >>> ax = plt.subplot()
    >>> distnames = ['laplace', 'norm', 'uniform']

    >>> for distname in distnames:
    ...     if distname == 'uniform':
    ...         dist = getattr(stats, distname)(loc=-2, scale=4)
    ...     else:
    ...         dist = getattr(stats, distname)
    ...     data = dist.rvs(size=1000)
    ...     kur = kurtosis(data, fisher=True)
    ...     y = dist.pdf(x)
    ...     ax.plot(x, y, label="{}, {}".format(distname, round(kur, 3)))
    ...     ax.legend()

    The Laplace distribution has a heavier tail than the normal distribution.
    The uniform distribution (which has negative kurtosis) has the thinnest
    tail.

    """
    # a, axis = _chk_asarray(a, axis)

    # contains_nan, nan_policy = _contains_nan(a, nan_policy)

    # if contains_nan and nan_policy == 'omit':
    #     a = ma.masked_invalid(a)
    #     return mstats_basic.kurtosis(a, axis, fisher, bias)

    # assert(a.dtype == torch.float64), 'kurtosis needs double'
    n = a.shape[axis]
    m2 = moment(a, 2, axis)
    m4 = moment(a, 4, axis)
    # print(m2.dtype)
    # print(m4.dtype)
    zero = (m2 == 0.0)
    # print(zero)
    # with np.errstate(all='ignore'):
    vals = torch.where(zero, 0.0, (m4 / m2**2.0).type(torch.float64)).type(a.dtype)

    # print('vals = ', vals)
    if not bias:
        can_correct = (n > 3) & (m2.detach().cpu().numpy() > 0)
        # print('can_correct = ', can_correct)
        if can_correct.any():
            # m2 = np.extract(can_correct, m2)
            # m4 = np.extract(can_correct, m4)
            # m2 = m2[can_correct]
            # m4 = m4[can_correct]
            nval = 1.0/(n-2)/(n-3) * ((n**2-1.0)*m4/m2**2.0 - 3*(n-1)**2.0)
            # vals[can_correct] = nval + 3.0
            vals = nval + 3.0
    return vals - 3 if fisher else vals


if __name__ == "__main__":
    from scipy.stats import kurtosis as spk
    import numpy as np
    from dgminv.utils.grad_check import GradChecker

    torch.random.manual_seed(0)
    x = torch.randn(1000, dtype=torch.float64).abs()
    
    torch_k = kurtosis(x)
    print(f'torch_k = {torch_k}')
    scipy_k = spk(x.numpy())
    print(f'scipy_k = {scipy_k}')
    print(f'dif = {np.max(np.abs(torch_k.numpy() - scipy_k))}')

    x = np.abs((stats.loggamma.rvs(5, size=1000) + 5))
    th_x = torch.tensor(x).type(torch.float64)

    def fun (x):
        return torch.sum(torch.tanh(kurtosis(x)))


    gc = GradChecker(fun, th_x, mul=1/10.0)
    gc.check()