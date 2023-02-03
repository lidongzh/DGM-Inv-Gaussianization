import numpy as np 
import torch 
from scipy import stats
import sys
sys.path.append('../../')
from dgminv.utils.moments import moment

def skew(a, axis=0, bias=True, nan_policy='propagate'):
    r"""
    Ported from scipy stats
    
    Compute the sample skewness of a data set.
    For normally distributed data, the skewness should be about zero. For
    unimodal continuous distributions, a skewness value greater than zero means
    that there is more weight in the right tail of the distribution. The
    function `skewtest` can be used to determine if the skewness value
    is close enough to zero, statistically speaking.
    Parameters
    ----------
    a : ndarray
        Input array.
    axis : int or None, optional
        Axis along which skewness is calculated. Default is 0.
        If None, compute over the whole array `a`.
    bias : bool, optional
        If False, then the calculations are corrected for statistical bias.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):
          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values
    Returns
    -------
    skewness : ndarray
        The skewness of values along an axis, returning 0 where all values are
        equal.
    Notes
    -----
    The sample skewness is computed as the Fisher-Pearson coefficient
    of skewness, i.e.
    .. math::
        g_1=\frac{m_3}{m_2^{3/2}}
    where
    .. math::
        m_i=\frac{1}{N}\sum_{n=1}^N(x[n]-\bar{x})^i
    is the biased sample :math:`i\texttt{th}` central moment, and :math:`\bar{x}` is
    the sample mean.  If ``bias`` is False, the calculations are
    corrected for bias and the value computed is the adjusted
    Fisher-Pearson standardized moment coefficient, i.e.
    .. math::
        G_1=\frac{k_3}{k_2^{3/2}}=
            \frac{\sqrt{N(N-1)}}{N-2}\frac{m_3}{m_2^{3/2}}.
    References
    ----------
    .. [1] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
       Probability and Statistics Tables and Formulae. Chapman & Hall: New
       York. 2000.
       Section 2.2.24.1
    Examples
    --------
    >>> from scipy.stats import skew
    >>> skew([1, 2, 3, 4, 5])
    0.0
    >>> skew([2, 8, 0, 4, 1, 9, 9, 0])
    0.2650554122698573
    """
    # a, axis = _chk_asarray(a, axis)
    assert(a.dtype == torch.float64), 'skewness needs double'
    n = a.shape[axis]

    # contains_nan, nan_policy = _contains_nan(a, nan_policy)

    # if contains_nan and nan_policy == 'omit':
    #     a = ma.masked_invalid(a)
    #     return mstats_basic.skew(a, axis, bias)

    m2 = moment(a, 2, axis)
    m3 = moment(a, 3, axis)
    # print(f'm2 = {m2}')
    zero = (m2 == 0)
    vals = torch.where(~zero, m3 / m2**1.5,
                      0.)
    if not bias:
        can_correct = (n > 2) & (m2.detach().cpu.numpy() > 0)
        if can_correct.any():
            # m2 = np.extract(can_correct, m2)
            # m3 = np.extract(can_correct, m3)
            nval = torch.sqrt((n - 1.0) * n) / (n - 2.0) * m3 / m2**1.5
            vals = nval
            # np.place(vals, can_correct, nval)

    # if vals.ndim == 0:
    #     return vals.item()

    return vals

if __name__ == "__main__":
    from scipy.stats import skew as ssk
    import numpy as np
    import sys
    sys.path.append('../../')
    from dgminv.utils.grad_check import GradChecker

    torch.random.manual_seed(0)
    x = torch.randn(1000, dtype=torch.float64).abs()
    
    torch_k = skew(x)
    print(f'torch_k = {torch_k}')
    scipy_k = ssk(x.numpy())
    print(f'scipy_k = {scipy_k}')
    print(f'dif = {np.max(np.abs(torch_k.numpy() - scipy_k))}')

    x = np.abs((stats.loggamma.rvs(5, size=1000) + 5))
    th_x = torch.tensor(x).type(torch.float64)

    def fun (x):
        return torch.sum(torch.tanh(skew(x)))


    gc = GradChecker(fun, th_x, mul=1/10.0)
    gc.check()