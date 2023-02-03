import numpy as np 
import torch 
from scipy import stats
import copy
# from scipy.stats import _contains_nan

def moment(a, moment=1, axis=0, nan_policy='propagate'):
    r"""
    Ported from scipy stats

    Calculate the nth moment about the mean for a sample.

    A moment is a specific quantitative measure of the shape of a set of
    points. It is often used to calculate coefficients of skewness and kurtosis
    due to its close relationship with them.

    Parameters
    ----------
    a : array_like
       Input array.
    moment : int or array_like of ints, optional
       Order of central moment that is returned. Default is 1.
    axis : int or None, optional
       Axis along which the central moment is computed. Default is 0.
       If None, compute over the whole array `a`.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values

    Returns
    -------
    n-th central moment : ndarray or float
       The appropriate moment along the given axis or over all values if axis
       is None. The denominator for the moment calculation is the number of
       observations, no degrees of freedom correction is done.

    See Also
    --------
    kurtosis, skew, describe

    Notes
    -----
    The k-th central moment of a data sample is:

    .. math::

        m_k = \frac{1}{n} \sum_{i = 1}^n (x_i - \bar{x})^k

    Where n is the number of samples and x-bar is the mean. This function uses
    exponentiation by squares [1]_ for efficiency.

    References
    ----------
    .. [1] https://eli.thegreenplace.net/2009/03/21/efficient-integer-exponentiation-algorithms

    Examples
    --------
    >>> from scipy.stats import moment
    >>> moment([1, 2, 3, 4, 5], moment=1)
    0.0
    >>> moment([1, 2, 3, 4, 5], moment=2)
    2.0

    """
    # a, axis = _chk_asarray(a, axis)
    a_np = a.detach().cpu().numpy()
    # contains_nan, nan_policy = _contains_nan(a_np, nan_policy)

    # if contains_nan and nan_policy == 'omit':
    #     a = ma.masked_invalid(a_)
    #     return mstats_basic.moment(a, moment, axis)

    if a.size == 0:
        # empty array, return nan(s) with shape matching `moment`
        if np.isscalar(moment):
            return torch.tensor(np.nan)
        else:
            return torch.tensor(np.full(np.asarray(moment).shape, np.nan, dtype=a_np.dtype))

    # for array_like moment input, return a value for each.
    if not np.isscalar(moment):
        mmnt = [_moment(a, i, axis) for i in moment]
        return torch.tensor(mmnt)
    else:
        return _moment(a, moment, axis)


def _moment(a, moment, axis):
    if np.abs(moment - np.round(moment)) > 0:
        raise ValueError("All moment parameters must be integers")

    if moment == 0:
        # When moment equals 0, the result is 1, by definition.
        shape = list(a.shape)
        del shape[axis]
        if shape:
            # return an actual array of the appropriate shape
            return torch.ones(shape, dtype=a.dtype)
        else:
            # the input was 1D, so return a scalar instead of a rank-0 array
            return torch.tensor(1.0, dtype=a.dtype)

    elif moment == 1:
        # By definition the first moment about the mean is 0.
        shape = list(a.shape)
        del shape[axis]
        if shape:
            # return an actual array of the appropriate shape
            return torch.zeros(shape, dtype=a.dtype)
        else:
            # the input was 1D, so return a scalar instead of a rank-0 array
            return torch.tensor(0.0, dtype=a.dtype)
    else:
        # Exponentiation by squares: form exponent sequence
        n_list = [moment]
        current_n = moment
        while current_n > 2:
            if current_n % 2:
                current_n = (current_n - 1) / 2
            else:
                current_n /= 2
            n_list.append(current_n)

        # Starting point for exponentiation by squares
        a_zero_mean = a - torch.mean(a, dim=axis).unsqueeze(axis)
        if n_list[-1] == 1:
            s = a_zero_mean
        else:
            s = a_zero_mean**2

        # Perform multiplications
        for n in n_list[-2::-1]:
            s = s**2
            if n % 2:
                s *= a_zero_mean
        return torch.mean(s, dim=axis)

if __name__ == "__main__":
    from scipy.stats import moment as spm
    import numpy as np
    import sys
    sys.path.append('../../')
    from dgminv.utils.grad_check import GradChecker

    torch.random.manual_seed(0)
    x = torch.randn(1000, dtype=torch.float64).abs()
    
    torch_m = moment(x, moment=4)
    print(f'torch_m = {torch_m}')
    scipy_m = spm(x.numpy(), moment=4)
    print(f'scipy_m = {scipy_m}')
    print(np.max(np.abs(torch_m.numpy() - scipy_m)))

    x = np.abs((stats.loggamma.rvs(5, size=1000) + 5))
    th_x = torch.tensor(x).type(torch.float64)

    def fun (x):
        return torch.sum(torch.tanh(moment(x, moment=2)))


    gc = GradChecker(fun, th_x, mul=1/10.0)
    gc.check()