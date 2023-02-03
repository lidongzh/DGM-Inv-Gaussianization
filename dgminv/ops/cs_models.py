# Modified from the code from pic-recon: https://github.com/comp-imaging-sci/pic-recon

import sys
import numpy as np 
import torch

class GaussianSensor(torch.nn.Module):
    """ A random gaussian forward operator for simple comressed sensing experiments. 
    """

    def __init__(self, in_shape, subsample_ratio=1./20., is_complex=False, seed=1234):
        super().__init__()
        n = np.prod(in_shape)
        m = int(n * subsample_ratio)

        if is_complex:  
            self.input_dtype = self.output_dtype = np.complex64
            rnd = np.random.RandomState(seed=seed)
            self.value = 1./np.sqrt(2) * (rnd.randn(m,n).astype(np.float32) + 1.j*rnd.randn(m,n).astype(np.float32))
        else:   
            self.input_dtype = self.output_dtype = np.float32
            self.value = np.random.RandomState(seed=seed).randn(m,n).astype(np.float32) / np.sqrt(m)
        
        self.register_buffer('sensor', torch.from_numpy(self.value))

        self.shape = (m,n)
        
    def forward(self, x):
        x1 = x.reshape(-1, 1)
        y1 = torch.matmul(self.sensor.to(x1.device), x1)
        return y1


# from pic-recon/fastMRI/common/supsample
class MaskFunc:
    """
    MaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the k-space data has N
    columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center corresponding to
           low-frequencies
        2. The other columns are selected uniformly at random with a probability equal to:
           prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs).
    This ensures that the expected number of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the MaskFunc object is
    called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there
    is a 50% probability that 4-fold acceleration with 8% center fraction is selected and a 50%
    probability that 8-fold acceleration with 4% center fraction is selected.
    """

    def __init__(self, center_fractions, accelerations):
        """
        Args:
            center_fractions (List[float]): Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly
                each time.

            accelerations (List[int]): Amount of under-sampling. This should have the same length
                as center_fractions. If multiple values are provided, then one of these is chosen
                uniformly each time. An acceleration of 4 retains 25% of the columns, but they may
                not be spaced evenly.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError('Number of center fractions should match number of accelerations')

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None):
        """
        Args:
            shape (iterable[int]): The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.
        Returns:
            torch.Tensor: A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')

        self.rng.seed(seed)
        num_cols = shape[-2]

        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        # Create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        mask = self.rng.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        # mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
        mask = mask.reshape(*mask_shape).astype(np.float32)
        mask = np.concatenate([mask]*shape[-1], axis=-1)

        return mask



class MRISubsampler(object):
    """ Forward model corresponding to MRI subsampling.
    Attributes : 
    `value`       : Values of elements in the forward operator as a matrix.
    `shape`       : matrix shape of the linear operator
    `shapeOI`     : Output-Input shape of the linear operator
    """

    def __init__(self, shapey, shapex, center_fractions=[0.08], accelerations=[4], fftnorm='ortho', loadfile=None, is_complex=False):
        """ Args :
        `shapey` :
        `shapex` : 
        `center_fractions` : Fraction of columns to be fully sampled in the center
        `accelerations` : Speedup on the rest. Refer to https://github.com/facebookresearch/fastMRI/blob/master/common/subsample.py
        `loadfile` : Load mask pattern from a file. If None, generate random lines mask
        """

        # assert shapex[0] == 1 and shapex[-1] == 1
        m = np.prod(shapey)
        n = np.prod(shapex)

        self.shape = (m,n)
        self.shapeOI = (shapey, shapex)
        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.fftnorm = fftnorm
        self.loadfile = loadfile
        if is_complex:
            self.input_dtype = np.complex64
        else:
            self.input_dtype = np.float32
        self.output_dtype = np.complex64

        if self.loadfile:
            print('loadfile\n')
            self.value = np.load(self.loadfile).astype(np.complex64)
            self.value = np.fft.ifftshift(self.value)
        else:
            self.value_img = MaskFunc(center_fractions, accelerations)(shapex)
            self.value = np.fft.ifftshift(self.value_img).astype(np.complex64)
        
        self.value_th = torch.from_numpy(self.value)
    
    def _np(self, x):
        x1 = x
        y = self.value * np.fft.fft2(x1, axes=(-2,-1), norm=self.fftnorm)
        return (y.astype(np.complex64)).reshape(self.shapeOI[0])


    def __call__(self, x):
        x1 = x.to(torch.complex64)
        y1 = self.value_th.to(x1.device) * torch.fft.fft2(x1)

        if self.fftnorm=='ortho':
            return y1.reshape(self.shapeOI[0])/np.sqrt(self.shape[1])
        else:
            return y1.reshape(self.shapeOI[0])

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    shapey = [1, 256, 256]
    shapex = [1, 256, 256]
    mri_samp = MRISubsampler(shapey, shapex)

    th_x = torch.randn(1, 256, 256)
    th_y = mri_samp(th_x)
    y = mri_samp._np(th_x.cpu().numpy())
    # print(th_y)
    # print(y)
    error_rel = np.sum(np.abs(y - th_y.cpu().numpy()))/np.sum(np.abs(y))
    if error_rel < 1e-5:
        print('pass')
    else:
        print('not pass')
    # plt.imshow(np.abs(mri_samp.value[0,...]))
    plt.imshow(mri_samp.value_img[0,...])
    plt.show()
