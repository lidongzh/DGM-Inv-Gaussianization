import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required

class Langevin(Optimizer):
    def __init__(self, params, lr=required, seed=1234):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        self.rs = np.random.RandomState(seed)

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """


        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
            # print(f'd_p_list = {d_p_list}')

            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]
                z = torch.from_numpy(self.rs.randn(*d_p.shape)).to(d_p.device).type(d_p.dtype)
                param.add_(d_p, alpha=-lr)
                param.add_(z, alpha=np.sqrt(2.0 * lr))


