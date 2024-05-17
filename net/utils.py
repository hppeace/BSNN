import os
import random
import numpy as np
import torch
import torch.nn.functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

def shift_based(x):
    return x + (x.sgn() * 2 ** x.abs().log2_().round_() - x).detach()

class ste(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        return w.clone().sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
    
class ste_clip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        ctx.save_for_backward(w)
        return w.clone().sign()

    @staticmethod
    def backward(ctx, grad_output):
        w, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.masked_fill_(w.ge(1) | w.le(-1), 0)
        return grad_input
    
class xnor_clip(torch.nn.Module):
    def __init__(self, compute_alpha: bool = True, center_weights: bool = False) -> None:
        super(xnor_clip, self).__init__()
        self.compute_alpha = compute_alpha
        self.center_weights = center_weights

    def _compute_alpha(self, x: torch.Tensor) -> torch.Tensor:
        n = x[0].nelement()
        if x.dim() == 4:
            alpha = x.norm(1, 3, keepdim=True).sum([2, 1], keepdim=True).div(n)
        elif x.dim() == 3:
            alpha = x.norm(1, 2, keepdim=True).sum([1], keepdim=True).div(n)
        elif x.dim() == 2:
            alpha = x.norm(1, 1, keepdim=True).div(n)
        else:
            raise ValueError(f"Expected ndims equal with 2 or 4, but found {x.dim()}")

        return alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.center_weights:
            mean = x.mean(1, keepdim=True)
            x = x.sub(mean)

        if self.compute_alpha:
            alpha = self._compute_alpha(x)
            x = ste_clip.apply(x).mul(alpha)
        else:
            x = ste_clip.apply(x)

        return x

class xnor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        alpha = w.abs().mean([i for i in range(1, len(w.shape))], keepdim=True)
        ctx.save_for_backward(w, alpha)
        return alpha * w.clone().sign()

    @staticmethod
    def backward(ctx, grad_output):
        w, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input_temp1 = grad_input / (w.numel() // w.shape[0])
        temp = w.abs() < 1.
        grad_input_temp2 = alpha * grad_input * temp.float()
        return grad_input_temp1 + grad_input_temp2
