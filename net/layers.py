import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from net.utils import ste, xnor, shift_based,ste_clip,xnor_clip
from torch.nn.modules.batchnorm import _NormBase
from einops import *
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor

#输入数据为[time, batch, channel, height, width] = [T, B, C, H, W]
class Block(nn.Module):
    def __init__(self, layer, neuron, norm):
        super().__init__()

        self.layer = layer
        self.neuron = neuron
        self.norm = norm

    def forward(self, x):
        # conv or linner or pool
        x  = self._Twrapper(x)
        # norm
        if self.norm is not None:
            x =self.norm(x)
        # neuron
        if self.neuron is not None:
            x = self.neuron(x)           
        return x
    
    # 使pytorch的层支持时间维度
    def _Twrapper(self, x):
        y_shape = [x.shape[0], x.shape[1]]
        y = self.layer(x.flatten(0, 1).contiguous())
        y_shape.extend(y.shape[1:])
        return y.view(y_shape)

# 根据参数返回二值化权重的方法
def get_binary_method(binary_method):
    if binary_method=='ste':
        return ste.apply
    elif binary_method=='xnor':
        return xnor.apply
    elif binary_method=='ste_a':
        return ste_clip.apply
    elif binary_method=='xnor_a':
        return xnor_clip()
    elif binary_method=='none':
        return None
    else:
        raise ValueError(f'Unexpected binary method  :{binary_method}')
    
class Conv2d(nn.Conv2d):
    def __init__(self, config, *args, **kwargs):
        super().__init__(bias= False, *args, **kwargs)
        self.binarize = get_binary_method(config['binarize_weight'])
        
    def forward(self, x):
        if self.binarize is not None:
            return F.conv2d(x, self.binarize(self.weight), self.bias, self.stride, self.padding, self.dilation, self.groups)
        return super().forward(x)
    
class Linear(nn.Linear):
    def __init__(self, config, *args, **kwargs):
        super().__init__(bias= False, *args, **kwargs)
        self.binarize = get_binary_method(config['binarize_weight'])

    def forward(self, x):
        if self.binarize is not None:
            return F.linear(x, self.binarize(self.weight), self.bias)
        return super().forward(x)   

# BatchNorm
class _CustomBatchNorm(_NormBase):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            device=None,
            dtype=None,
            quantize=False
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.quantize = quantize

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        # print(bn_training, self.training)
        if bn_training:
            # print(input.shape, input.unique())
            mean = reduce(input.clone(), 'b c ... -> c', 'mean')
            var = reduce(input.clone(), 'b c ... -> c', reduction=lambda x, dim: torch.var(x, dim, unbiased=False))
            n = input.numel() // input.size(1)
            if self.track_running_stats:
                with torch.no_grad():
                    assert self.running_mean is not None
                    self.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * \
                                        self.running_mean
                    self.running_var = exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * \
                                       self.running_var
                    # print(1)
        else:
            mean, var = self.running_mean, self.running_var

        std = 1. / ((var + self.eps).sqrt())
        if self.quantize:
            std = shift_based(std)

        assert input.ndim >= 2
        pat = 'c->()c' + ''.join(['()'] * (input.ndim - 2))

        input = (input - rearrange(mean, pattern=pat)) * rearrange(std, pattern=pat)
        if self.affine:
            weight = shift_based(self.weight) if self.quantize else self.weight.clone()
            input = input * rearrange(weight, pattern=pat) + rearrange(self.bias, pattern=pat)

        return input

class CustomBatchNorm1d(_CustomBatchNorm):
    def _check_input_dim(self, x):
        if x.dim() != 2 and x.dim() != 3:
            raise ValueError(f'Unexpected input with shape {x.shape}')

class CustomBatchNorm2d(_CustomBatchNorm):
    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError(f'Unexpected input with shape {x.shape}')

class BN1d(CustomBatchNorm1d):
    def __init__(self, config, num_features, *args, **kwargs):
        super().__init__(num_features=num_features, quantize=config['binarize_norm'], *args, **kwargs)
        self.T = config['T']
    
    def forward(self, x):
        y = []
        for t in range(self.T):
            y.append(super().forward(x[t]))
        return torch.stack(y, dim=0)  

class BN2d(CustomBatchNorm2d):
    def __init__(self, config, num_features, *args, **kwargs):
        super().__init__(num_features=num_features, quantize=config['binarize_norm'], *args, **kwargs)
        self.T = config['T']
    
    def forward(self, x):
        y = []
        for t in range(self.T):
            y.append(super().forward(x[t]))
        return torch.stack(y, dim=0)  

class BNTT1d(nn.Module):
    def __init__(self, config, num_features, *args, **kwargs):
        super().__init__()
        self.T = config['T']
        self.layer = nn.ModuleList([CustomBatchNorm1d(num_features=num_features, quantize=config['binarize_norm'], *args, **kwargs) for i in range(self.T)])
       
    
    def forward(self, x):
        y = []
        for t in range(self.T):
            y.append(self.layer[t](x[t]))
        return torch.stack(y, dim=0)  
    
class BNTT2d(nn.Module):
    def __init__(self, config, num_features, *args, **kwargs):
        super().__init__()
        self.T = config['T']
        self.layer = nn.ModuleList([CustomBatchNorm2d( num_features=num_features, quantize=config['binarize_norm'], *args, **kwargs) for i in range(self.T)])
    
    def forward(self, x):
        y = []
        for t in range(self.T):
            y.append(self.layer[t](x[t]))
        return torch.stack(y, dim=0)  

class _tdBN(_CustomBatchNorm):
    def __init__(self, config, alpha=1.0, *args, **kwargs):
        super().__init__(quantize=config['binarize_norm'],*args, **kwargs)
        assert self.affine, "ThresholdDependentBatchNorm needs to set 'affine = True' !"
        nn.init.constant_(self.weight, alpha * config['threshold'])

    def forward(self, x):
        y_shape = x.shape
        y = x.flatten(0, 1)
        y = super().forward(y)
        return y.view(y_shape)  

class tdBN1d(_tdBN):
    def __init__(self, config, alpha=1.0, *args, **kwargs):
        super().__init__(config=config, alpha=alpha, *args, **kwargs)

    def _check_input_dim(self, x):
        if x.dim() != 2 and x.dim() != 3:
            raise ValueError(f'Unexpected input with shape {x.shape}')

class tdBN2d(_tdBN):
    def __init__(self, config, alpha=1.0, *args, **kwargs):
        super().__init__(config=config, alpha=alpha, *args, **kwargs)

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError(f'Unexpected input with shape {x.shape}')
    
class TEBN1d(nn.Module):
    def __init__(self, config, num_features, *args, **kwargs):
        super().__init__()
        self.T = config['T']
        self.layer = CustomBatchNorm1d(num_features=num_features, quantize=config['binarize_norm'], *args, **kwargs)
        self.scale = nn.Parameter(torch.ones([self.T]))

    def forward(self, x):
        y = []
        for t in range(self.T):
            y.append(self.layer(x[t]) * self.scale[t].view([1, 1]))
        return torch.stack(y, dim=0)  

class TEBN2d(nn.Module):
    def __init__(self, config, num_features, *args, **kwargs):
        super().__init__()
        self.T = config['T']
        self.layer = CustomBatchNorm2d(num_features=num_features, quantize=config['binarize_norm'], *args, **kwargs)
        self.scale = nn.Parameter(torch.ones([self.T]))

    def forward(self, x):
        y = []
        for t in range(self.T):
            y.append(self.layer(x[t]) * self.scale[t].view([1, 1, 1, 1]))
        return torch.stack(y, dim=0)  

# 激活函数
def ActFun(gamma=1.):   
    class BaseFun(Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return input.ge(0.).float()

        @staticmethod
        def backward(ctx, grad_output):
            input,  = ctx.saved_tensors
            grad_input = grad_output.clone()
            # 有两种计算temp的方式
            # temp = torch.exp_(-(input).abs() / gamma)
            tmp = (input.abs() < (gamma/2.)).float()
            grad_input = grad_input * tmp
            return grad_input
    return BaseFun.apply

# 神经元（可以扩展实现阈值或膜电压可学习
class LIF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.T = config['T']
        
        self.register_buffer('threshold',torch.tensor(config['threshold']))
        self.register_buffer('decay',torch.tensor(config['decay']))

        self.actfun = ActFun(config['gamma'])

    def forward(self , input):
        v_ = 0.
        spike = []

        for t in range(self.T):
            v_ = v_ * self.decay  + input[t]
            s_ = self.actfun(v_ - self.threshold)
            v_ = v_ * (1. - s_)
            spike.append(s_)
        return torch.stack(spike, dim=0)
    

