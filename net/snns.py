import torch
import torch.nn as nn
from net.layers import *

# 根据参数返回所使用的nrom方法
def get_norm(layer, norm_method):
      if layer == 'Linear':
            if norm_method == 'BN':
                  return BN1d
            elif norm_method == 'BNTT':
                  return BNTT1d
            elif norm_method == 'tdBN':
                  return tdBN1d
            elif norm_method == 'TEBN':
                  return TEBN1d
            else:
                 raise ValueError(f'Unexpected norm method  :{norm_method}')
      elif layer == 'Conv2d':
            if norm_method == 'BN':
                  return BN2d
            elif norm_method == 'BNTT':
                  return BNTT2d
            elif norm_method == 'tdBN':
                  return tdBN2d
            elif norm_method == 'TEBN':
                  return TEBN2d
            else:
                  raise ValueError(f'Unexpected norm method  :{norm_method}')
      else:
           raise ValueError(f'Unexpected layer {layer} for norm method  :{norm_method}')
# 根据参数返回所使用的pool方法
def get_pool(pool_method):
      if pool_method == 'max':
            return nn.MaxPool2d(kernel_size=2, stride= 2)
      elif pool_method == 'avg':
            return nn.AvgPool2d(kernel_size=2, stride= 2)
      else:
            raise ValueError(f'Unexpected pool method  :{pool_method}')
    
class SNN(nn.Module):
      def __init__(self, config):
            super().__init__()
            self.config = config

            net_structure = config['net_structure']
            self.net = nn.ModuleList([
                  Block(
                        layer = Conv2d(in_channels=2, out_channels=64,kernel_size=3, stride=1, config=config),
                        norm = get_norm('Conv2d', norm_method=config['norm_method'])(num_features=64, config=config) if net_structure[1] == 'B' else None,
                        neuron = LIF(config=config) if net_structure[2] == 'N' else None,
                  ),
                  Block(
                        layer = get_pool(config['pool_method']),
                        norm = get_norm('Conv2d', norm_method=config['norm_method'])(num_features=64, config=config) if net_structure[4] == 'B' else None,
                        neuron = LIF(config=config) if net_structure[5] == 'N' else None,
                  ),
                  Block(
                        layer = Conv2d(in_channels=64, out_channels=128,kernel_size=3, stride=1, config=config),
                        norm = get_norm('Conv2d', norm_method=config['norm_method'])(num_features=128, config=config) if net_structure[1] == 'B' else None,
                        neuron = LIF(config=config) if net_structure[2] == 'N' else None,
                  ),
                  Block(
                        layer = get_pool(config['pool_method']),
                        norm = get_norm('Conv2d', norm_method=config['norm_method'])(num_features=128, config=config) if net_structure[4] == 'B' else None,
                        neuron = LIF(config=config) if net_structure[5] == 'N' else None,
                  ),
                  nn.Flatten(start_dim=2, end_dim=-1),
                  Block(layer = Linear(in_features=10 * 10 * 128, out_features=256, config=config),
                        norm = get_norm('Linear', norm_method=config['norm_method'])(num_features=256, config=config), 
                        neuron = LIF(config=config)), 

                  Block(layer = Linear(in_features=256, out_features=10, config=config), 
                        norm = get_norm('Linear', norm_method=config['norm_method'])(num_features=10, config=config), 
                        neuron = LIF(config=config)), 
            ])

      def forward(self, input):
            for model in self.net:
                  input = model(input)
            return input.mean(0)