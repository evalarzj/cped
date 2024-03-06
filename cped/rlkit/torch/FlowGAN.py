import torch
import numpy as np
from torch import nn as nn
from torch.nn import functional as F
from torch.backends import cudnn
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.modules import LayerNorm


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed) # sets the seed for generating random numbers.
    torch.cuda.manual_seed(seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed)# Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

    torch.backends.cudnn.deterministic = True  #cpu/gpu result consistent
    torch.backends.cudnn.benchmark = True   # speed up the training process when train set has no significant change.

# set random seed
#setup_seed(0)


# Abstract class that can propagate both forward/backward,
# along with jacobians.

def batch_norm(input_,
                train=True,
                epsilon=1e-6,
                decay=.1,
                bn_lag=0.,
                device=None,
                scaling = True):
  """Batch normalization with corresponding log determinant Jacobian."""
  var=torch.tensor(1.,dtype=torch.float32)
  mean = torch.tensor(0., dtype=torch.float32)
  step = torch.tensor(0., dtype=torch.float32)
  if scaling:
      scale_g = nn.Parameter(torch.tensor(1., dtype=torch.float32,requires_grad=True))
      shift_b = nn.Parameter(torch.tensor(0., dtype=torch.float32,requires_grad=True))
# choose the appropriate moments

  if train:
      used_mean = input_.mean(-1, keepdim=True)
      used_var = input_.std(-1, keepdim=True)
      cur_mean, cur_var = used_mean, used_var
      if bn_lag > 0.:
          #used_var = stable_var(input_=input_, mean=used_mean, axes=axes)
          cur_var = used_var
          used_mean -= (1 - bn_lag) * (used_mean - mean)
          used_mean /= (1. - bn_lag ** (step + 1))
          used_var -= (1 - bn_lag) * (used_var - var)
          used_var /= (1. - bn_lag ** (step + 1))
  else:
        used_mean, used_var = mean, var
        cur_mean, cur_var = used_mean, used_var
  if train:
      if not torch.isnan(decay * (mean - cur_mean)):
          new_mean=mean-decay * (mean - cur_mean)
      else:
          new_mean = mean
      if not torch.isnan(decay * (var - cur_var)):
          new_var=var-decay * (var - cur_var)
      else:
          new_var = var
      new_step=step+1.
      used_var += 0. * new_mean * new_var * new_step
  used_var += epsilon
  if scaling:
      return ((input_ - used_mean) / torch.sqrt(used_var)) * scale_g + shift_b
  else:
      return ((input_ - used_mean) / torch.sqrt(used_var))


def simple_batch_norm(x):
    mu = torch.mean(x)
    sig2 = torch.mean(torch.square(x-mu))
    x = (x-mu)/torch.sqrt(sig2 + 1.0e-6)
    return x

def get_weight(fc,init_type):
    if isinstance(fc, (nn.Conv2d, nn.Linear)):
        if init_type == "uniform":
            nn.init.uniform_(fc.weight,-0.01, 0.01)
        elif init_type == "normal":
            nn.init.normal_(fc.weight, 0, 0.02)
        elif init_type == "orthogonal":
            nn.init.orthogonal(fc.weight)
        elif init_type == "xavier":
            nn.init.xavier_uniform_(fc.weight)


class NICE(nn.Module):
    # |mask_type| can be 'checkerboard0', 'checkerboard1'
    def __init__(self,mask_type,xdim,seed=0,init_type= "uniform",hidden_size=750, no_of_layers=4,hidden_layer=2):
        super().__init__()
        self.mask_type = mask_type
        self.seed = seed
        self.init_type = init_type
        self.hidden_states = hidden_size
        self.hidden_layer=hidden_layer
        self.no_of_layers = no_of_layers
        self.xdim=xdim
        self.seed=seed
        self.alpha=1e-7

        setup_seed(self.seed)
        self.fcs = nn.ModuleList()
        self.m_network()
        self.fcs = self.fcs.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.sum_log_det_jacobians = 0
        self.scale_factor = nn.Parameter(ptu.zeros(1, xdim, dtype=torch.float32, requires_grad=True))
        
    def m_network(self):
            if self.mask_type == 'checkerboard0':
               in_size=int(self.xdim/2)
               out_size=self.xdim-in_size

            else:
               out_size=int(self.xdim/2)
               in_size=self.xdim-out_size
            
            for r in range(self.no_of_layers):
                self.seed=self.seed+1
                setup_seed(self.seed)
                tempsize=in_size
                for i in range(self.hidden_layer):
                   fc = nn.Linear(in_size, self.hidden_states)
                   in_size = self.hidden_states
                   # initailize the weight and bias of the layer of m network
                   get_weight(fc, self.init_type)
                   nn.init.constant_(fc.bias, 0.0)
                   self.fcs.append(fc)
                   
                fcn = nn.Linear(self.hidden_states,out_size)
                get_weight(fcn, self.init_type)
                nn.init.constant_(fcn.bias, 0.0)
                self.fcs.append(fcn)
                
                in_size=out_size
                out_size=tempsize
                   
        

    # performs the operation described in the NICE paper
    def function_l_m(self, x,start,end,train=False):
        y = x
        index=x.size()[1]
        #activation of the layer(no last layer) of m network
        for i, fc in enumerate(self.fcs):
            if i>=start and i<end:
               y = batch_norm(input_=y, train=train)
               #print('inputs on cuda: ', y.is_cuda)
               if i <=end-1:
                  y = F.leaky_relu(fc(y))
               else:
                  y=F.tanh(fc(y))
                  #y = F.leaky_relu(fc(y))
        return y

    def forward(self,*inputs,train=False):
        x = torch.cat(inputs, dim=1)
        setup_seed(self.seed)

        y = x
        start=0
        end=self.hidden_layer+1
        for i in range(self.no_of_layers):
            split_value = int(int_shape(x)[1] / 2)
            x1 = x[:, :split_value]
            x2 = x[:, split_value:]
            
            if self.mask_type == 'checkerboard0':
                mx1 = self.function_l_m(x1,start,end,train=False)
                delta = torch.concat([torch.zeros_like(x1), mx1], 1)
                self.mask_type = 'checkerboard1'
                start=end
                end=end+self.hidden_layer+1
            else:
                mx2 = self.function_l_m(x2,start,end,train=False)
                delta = torch.concat([mx2, torch.zeros_like(x2)], 1)
                self.mask_type = 'checkerboard0'
                start=end
                end=end+self.hidden_layer+1
                
            y =y+ delta
            x=y
        
        y = torch.mul(y, torch.exp(self.scale_factor))

        return y, self.sum_log_det_jacobians+torch.sum(self.scale_factor)

    def inverse(self,y,train=False):
        setup_seed(self.seed)
        y=y.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        x = y
        end=(self.hidden_layer+1)*(self.no_of_layers)
        start=end-self.hidden_layer-1
        
        for i in range(self.no_of_layers):
            split_value = int(int_shape(x)[1] / 2)
            y1 = y[:, :split_value]
            y2 = y[:, split_value:]
            if self.mask_type == 'checkerboard0':
                my2 = self.function_l_m(y2,start,end, train=train)
                delta = torch.concat([my2, torch.zeros_like(y2)], 1)
                self.mask_type = 'checkerboard1'
                end=start
                start=start-self.hidden_layer-1
            else:
                my1 = self.function_l_m(y1,start,end, train=train)
                delta = torch.concat([torch.zeros_like(y1), my1], 1)
                self.mask_type = 'checkerboard0'
                end=start
                start=start-self.hidden_layer-1
            x = x - delta
            y = x
        x = torch.mul(x, torch.exp(-self.scale_factor))

        return x





 # Weight normalization technique
def get_normalized_weights(fc, scale=False):
    nn.init.xavier_uniform_(fc.weight)
    #nn.init.kaiming_uniform_(fc.weight.data)
    fc=nn.utils.weight_norm(fc)
    return (fc)


class RealNVP(nn.Module):
    def __init__(self, mask_type,xdim, scaling=True,seed=0,hidden_size=750,no_of_layers=4,hidden_layer=2):
        super().__init__()
        self.mask_type = mask_type
        self.seed = seed
        self.scaling = scaling
        self.no_of_layers=no_of_layers
        self.hidden_states = hidden_size
        self.hidden_layer=hidden_layer
        self.xdim=xdim



        if self.scaling == False:
            print("No scaling")

        setup_seed(self.seed)
       
        
        self.fcs = nn.ModuleList()
        self.st_network()
        self.fcs = self.fcs.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        
    def st_network(self):
            if self.mask_type == 'checkerboard0':

               in_size=int(self.xdim/2)*2
               out_size=(self.xdim-int(self.xdim/2))*2

            else:
               out_size=int(self.xdim/2)*2
               in_size=(self.xdim-int(self.xdim/2))*2
            
            for r in range(self.no_of_layers):
                tempsize=in_size
                for i in range(self.hidden_layer):
                   fc = nn.Linear(in_size, self.hidden_states)
                   in_size = self.hidden_states
                   # initailize the weight and bias of the layer of m network
                   fc=get_normalized_weights(fc)
                   nn.init.constant_(fc.bias, 0.0)
                   self.fcs.append(fc)
                   
                fcn = nn.Linear(self.hidden_states,out_size)
                fcn=get_normalized_weights(fcn)
                nn.init.constant_(fcn.bias, 0.0)
                self.fcs.append(fcn)
                
                in_size=out_size
                out_size=tempsize
                   
        
        
    def function_s_t(self, x,start,end, train=False):
        y = x
        
        if not self.scaling:
            y = simple_batch_norm(y)
        else:
            y = batch_norm(input_=y,train=train)

        for i, fc in enumerate(self.fcs):
            if i>=start and i<end:
               if not self.scaling:
                      y = simple_batch_norm(y)
               else:
                      y = batch_norm(input_=y, train=train)
               #print('inputs on cuda: ', y.is_cuda)
               y = F.relu(fc(y))


        if self.scaling:
           y = batch_norm(input_=y,train=train)

        (s_fun, t_fun) = y.split(int(y.shape[1]/2), dim=1)
        return s_fun, t_fun

    def forward(self,  *inputs,train=False):
        """Forward pass.
        Args:
            x: input tensor.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """
        x=torch.cat(inputs, dim=1)
        sum_log_det_jacobians=0
        y = x
        start=0
        end=self.hidden_layer+1
        
        for i in range(self.no_of_layers):
            split_value=int(int_shape(x)[1]/2)
            b = torch.cat([ptu.ones(split_value), ptu.zeros(self.xdim-split_value)], 0)
            x1 = x[:, :split_value]
            x2 = x[:, split_value:]
            if self.mask_type == 'checkerboard0':
                x1_t=torch.concat([x1,x1], 1)
                sx1,tx1 = self.function_s_t(x1_t,start,end,train=train)
                s=torch.concat([torch.zeros_like(x1),sx1], 1)
                t = torch.concat([torch.zeros_like(x1), tx1], 1)
                y = x * b + torch.mul(x*(1-b), torch.exp(s)) + t
                log_det_jacobian = torch.sum(sx1, 1)
                sum_log_det_jacobians += log_det_jacobian
                self.mask_type='checkerboard1'
                start=end
                end=end+self.hidden_layer+1
            else:
                x2_t = torch.concat([x2, x2], 1)
                sx2,tx2 = self.function_s_t(x2_t, start,end,train=train)
                s = torch.concat([sx2,torch.zeros_like(x2)], 1)
                t = torch.concat([tx2,torch.zeros_like(x2)], 1)
                y = x * (1-b) + torch.mul(x*b, torch.exp(s)) + t
                log_det_jacobian = torch.sum(sx2, 1)
                
                sum_log_det_jacobians += log_det_jacobian
                self.mask_type = 'checkerboard0'
                start=end
                end=end+self.hidden_layer+1
            x=y
        return y, sum_log_det_jacobians

    def inverse(self,y,train=False):
        #setup_seed(self.seed)
        y=y.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        x = y
        end=(self.hidden_layer+1)*(self.no_of_layers)
        start=end-self.hidden_layer-1
        for i in range(self.no_of_layers):
            split_value = int(int_shape(y)[1] / 2)
            b = torch.concat([torch.ones(split_value), torch.zeros(self.xdim-split_value)], 0).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            y1 = y[:, :split_value]
            y2 = y[:, split_value:]
            if self.mask_type == 'checkerboard0':
                y2_t = torch.concat([y2, y2], 1)
                sy2, ty2 = self.function_s_t(y2_t,start,end, train=train)
                s = torch.concat([sy2, torch.zeros_like(y2)], 1)
                t = torch.concat([ty2, torch.zeros_like(y2)], 1)
                x = y * (1 - b) + torch.mul(y*b-t, torch.exp(-s))
                self.mask_type = 'checkerboard1'
                end=start
                start=start-self.hidden_layer-1
            else:
                y1_t = torch.concat([y1, y1], 1)
                sy1, ty1 = self.function_s_t(y1_t,start,end, train=train)
                s = torch.concat([torch.zeros_like(y1), sy1], 1)
                t = torch.concat([torch.zeros_like(y1), ty1], 1)
                x = y * b+ torch.mul(y*(1-b)-t, torch.exp(-s))
                self.mask_type = 'checkerboard0'
                end=start
                start=start-self.hidden_layer-1
            y = x
        return x


def int_shape(x):
    return list(map(int, x.size()))

# Given the output of the network and all jacobians,
# compute the log probability.
def compute_log_density_x(z, sum_log_det_jacobians, prior):

  zs = int_shape(z)
  if len(zs) == 4:
    K = zs[1]*zs[2]*zs[3] #dimension of the Gaussian distribution
    z = torch.reshape(z, (-1, K))
  else:
    K = zs[1]
  log_density_z = 0
  if prior == "gaussian":
    log_density_z = -0.5*torch.sum(torch.square(z), [1]) -0.5*K*np.log(2*np.pi)
  elif prior == "logistic":
    log_density_z = -torch.sum(-z + 2*int(torch.nn.Softplus(z)),[1])
  elif prior == "uniform":
    log_density_z = 0
  log_density_x = log_density_z + sum_log_det_jacobians

  return log_density_x


# Computes log_likelihood of the network
def log_likelihood(z, sum_log_det_jacobians, prior):
  return -torch.sum(compute_log_density_x(z, sum_log_det_jacobians, prior))

def identity(x):
    return x

class Discriminator(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.leaky_relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            drop_rate=0.0,
            layer_norm=True,
            layer_norm_kwargs=None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.drop_rate = drop_rate
        in_size = input_size


        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)
        self.fcs = self.fcs.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.layer_norms=self.layer_norms.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)
        self.last_fc = self.last_fc.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, *inputs, return_preactivations=False):
        h = torch.cat(inputs, dim=1)
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs):
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
            h = F.dropout(h,p=self.drop_rate)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        logit=torch.sigmoid(output)

        return output,output
