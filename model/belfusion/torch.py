import torch
import numpy as np
from torch.optim import lr_scheduler
import torch.nn as nn

# https://raw.githubusercontent.com/Khrylx/DLow/master/utils/torch.py

tensor = torch.tensor
DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
ones = torch.ones
zeros = torch.zeros

class to_cpu:

    def __init__(self, *models):
        self.models = list(filter(lambda x: x is not None, models))
        self.prev_devices = [x.device if hasattr(x, 'device') else next(x.parameters()).device for x in self.models]
        for x in self.models:
            x.to(torch.device('cpu'))

    def __enter__(self):
        pass

    def __exit__(self, *args):
        for x, device in zip(self.models, self.prev_devices):
            x.to(device)
        return False


class to_device:

    def __init__(self, device, *models):
        self.models = list(filter(lambda x: x is not None, models))
        self.prev_devices = [x.device if hasattr(x, 'device') else next(x.parameters()).device for x in self.models]
        for x in self.models:
            x.to(device)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        for x, device in zip(self.models, self.prev_devices):
            x.to(device)
        return False


class to_test:

    def __init__(self, *models):
        self.models = list(filter(lambda x: x is not None, models))
        self.prev_modes = [x.training for x in self.models]
        for x in self.models:
            x.train(False)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        for x, mode in zip(self.models, self.prev_modes):
            x.train(mode)
        return False


class to_train:

    def __init__(self, *models):
        self.models = list(filter(lambda x: x is not None, models))
        self.prev_modes = [x.training for x in self.models]
        for x in self.models:
            x.train(True)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        for x, mode in zip(self.models, self.prev_modes):
            x.train(mode)
        return False


def batch_to(dst, *args):
    return [x.to(dst) if x is not None else None for x in args]


def get_flat_params_from(models):
    if not hasattr(models, '__iter__'):
        models = (models, )
    params = []
    for model in models:
        for param in model.parameters():
            params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(inputs, grad_grad=False):
    grads = []
    for param in inputs:
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            if param.grad is None:
                grads.append(zeros(param.view(-1).shape))
            else:
                grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad


def compute_flat_grad(output, inputs, filter_input_ids=set(), retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    inputs = list(inputs)
    params = []
    for i, param in enumerate(inputs):
        if i not in filter_input_ids:
            params.append(param)

    grads = torch.autograd.grad(output, params, retain_graph=retain_graph, create_graph=create_graph)

    j = 0
    out_grads = []
    for i, param in enumerate(inputs):
        if i in filter_input_ids:
            out_grads.append(zeros(param.view(-1).shape))
        else:
            out_grads.append(grads[j].view(-1))
            j += 1
    grads = torch.cat(out_grads)

    for param in params:
        param.grad = None
    return grads


def set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def filter_state_dict(state_dict, filter_keys):
    for key in list(state_dict.keys()):
        for f_key in filter_keys:
            if f_key in key:
                del state_dict[key]
                break


def get_scheduler(optimizer, policy, nepoch_fix=None, nepoch=None, decay_step=None):
    if policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - nepoch_fix) / float(nepoch - nepoch_fix + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=decay_step, gamma=0.1)
    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', policy)
    return scheduler



def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode="fan_out")
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Linear):
        # print("weights ", module)
        for name, param in module.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_uniform_(param)
    elif (
        isinstance(module, nn.LSTM)
        or isinstance(module, nn.RNN)
        or isinstance(module, nn.LSTMCell)
        or isinstance(module, nn.RNNCell)
        or isinstance(module, nn.GRU)
        or isinstance(module, nn.GRUCell)
    ):
        # https://www.cse.iitd.ac.in/~mausam/courses/col772/spring2018/lectures/12-tricks.pdf
        # • It can take a while for a RNN to learn to remember information
        # • Initialize biases for LSTM’s forget gate to 1 to remember more by default.
        # • Similarly, initialize biases for GRU’s reset gate to -1.
        DIV = 3 if isinstance(module, nn.GRU) or isinstance(module, nn.GRUCell) else 4
        for name, param in module.named_parameters():
            if "bias" in name:
                #print(name)
                nn.init.constant_(
                    param, 0.0
                )  
                if isinstance(module, nn.LSTMCell) \
                    or isinstance(module, nn.LSTM):
                    n = param.size(0)
                    # LSTM: (W_ii|W_if|W_ig|W_io), W_if (forget gate) => bias 1
                    start, end = n // DIV, n // 2
                    param.data[start:end].fill_(1.) # to remember more by default
                elif isinstance(module, nn.GRU) \
                    or isinstance(module, nn.GRUCell):
                    # GRU: (W_ir|W_iz|W_in), W_ir (reset gate) => bias -1
                    end = param.size(0) // DIV
                    param.data[:end].fill_(-1.) # to remember more by default
            elif "weight" in name:
                nn.init.xavier_normal_(param)
                if isinstance(module, nn.LSTMCell) \
                    or isinstance(module, nn.LSTM) \
                    or isinstance(module, nn.GRU) \
                    or isinstance(module, nn.GRUCell):
                    if 'weight_ih' in name: # input -> hidden weights
                        mul = param.shape[0] // DIV
                        for idx in range(DIV):
                            nn.init.xavier_uniform_(param[idx * mul:(idx + 1) * mul])
                    elif 'weight_hh' in name: # hidden -> hidden weights (recurrent)
                        mul = param.shape[0] // DIV
                        for idx in range(DIV):
                            nn.init.orthogonal_(param[idx * mul:(idx + 1) * mul]) # orthogonal initialization https://arxiv.org/pdf/1702.00071.pdf
    else:
        print(f"[WARNING] Module not initialized: {module}")
