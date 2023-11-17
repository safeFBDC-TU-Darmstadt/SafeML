import torch

import util

tensor_functions = {
    'torch': {
        'rand': torch.rand,
        'argmax': torch.argmax,
        'diag': torch.diag,
        'empty_like': torch.empty_like,
        'empty': torch.empty,
        'zeros': torch.zeros,
        'zeros_like': torch.zeros_like,
        'randn': torch.randn,
        'exp': torch.exp,
        'dot': torch.dot,
        'clone': torch.clone,
        'randint': torch.randint,
        'pad': torch.nn.functional.pad
    }
}


tensor_types = {
    'torch': {
        'int32': torch.int32
    }
}


def tensor_function(function_name):
    return tensor_functions[util.constants.tensor_package][function_name]


def tensor_type(type_name):
    return tensor_types[util.constants.tensor_package][type_name]
