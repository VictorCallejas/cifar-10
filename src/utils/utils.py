
import numpy as np

import torch

import neptune.new as neptune



def set_deterministic(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    #torch.backends.cudnn.deterministic = True


def config_logs():
    run = neptune.init(project='victorcallejas/CIFAR-APR')
    return run


def get_device(device):

    if torch.cuda.is_available() & (device == 'cuda'):    
    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
        print('Properties:', torch.cuda.get_device_properties(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    return device