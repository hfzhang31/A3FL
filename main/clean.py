import sys
sys.path.append("../")

import wandb
import argparse
import yaml
import traceback

import torch
import torchvision
import numpy as np
import random

from fl_utils.helper import Helper
from fl_utils.fler import FLer

import os

def setup_wandb(config_path, sweep):
    with open(config_path, 'r') as stream:
        sweep_configuration = yaml.safe_load(stream)
    if sweep:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project='FanL-clean')
        return sweep_id
    else:
        config = sweep_configuration['parameters']
        d = dict()
        for k in config.keys():
            v = config[k][list(config[k].keys())[0]]
            if type(v) is list:
                d[k] = {'value':v[0]}
            else:
                d[k] = {'value':v}  
        yaml.dump(d, open('./yamls/tmp.yaml','w'))
        wandb.init(config='./yamls/tmp.yaml')
        return None

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    run = wandb.init()
    set_seed(wandb.config.seed)
    helper = Helper(wandb.config)
    fler = FLer(helper)
    fler.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', default = './yamls/poison.yaml')
    parser.add_argument('--gpu', default = 7)
    parser.add_argument('--sweep', action = 'store_true')
    args = parser.parse_args()
    torch.cuda.set_device(int(args.gpu))
    sweep_id = setup_wandb(args.params, args.sweep)
    if args.sweep:
        wandb.agent(sweep_id, function=main, count=1)
    else:
        main()