import sys
sys.path.append("../")
import wandb
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import datasets
from torchvision import datasets, transforms
from collections import defaultdict, OrderedDict
import random
import numpy as np
from models.resnet import ResNet18
import copy
import os
from sklearn.cluster import DBSCAN

class Aggregator:
    def __init__(self, helper):
        self.helper = helper
        self.Wt = None
        self.krum_client_ids = []

    def agg(self, global_model, weight_accumulator, weight_accumulator_by_client, client_models, sampled_participants):
        if self.helper.config.agg_method == 'avg':
            return self.average_shrink_models(global_model, weight_accumulator)
        elif self.helper.config.agg_method == 'clip':
            self.clip_updates(weight_accumulator)
            return self.average_shrink_models(global_model, weight_accumulator)
        else:
            raise NotImplementedError


    def average_shrink_models(self,  global_model, weight_accumulator):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.
        """
        lr = 1

        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = weight_accumulator[name] * \
                               (1/self.helper.config.num_sampled_participants) * lr
            update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
            data.add_(update_per_layer.cuda())

        return True
    
    def clip_updates(self, agent_updates_dict):
        for key in agent_updates_dict:
            if 'num_batches_tracked' not in key:
                update = agent_updates_dict[key]
                l2_update = torch.norm(update, p=2) 
                update.div_(max(1, l2_update/self.helper.config.clip_factor))
        return
