import sys
sys.path.append("../")
import os

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

import torchvision
from torchvision import datasets
from torchvision import datasets, transforms
from PIL import Image

from collections import defaultdict
import random
import numpy as np
from models.resnet import ResNet18

class Helper:
    def __init__(self, config):
        self.config = config
        
        self.config.data_folder = './datasets'
        self.local_model = None
        self.global_model = None
        self.client_models = []
        self.setup_all()

    def setup_all(self):
        self.load_data()
        self.load_model()
        self.config_adversaries()

    def load_model(self):
        self.local_model = ResNet18(num_classes = self.num_classes)
        self.local_model.cuda()
        self.global_model = ResNet18(num_classes = self.num_classes)
        self.global_model.cuda()
        for i in range(self.config.num_total_participants):
            t_model = ResNet18(num_classes = self.num_classes)
            t_model.cuda()
            self.client_models.append(t_model)
        
    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list
    
    def get_train(self, indices):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers=self.config.num_worker)
        return train_loader

    def get_test(self):

        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=self.config.num_worker)

        return test_loader

    def load_data(self):
        self.num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.train_dataset = datasets.CIFAR10(
            self.config.data_folder, train=True, 
            download=True, transform=transform_train)
        self.test_dataset = datasets.CIFAR10(
            self.config.data_folder, train=False, transform=transform_test)
        
        indices_per_participant = self.sample_dirichlet_train_data(
            self.config.num_total_participants,
            alpha=self.config.dirichlet_alpha)
        
        train_loaders = [self.get_train(indices) 
            for pos, indices in indices_per_participant.items()]

        self.train_data = train_loaders
        self.test_data = self.get_test()
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_worker)
    
    def config_adversaries(self):
        if self.config.is_poison:
            self.adversary_list = list(range(self.config.num_adversaries))
        else:
            self.adversary_list = list()