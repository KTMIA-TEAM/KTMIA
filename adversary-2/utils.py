import numpy as np
import torch
import random
from torch.nn import init


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv') or classname == 'Linear':
        if getattr(m, 'bias', None) is not None:
            init.constant_(m.bias, 0.0)
        if getattr(m, 'weight', None) is not None:
            init.xavier_normal_(m.weight)
    elif 'Norm' in classname:
        if getattr(m, 'weight', None) is not None:
            m.weight.data.fill_(1)
        if getattr(m, 'bias', None) is not None:
            m.bias.data.zero_()


def get_model(model_type, num_cls, input_dim, num_submodule=2):
    if model_type == "resnet18":
        from networks import ResNet18
        model = ResNet18(num_classes=num_cls)
    elif model_type == "resnet50":
        from networks import ResNet50
        model = ResNet50(num_classes=num_cls)
    elif model_type == "vgg16":
        from networks import VGG16
        model = VGG16(num_classes=num_cls)
    elif model_type == "densenet121":
        from networks import DenseNet121
        model = DenseNet121(num_classes=num_cls)
    elif model_type == "columnfc":
        from networks import ColumnFC
        model = ColumnFC(input_dim=input_dim, output_dim=num_cls)
    elif model_type == "mia_fc":
        from models import MIAFC
        model = MIAFC(input_dim=num_cls, output_dim=1)
    else:
        print(model_type)
        raise ValueError
    return model


def get_optimizer(optimizer_name, parameters, lr, weight_decay=0):
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    elif optimizer_name == "":
        optimizer = None
        # print("Do not use optimizer.")
    else:
        print(optimizer_name)
        raise ValueError
    return optimizer


def get_scheduler(scheduler_name, optimizer, epochs):
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs // 2, epochs * 3 // 4], gamma=0.1)
    elif scheduler_name == "":
        scheduler = None
        # print("Do not use scheduler.")
    else:
        print(scheduler_name)
        raise ValueError
    return scheduler


def set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)