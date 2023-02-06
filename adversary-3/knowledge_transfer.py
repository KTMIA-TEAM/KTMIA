import argparse
import json
import numpy as np
import os
import pickle
import random
import copy
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from base_model import BaseModel
from datasets import get_dataset
from utils import seed_worker, get_optimizer, get_scheduler, set_module

parser = argparse.ArgumentParser()
parser.add_argument('device', default=0, type=int, help="GPU id to use")
parser.add_argument('config_path', default=0, type=str, help="config file path")
parser.add_argument('--dataset_name', default="mnist", type=str)
parser.add_argument('--model_name', default="mnist", type=str)
parser.add_argument('--num_cls', default=10, type=int)
parser.add_argument('--input_dim', default=1, type=int)
parser.add_argument('--image_size', default=28, type=int)
parser.add_argument('--hidden_size', default=128, type=int)
parser.add_argument('--seed', default=3407, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--early_stop', default=0, type=int, help="patience for early stopping")
parser.add_argument('--trans_lr', default=0.1, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--trans_optimizer', default="sgd", type=str)
parser.add_argument('--trans_scheduler', default="cosine", type=str)
parser.add_argument('--mode', default="transfer", type=str)
parser.add_argument('--transfer_layer_list', default=[], type=list)


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = f"cuda:{args.device}"
    cudnn.deterministic = True
    cudnn.benchmark = False

    base_folder = f"results/{args.dataset_name}_{args.model_name}"
    print(f"Base Folder: {base_folder}")

    trainset = get_dataset(args.dataset_name, train=True)
    testset = get_dataset(args.dataset_name, train=False)
    aug_trainset = get_dataset(args.dataset_name, train=True, augment=True)
    aug_testset = get_dataset(args.dataset_name, train=False, augment=True)
    if testset is None:
        total_dataset = trainset
        aug_total_dataset = aug_trainset
    else:
        total_dataset = ConcatDataset([trainset, testset])
        aug_total_dataset = ConcatDataset([aug_trainset, aug_testset])
    total_size = len(total_dataset)
    data_path = f"{base_folder}/data_index.pkl"
    print(f"Total Data Size: {total_size}")

    with open(data_path, 'rb') as f:
        _, _, _, _, transfer_train_list, transfer_test_list = pickle.load(f)

    print(f"Transfer Train Size: {len(transfer_train_list)}, "
          f"Transfer Test Size: {len(transfer_test_list)}")
    transfer_train_dataset = Subset(aug_total_dataset, transfer_train_list)
    transfer_test_dataset = Subset(total_dataset, transfer_test_list)

    victim_train_loader = DataLoader(transfer_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                     pin_memory=True, worker_init_fn=seed_worker)
    victim_test_loader = DataLoader(transfer_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=True, worker_init_fn=seed_worker)

    attack_train_loader = DataLoader(transfer_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                     pin_memory=True, worker_init_fn=seed_worker)
    attack_test_loader = DataLoader(transfer_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=True, worker_init_fn=seed_worker)

    victim_model_save_folder = base_folder + "/victim_model"
    if not os.path.exists(f"{victim_model_save_folder}/best.pth"):
        raise FileNotFoundError("no pretrained victim model")
    victim_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
    victim_model.load(f"{victim_model_save_folder}/best.pth")

    shadow_model_save_folder = f"{base_folder}/shadow_model"
    if not os.path.exists(f"{shadow_model_save_folder}/best.pth"):
        raise FileNotFoundError("no pretrained shadow model")
    shadow_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
    shadow_model.load(f"{shadow_model_save_folder}/best.pth")

    tranfer_mode = args.mode.split(',')

    # Train victim models
    print(f"Train {args.mode} Victim Model")

    if "transfer" in tranfer_mode:
        if args.transfer_layer_list == []:
            raise ValueError("transfer_layer_list is empty")

        transfer_victim_model_save_folder = f"{base_folder}/{args.transfer_layer_list[-1]}/transfer_victim_model"
        if not os.path.exists(transfer_victim_model_save_folder):
            os.makedirs(transfer_victim_model_save_folder)
        trans_victim_model = BaseModel(
                args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, 
                save_folder=transfer_victim_model_save_folder, device=device, 
                optimizer=args.trans_optimizer, lr=args.trans_lr, weight_decay=args.weight_decay, 
                scheduler=args.trans_scheduler, epochs=args.epochs)

        print(f"Transfer List: {args.transfer_layer_list}")
        for n, m in victim_model.model.named_modules():
            if n in args.transfer_layer_list:
                transfer_m = copy.deepcopy(m)
                for p in transfer_m.parameters():
                    p.requires_grad = False
                set_module(trans_victim_model.model, n, transfer_m)

        trans_victim_model.optimizer = get_optimizer(args.trans_optimizer, 
                                            filter(lambda p: p.requires_grad, trans_victim_model.model.parameters()), 
                                            lr=args.trans_lr, weight_decay=args.weight_decay)
        trans_victim_model.scheduler = get_scheduler(args.trans_scheduler, 
                                            trans_victim_model.optimizer, args.epochs)

        best_acc = 0
        count = 0
        for epoch in range(args.epochs):
            train_acc, train_loss = trans_victim_model.train(victim_train_loader, f"Epoch {epoch} Transfer Victim Train")
            test_acc, test_loss = trans_victim_model.test(victim_test_loader, f"Epoch {epoch} Transfer Victim Test")
            if test_acc > best_acc:
                best_acc = test_acc
                trans_victim_model.save(epoch, test_acc, test_loss)
                count = 0
            elif args.early_stop > 0:
                count += 1
                if count > args.early_stop:
                    print(f"Early Stop at Epoch {epoch}")
                    break

    # Train shadow models
    print(f"Train {args.mode} Shadow Model")

    if "transfer" in tranfer_mode:
        if args.transfer_layer_list == []:
            raise ValueError("transfer_layer_list is empty")

        transfer_shadow_model_save_folder = f"{base_folder}/{args.transfer_layer_list[-1]}/transfer_shadow_model"
        if not os.path.exists(transfer_shadow_model_save_folder):
            os.makedirs(transfer_shadow_model_save_folder)
        trans_shadow_model = BaseModel(
                args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, 
                save_folder=transfer_shadow_model_save_folder, device=device, 
                optimizer=args.trans_optimizer, lr=args.trans_lr, weight_decay=args.weight_decay, 
                scheduler=args.trans_scheduler, epochs=args.epochs)

        print(f"Transfer List: {args.transfer_layer_list}")
        for n, m in shadow_model.model.named_modules():
            if n in args.transfer_layer_list:
                transfer_m = copy.deepcopy(m)
                for p in transfer_m.parameters():
                    p.requires_grad = False
                set_module(trans_shadow_model.model, n, transfer_m)

        trans_shadow_model.optimizer = get_optimizer(args.trans_optimizer, 
                                            filter(lambda p: p.requires_grad, trans_shadow_model.model.parameters()), 
                                            lr=args.trans_lr, weight_decay=args.weight_decay)
        trans_shadow_model.scheduler = get_scheduler(args.trans_scheduler, 
                                            trans_shadow_model.optimizer, args.epochs)

        best_acc = 0
        count = 0
        for epoch in range(args.epochs):
            train_acc, train_loss = trans_shadow_model.train(attack_train_loader, f"Epoch {epoch} Transfer Shadow Train")
            test_acc, test_loss = trans_shadow_model.test(attack_test_loader, f"Epoch {epoch} Transfer Shadow Test")
            if test_acc > best_acc:
                best_acc = test_acc
                trans_shadow_model.save(epoch, test_acc, test_loss)
                count = 0
            elif args.early_stop > 0:
                count += 1
                if count > args.early_stop:
                    print(f"Early Stop at Epoch {epoch}")
                    break


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config_path) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    print(args)
    main(args)
