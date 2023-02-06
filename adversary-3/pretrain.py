import argparse
import json
import numpy as np
import os
import pickle
import random
import torch
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Subset
from base_model import BaseModel
from datasets import get_dataset
from utils import seed_worker

parser = argparse.ArgumentParser()
parser.add_argument('device', default=0, type=int, help="GPU id to use")
parser.add_argument('config_path', default=0, type=str, help="config file path")
parser.add_argument('--dataset_name', default='mnist', type=str)
parser.add_argument('--model_name', default='mnist', type=str)
parser.add_argument('--num_cls', default=10, type=int)
parser.add_argument('--input_dim', default=1, type=int)
parser.add_argument('--image_size', default=28, type=int)
parser.add_argument('--hidden_size', default=128, type=int)
parser.add_argument('--seed', default=3407, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--early_stop', default=0, type=int, help="patience for early stopping")
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--optimizer', default="sgd", type=str)
parser.add_argument('--scheduler', default="cosine", type=str)


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = f"cuda:{args.device}"
    cudnn.deterministic = True
    cudnn.benchmark = False

    save_folder = f"results/{args.dataset_name}_{args.model_name}"
    print(f"Save Folder: {save_folder}")

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
    data_path = f"{save_folder}/data_index.pkl"
    print(f"Total Data Size: {total_size}")

    # Random split total dataset into victim dataset, shadow dataset and transfer dataset. 
    # Victim : Shadow : Transfer = 1 : 1 : 1
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    victim_list, attack_list = train_test_split(list(range(total_size)), test_size=2/3, random_state=args.seed)
    attack_list, transfer_list = train_test_split(attack_list, test_size=0.5, random_state=args.seed)
    victim_train_list, victim_test_list = train_test_split(victim_list, test_size=0.5, random_state=args.seed)
    attack_train_list, attack_test_list = train_test_split(attack_list, test_size=0.5, random_state=args.seed)
    transfer_train_list, transfer_test_list = train_test_split(transfer_list, test_size=0.5, random_state=args.seed)
    with open(data_path, 'wb') as f:
        pickle.dump([victim_train_list, victim_test_list, 
                     attack_train_list, attack_test_list, 
                     transfer_train_list, transfer_test_list], f)

    # Train victim model
    print(f"Victim Train Size: {len(victim_train_list)}, "
          f"Victim Test Size: {len(victim_test_list)}")
    victim_train_dataset = Subset(aug_total_dataset, victim_train_list)
    victim_test_dataset = Subset(total_dataset, victim_test_list)
    victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                     pin_memory=True, worker_init_fn=seed_worker)
    victim_test_loader = DataLoader(victim_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=True, worker_init_fn=seed_worker)

    victim_model_save_folder = save_folder + "/victim_model"
    print("Train Victim Model")
    if not os.path.exists(victim_model_save_folder):
        os.makedirs(victim_model_save_folder)
    victim_model = BaseModel(
        args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, save_folder=victim_model_save_folder,
        device=device, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, epochs=args.epochs)
    best_acc = 0
    count = 0
    for epoch in range(args.epochs):
        train_acc, train_loss = victim_model.train(victim_train_loader, f"Epoch {epoch} Train")
        test_acc, test_loss = victim_model.test(victim_test_loader, f"Epoch {epoch} Test")
        if test_acc > best_acc:
            best_acc = test_acc
            victim_model.save(epoch, test_acc, test_loss)
            count = 0
        elif args.early_stop > 0:
            count += 1
            if count > args.early_stop:
                print(f"Early Stop at Epoch {epoch}")
                break

    # Train shadow model
    print(f"Shadow Train Size: {len(attack_train_list)}, "
          f"Shadow Test Size: {len(attack_test_list)}")
    attack_train_dataset = Subset(aug_total_dataset, attack_train_list)
    attack_test_dataset = Subset(total_dataset, attack_test_list)
    attack_train_loader = DataLoader(attack_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                     pin_memory=True, worker_init_fn=seed_worker)
    attack_test_loader = DataLoader(attack_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=True, worker_init_fn=seed_worker)

    shadow_model_save_folder = f"{save_folder}/shadow_model"
    print(f"Train Shadow Model")
    if not os.path.exists(shadow_model_save_folder):
        os.makedirs(shadow_model_save_folder)
    shadow_model = BaseModel(
            args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, save_folder=shadow_model_save_folder,
            device=device, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, epochs=args.epochs)
    best_acc = 0
    count = 0
    for epoch in range(args.epochs):
        train_acc, train_loss = shadow_model.train(attack_train_loader, f"Epoch {epoch} Shadow Train")
        test_acc, test_loss = shadow_model.test(attack_test_loader, f"Epoch {epoch} Shadow Test")
        if test_acc > best_acc:
            best_acc = test_acc
            shadow_model.save(epoch, test_acc, test_loss)
            count = 0
        elif args.early_stop > 0:
            count += 1
            if count > args.early_stop:
                print(f"Early Stop at Epoch {epoch}")
                break

    # Train trivial model
    print(f"Transfer Train Size: {len(transfer_train_list)}, "
          f"Transfer Test Size: {len(transfer_test_list)}")
    trivial_train_dataset = Subset(aug_total_dataset, transfer_train_list)
    trivial_test_dataset = Subset(total_dataset, transfer_test_list)
    trivial_train_loader = DataLoader(trivial_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                      pin_memory=True, worker_init_fn=seed_worker)
    trivial_test_loader = DataLoader(trivial_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                     pin_memory=True, worker_init_fn=seed_worker)

    trivial_model_save_folder = f"{save_folder}/trivial_model"
    print(f"Train Trivial Model")
    if not os.path.exists(trivial_model_save_folder):
        os.makedirs(trivial_model_save_folder)
    trivial_model = BaseModel(
            args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, 
            save_folder=trivial_model_save_folder, device=device, 
            optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, 
            scheduler=args.scheduler, epochs=args.epochs)
    best_acc = 0
    count = 0
    for epoch in range(args.epochs):
        train_acc, train_loss = trivial_model.train(trivial_train_loader, f"Epoch {epoch} Trivial Train")
        test_acc, test_loss = trivial_model.test(trivial_test_loader, f"Epoch {epoch} Trivial Test")
        if test_acc > best_acc:
            best_acc = test_acc
            trivial_model.save(epoch, test_acc, test_loss)
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