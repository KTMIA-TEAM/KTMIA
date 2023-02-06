import argparse
import json
import numpy as np
import pickle
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from base_model import BaseModel
from datasets import get_dataset
from attackers import MiaAttack

parser = argparse.ArgumentParser(description='Membership inference Attacks against collaborative inference')
parser.add_argument('device', default=0, type=int, help="GPU id to use")
parser.add_argument('config_path', default=0, type=str, help="config file path")
parser.add_argument('--dataset_name', default='mnist', type=str)
parser.add_argument('--model_name', default='mnist', type=str)
parser.add_argument('--num_cls', default=10, type=int)
parser.add_argument('--input_dim', default=1, type=int)
parser.add_argument('--image_size', default=28, type=int)
parser.add_argument('--hidden_size', default=128, type=int)
parser.add_argument('--seed', default=3407, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--attacks', default='ktmia_loss', type=str)
parser.add_argument('--transfer_layer_list', default=[], type=list)


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = f"cuda:{args.device}"
    cudnn.deterministic = True
    cudnn.benchmark = False

    if args.transfer_layer_list == []:
        print(args.transfer_layer_list)
        raise ValueError
    base_folder = f"results/{args.dataset_name}_{args.model_name}"
    print(f"Base Folder: {base_folder}")

    # Load Datasets
    trainset = get_dataset(args.dataset_name, train=True)
    testset = get_dataset(args.dataset_name, train=False)
    if testset is None:
        total_dataset = trainset
    else:
        total_dataset = ConcatDataset([trainset, testset])
    total_size = len(total_dataset)
    data_path = f"{base_folder}/data_index.pkl"
    with open(data_path, 'rb') as f:
        victim_train_list, victim_test_list, attack_train_list, attack_test_list, \
            trans_train_list, trans_test_list = pickle.load(f)
    print(f"Total Data Size: {total_size}")

    # Load Victim Dataset
    victim_train_dataset = Subset(total_dataset, victim_train_list)
    victim_test_dataset = Subset(total_dataset, victim_test_list)
    print(f"Victim Train Size: {len(victim_train_list)}, "
          f"Victim Test Size: {len(victim_test_list)}")
    victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                     pin_memory=False)
    victim_test_loader = DataLoader(victim_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=False)

    # Load Transfer Dataset
    trans_train_dataset = Subset(total_dataset, trans_train_list)
    trans_test_dataset = Subset(total_dataset, trans_test_list)
    print(f"Transfer Train Size: {len(trans_train_list)}")
    print(f"Transfer Test Size: {len(trans_test_list)}")
    trans_train_loader = DataLoader(trans_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=False)
    trans_test_loader = DataLoader(trans_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=False)

    # Load Shadow Dataset
    shadow_train_dataset = Subset(total_dataset, attack_train_list)
    shadow_test_dataset = Subset(total_dataset, attack_test_list)
    print(f"Shadow Train Size: {len(attack_train_list)}, "
            f"Shadow Test Size: {len(attack_test_list)}")
    shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                        pin_memory=False)
    shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=False)

    trans_victim_model_list = []
    trans_shadow_model_list = []
    for layer in args.transfer_layer_list:
        save_folder = f"{base_folder}/{layer}"

        # Load Transfer Victim
        trans_victim_model_path = f"{save_folder}/transfer_victim_model/best.pth"
        print(f"Load Transfer Victim Model from {trans_victim_model_path}")
        trans_victim_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
        trans_victim_model.load(trans_victim_model_path)
        trans_victim_model.test(trans_train_loader, "Transfer Victim Model Train")
        trans_victim_model.test(trans_test_loader, "Transfer Victim Model Test")

        # Load Transfer Shadow
        trans_shadow_model_path = f"{save_folder}/transfer_shadow_model/best.pth"
        print(f"Load Transfer Shadow Model From {trans_shadow_model_path}")
        trans_shadow_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
        trans_shadow_model.load(trans_shadow_model_path)
        trans_shadow_model.test(trans_train_loader, "Transfer Shadow Model Train")
        trans_shadow_model.test(trans_test_loader, "Transfer Shadow Model Test")

        trans_victim_model_list.append(trans_victim_model)
        trans_shadow_model_list.append(trans_shadow_model)

    del trans_train_dataset, trans_test_dataset
    del trans_train_loader, trans_test_loader

    attacker = MiaAttack(
        victim_train_loader, victim_test_loader,
        shadow_train_loader, shadow_test_loader,
        trans_victim_model_list=trans_victim_model_list, trans_shadow_model_list=trans_shadow_model_list,
        device=device, num_cls=args.num_cls, epochs=100, batch_size=args.batch_size, 
        lr=0.001, weight_decay=5e-4, optimizer="adam", scheduler="", 
        dataset_name=args.dataset_name, model_name=args.model_name, 
        attack_original=False, exploit_layer_list=args.transfer_layer_list)

    print("Start Membership Inference Attacks")

    attacks = args.attacks.split(',')

    if "ktmia_loss" in attacks:
        kt_loss_acc = attacker.kt_attack("kt_loss")
        print(f"KTMIA_Loss attack accuracy {kt_loss_acc:.3f}")


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config_path) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    print(args)
    main(args)
