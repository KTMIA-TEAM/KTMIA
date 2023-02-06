import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from base_model import BaseModel
from utils import seed_worker


class MiaAttack:
    def __init__(self, victim_train_loader, victim_test_loader,
                 shadow_train_loader, shadow_test_loader,
                 trans_victim_model_list=None, trans_shadow_model_list=None,
                 device="cuda", num_cls=10, lr=0.001, weight_decay=5e-4,
                 optimizer="adam", scheduler="", epochs=100, batch_size=128, 
                 dataset_name="mnist", model_name="mnist", attack_original=False,
                 exploit_layer_list=[]):
        self.victim_train_loader = victim_train_loader
        self.victim_test_loader = victim_test_loader
        self.shadow_train_loader = shadow_train_loader
        self.shadow_test_loader = shadow_test_loader
        self.device = device
        self.num_cls = num_cls
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.attack_original = attack_original
        self.exploit_layer_list = exploit_layer_list
        self.trans_victim_model_list = trans_victim_model_list
        self.trans_shadow_model_list = trans_shadow_model_list
        self._prepare_si()

    def _prepare_si(self):
        trans_attack_in_losses_list = []
        trans_attack_out_losses_list = []
        trans_victim_in_losses_list = []
        trans_victim_out_losses_list = []
        for trans_shadow_model, trans_victim_model in zip(self.trans_shadow_model_list, self.trans_victim_model_list):
            trans_attack_in_losses, _ = \
                trans_shadow_model.predict_target_loss(self.shadow_train_loader)
            trans_attack_out_losses, _ = \
                trans_shadow_model.predict_target_loss(self.shadow_test_loader)

            trans_victim_in_losses, _ = \
                trans_victim_model.predict_target_loss(self.victim_train_loader)
            trans_victim_out_losses, _ = \
                trans_victim_model.predict_target_loss(self.victim_test_loader)

            trans_attack_in_losses_list.append(trans_attack_in_losses)
            trans_attack_out_losses_list.append(trans_attack_out_losses)
            trans_victim_in_losses_list.append(trans_victim_in_losses)
            trans_victim_out_losses_list.append(trans_victim_out_losses)

        self.attack_in_losses = torch.cat(trans_attack_in_losses_list, dim=1)
        self.attack_out_losses = torch.cat(trans_attack_out_losses_list, dim=1)
        self.victim_in_losses = torch.cat(trans_victim_in_losses_list, dim=1)
        self.victim_out_losses = torch.cat(trans_victim_out_losses_list, dim=1)

    def kt_attack(self, mia_type="kt_loss", model_name="mia_fc"):
        # transfer shadow model
        attack_losses = torch.cat([self.attack_in_losses, self.attack_out_losses], dim=0)
        attack_labels = torch.cat([torch.ones(self.attack_in_losses.size(0)),
                                   torch.zeros(self.attack_out_losses.size(0))], dim=0).unsqueeze(1)

        # transfer victim model
        victim_losses = torch.cat([self.victim_in_losses, self.victim_out_losses], dim=0)
        victim_labels = torch.cat([torch.ones(self.victim_in_losses.size(0)),
                                   torch.zeros(self.victim_out_losses.size(0))], dim=0).unsqueeze(1)

        save_folder = f"results/{self.dataset_name}_{self.model_name}/{self.exploit_layer_list[0]}"
        if not os.path.exists(save_folder):
            raise FileNotFoundError("no pretrained transfer victim/shadow models")

        if mia_type == "kt_loss":
            new_attack_data = attack_losses
            new_victim_data = victim_losses
            attack_model_save_folder = save_folder + "/kt_loss"
        else:
            print(mia_type)
            raise ValueError

        attack_train_dataset = TensorDataset(new_attack_data, attack_labels)
        attack_test_dataset = TensorDataset(new_victim_data, victim_labels)
        attack_train_dataloader = DataLoader(
            attack_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
            worker_init_fn=seed_worker)
        attack_test_dataloader = DataLoader(
            attack_test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
            worker_init_fn=seed_worker)

        if not os.path.exists(attack_model_save_folder):
            os.makedirs(attack_model_save_folder)

        attack_model = BaseModel(
            model_name, device=self.device, save_folder=attack_model_save_folder, num_cls=new_victim_data.size(1), 
            optimizer=self.optimizer, lr=self.lr, weight_decay=self.weight_decay, scheduler=self.scheduler, epochs=self.epochs)

        best_acc = 0
        for epoch in range(self.epochs):
            train_acc, train_loss = attack_model.attack_train(attack_train_dataloader)
            test_acc, test_loss = attack_model.attack_test(attack_test_dataloader)

            if test_acc > best_acc:
                best_acc = test_acc
                save_path = attack_model.save(epoch, test_acc, test_loss)
                best_path = save_path

        return best_acc