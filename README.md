[comment]: <> (# mia_against_cis)
# Membership Inference Attacks against Split Inference via Knowledge Transfer

The repository contains the main code of Membership Inference Attacks against Split Inference via Knowledge Transfer. 
The code is tested on Python 3.8, Pytorch 1.10.1, and Ubuntu 20.04. 
GPUs are needed to accelerate neural network training and membership inference attacks.

# Installation
Install Python packages.
```
pip install -r requirements.txt
```
Enter the repository of Adversary-1:

```
cd adversary-1
```

Create a folder for storing datasets. The data folder location can be updated in `datasets.py`.

```
mkdir -p data/datasets
```
Create a folder for storing the models.
```
mkdir results
```

# Usage

## Attacks
1. Train original target/shadow model:
```
python pretrain.py [GPU-ID] [config_path] 
```
2. Train multiple reconstructed target/shadow models:
```
python knowledge_transfer.py [GPU-ID] [config_path] --mode transfer
```
3. Conduct Knowledge Transfer Membership Inference Attacks.
```
python mia.py [GPU-ID] [config_path] --attacks ktmia
```

# Examples
Train target/shadow ResNet50 models on CIFAR100 dataset using GPU0.
```
python pretrain.py 0 config/cifar100_resnet50/pretrain.json
```
When the split point is at block-1 (layer2 for ResNet50), train multiple reconstructed target/shadow ResNet50 models.
```
python knowledge_transfer.py 0 config/cifar100_resnet50/transfer_fc.json --mode transfer
python knowledge_transfer.py 0 config/cifar100_resnet50/transfer_layer4.json --mode transfer
python knowledge_transfer.py 0 config/cifar100_resnet50/transfer_layer3.json --mode transfer
python knowledge_transfer.py 0 config/cifar100_resnet50/transfer_layer2.json --mode transfer
```
Attack the model using KTMIA.
```
python mia.py 0 config/cifar100_resnet50/transfer_layer2.json --attacks ktmia_loss
```

# Adversary-2 & Adversary-3
Follow the above examples to conduct the attack.

# Acknowledgement
We reuse some code of the official implementation of the USENIX Security 2022 paper [Membership Inference Attacks and Defenses in Neural Network Pruning](https://github.com/Machine-Learning-Security-Lab/mia_prune).