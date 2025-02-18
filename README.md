# eTag: Class-Incremental Learning via Hierarchical Embedding Distillation and Task-Oriented Generation

## Introduction

This repository contains the key training and evaluation codes for the AAAI-2024 paper titled **"eTag: Class-Incremental Learning via Hierarchical Embedding Distillation and Task-Oriented Generation"**.

## Requirements

To run the code, ensure the following dependencies are installed:

- Python 3.8.5
- PyTorch 1.7.1
- torchvision 0.8.2

## How to Run

### Dataset Preparation
Before running the code, ensure the dataset is properly downloaded or softly linked in the `./dataset` directory.

### Execution
You can test our method by executing the provided scripts or running the following commands in the `./scripts` directory:

#### CIFAR-100 Dataset
```sh
# 5 tasks
bash -i run.sh cifar 0 5
# 10 tasks
bash -i run.sh cifar 0 10
# 25 tasks
bash -i run.sh cifar 0 25
```

#### ImageNet Subset Dataset
```sh
# 5 tasks
bash -i run.sh imagenet 0 5
# 10 tasks
bash -i run.sh imagenet 0 10
# 25 tasks
bash -i run.sh imagenet 0 25
```

### Arguments
- `-data`: Dataset name. Choose from `cifar100` or `imagenet_sub`.
- `-log_dir`: Directory to save models, logs, and events.
- `-num_task`: Number of tasks after the initial task.
- `-nb_cl_fg`: Number of classes in the first task.

For additional tunable arguments, refer to the `opts_eTag.py` file.

## License

This project is licensed under the **Apache License 2.0**.  
A permissive license that requires preservation of copyright and license notices. Contributors provide an express grant of patent rights. Licensed works, modifications, and larger works may be distributed under different terms and without source code.

| Permissions         | Conditions                      | Limitations      |
| ------------------- | ------------------------------- | ---------------- |
| :white_check_mark: Commercial use | ⓘ License and copyright notice | :x: Trademark use |
| :white_check_mark: Modification   | ⓘ State changes                | :x: Liability     |
| :white_check_mark: Distribution   |                                 | :x: Warranty     |
| :white_check_mark: Patent use     |                                 |                  |
| :white_check_mark: Private use    |                                 |                  |

