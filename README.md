# eTag: Class-Incremental Learning via Hierarchical Embedding Distillation and Task-Oriented

## Introduction

There is the key training and evaluation codes for the AAAI-2024 paper "eTag: Class-Incremental Learning via Hierarchical Embedding Distillation and Task-Oriented Generation".

## Requirement

- python = 3.8.5
- torch = 1.7.1
- torchvision = 0.8.2

## How to run

Please confirm the dataset is well downloaded or softly linked in the file "./dataset" at first.

You can test our method by executing the script we provide, or by running the following command in the path "./scripts"

```sh
# on cifar with 5 tasks
bash -i run.sh cifar 0 5
# on cifar with 10 tasks
bash -i run.sh cifar 0 10
# on cifar with 25 tasks
bash -i run.sh cifar 0 25
```

```sh
# on imagenet_sub with 5 tasks
bash -i run.sh imagenet 0 5
# on imagenet_sub with 10 tasks
bash -i run.sh imagenet 0 10
# on imagenet_sub with 25 tasks
bash -i run.sh imagenet 0 25
```

### arguments

- `-data`: The name of the data you want to test, you can choose from one of them {`cifar100`, `imagenet_sub`} currently.
- `-log_dir`: Where the models, logs, and events to save.
- `-num_task`: Number of tasks after initializing the first task.
- `-nb_cl_fg`: Number of class in the first task.

For more tunable arguments, please take a look at the `opts_eTag.py` file.

## License

**Apache License 2.0**
A permissive license whose main conditions require preservation of copyright and license notices. Contributors provide an express grant of patent rights. Licensed works, modifications, and larger works may be distributed under different terms and without source code.

| Permissions         | Conditions                      | Limitations      |
| ------------------- | ------------------------------- | ---------------- |
| ✔️ Commercial use | ⓘ License and copyright notice | ❌ Trademark use |
| ✔️ Modification   | ⓘ State changes                | ❌ Liability     |
| ✔️ Distribution   |                                 | ❌  Warranty     |
| ✔️ Patent use     |                                 |                  |
| ✔️ Private use    |                                 |                  |
