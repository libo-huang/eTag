#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import os

def parse_common_args(parser):
    parser.add_argument('-data', default='cifar100', help='path to Data Set',
                        choices=['imagenet_sub', 'cifar100', 'imagenet_full '])
    parser.add_argument('-num_class', default=100, type=int, metavar='n', help='dimension of embedding space')
    parser.add_argument('-nb_cl_fg', type=int, default=50, help="Number of class, first group") # TODO
    parser.add_argument('-num_task', type=int, default=5, help="Number of Task after initial Task")  # TODO
    parser.add_argument('-log_dir', type=str, default='./checkpoints/debug', metavar='PATH',   # TODO
                        help='where the models, logs, and events to save')
    parser.add_argument('-dir', default='../a_Data', help='data dir')
    parser.add_argument("-gpu", type=str, default='3', help='which gpu to choose')
    parser.add_argument('-nThreads', '-j', default=4, type=int, metavar='N', help='number of data loading threads')
    parser.add_argument('-BatchSize', '-b', default=128, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('-epochs', default=1, type=int, metavar='N', help='epochs for training process')  # TODO
    parser.add_argument('-epochs_vae', default=1, type=int, metavar='N', help='epochs for training GAN')  # TODO
    parser.add_argument('-seed', default=1, type=int, metavar='N', help='seeds for training process')
    return parser

def parse_train_args(parser):
    parser.add_argument('-tradeoff', type=float, default=1.0, help="tradeoff parameter between feature extractor and classifier")
    parser.add_argument('-lr', type=float, default=1e-3, help="learning rate of backbone network")
    parser.add_argument('-lr_decay', type=float, default=0.1, help='Decay learning rate of backbone network')
    parser.add_argument('-lr_decay_step', type=float, default=200, help='Decay learning rate every x steps of backbone network')
    parser.add_argument('-weight_decay', type=float, default=2e-4, help='weight decay of backbone network')
    parser.add_argument('-vae_tradeoff', type=float, default=1e-3, help='tradeoff parameter of lifelong training vae model')
    parser.add_argument('-vae_lr', type=float, default=0.001, help="learning rate of vae")
    parser.add_argument('-latent_dim', type=int, default=200, help="dimentions of latent variable")
    parser.add_argument('-feat_dim', type=int, default=512, help="dimention of feature")
    parser.add_argument('-hidden_dim', type=int, default=512, help="dimention of hidden linear layer")
    parser.add_argument('-start', default=0, type=int, help='start from which task to train')

    parser.add_argument('-tau', default=3, type=int, help='KD temperature')
    return parser

def parse_test_args(parser):
    parser.add_argument('-top5', action='store_true', help='output top5 accuracy')
    return parser

def get_train_args():
    parser = argparse.ArgumentParser(description='eTag')
    parser = parse_common_args(parser)
    parser = parse_train_args(parser)
    args = parser.parse_args()

    args.log_dir = os.path.join(args.log_dir,  args.data+'_{}tasks_s{}_{}'.format(args.num_task, args.nb_cl_fg, args.seed))


    return args

def get_test_args():
    parser = argparse.ArgumentParser(description='PyTorch Testing')
    parser = parse_common_args(parser)
    parser = parse_test_args(parser)
    args = parser.parse_args()

    args.log_dir = os.path.join(args.log_dir, args.data + '_{}tasks_s{}_{}'.format(args.num_task, args.nb_cl_fg, args.seed))
    return args


if __name__ == '__main__':
    train_args = get_train_args()
    test_args = get_test_args()