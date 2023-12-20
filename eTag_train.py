#!/usr/bin/env Python
# coding=utf-8
from __future__ import absolute_import, print_function
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from models.resnet_aug import resnet18_cifar_aux
from models.resnet_imagenet import resnet18_imagenet_aux

from utils import mkdir_if_missing, logging, display
from torch.optim.lr_scheduler import StepLR
from ImageFolder import *
import torch.utils.data
import torchvision.transforms as transforms
from models.cvae import CVAE_Cifar
from CIFAR100 import CIFAR100
from copy import deepcopy
from opts_eTag import get_train_args
import sys
 
cudnn.benchmark = True

def to_binary(labels,args):
    # Y_onehot is used to generate one-hot encoding
    y_onehot = torch.FloatTensor(len(labels), args.num_class)
    y_onehot.zero_()
    y_onehot.scatter_(1, labels.cpu()[:,None], 1)
    code_binary = y_onehot.cuda()
    return code_binary

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model, state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return model

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

class DistillKL(torch.nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = torch.nn.functional.log_softmax(y_s/self.T, dim=1)
        p_t = torch.nn.functional.softmax(y_t/self.T, dim=1)
        loss = torch.nn.functional.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        return loss

def train_task(args, train_loader, current_task, pre_index=0):
    num_class_per_task = (args.num_class-args.nb_cl_fg) // args.num_task
    if num_class_per_task==0:
        pass  # JT
    else:
        old_task_factor = args.nb_cl_fg // num_class_per_task + current_task - 1
    log_dir = args.log_dir
    mkdir_if_missing(log_dir)

    sys.stdout = logging.Logger(os.path.join(log_dir, 'log_task{}.txt'.format(current_task)))
    tb_writer = SummaryWriter(log_dir)
    display(args)
    # One-hot encoding or attribute encoding
    if 'imagenet' in args.data:
        model = resnet18_imagenet_aux(num_classes=args.num_class)
    elif 'cifar' in args.data:
        model = resnet18_cifar_aux(num_classes = args.num_class)

    if current_task > 0:  # TODO
        model = torch.load(os.path.join(log_dir, 'task_' + str(current_task - 1).zfill(2) + '_%d_model.pkl' % int(args.epochs - 1)))
        model_old = deepcopy(model)
        model_old.eval()
        model_old = freeze_model(model_old)

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
    loss_mse = torch.nn.MSELoss(reduction='sum')

    # Initialize generator and discriminator
    if current_task == 0:
        args.encoder_layer_sizes = [args.feat_dim, args.hidden_dim, args.hidden_dim]
        args.decoder_layer_sizes = [args.hidden_dim, args.hidden_dim, args.feat_dim]
        args.latent_size = args.latent_dim
        args.class_dim = args.num_class
        cvae = CVAE_Cifar(args)
    else:
        cvae = torch.load(os.path.join(log_dir, 'task_' + str(current_task - 1).zfill(2) + '_%d_model_vae.pkl' % int(args.epochs_vae - 1)))
        cvae_old = deepcopy(cvae)
        cvae_old.eval()
        cvae_old = freeze_model(cvae_old)
    cvae = cvae.cuda()

    optimizer_cvae = torch.optim.Adam(cvae.parameters(), lr=vae_lr)

    for p in cvae.parameters():  # set requires_grad to False
        p.requires_grad = False    

    ###############################################################Feature extractor training####################################################
    if current_task > 0:
        model = model.eval()
    if not os.path.exists(os.path.join(log_dir, 'task_' + str(current_task).zfill(2) + '_%d_model.pkl' % (args.epochs-1))):
        for epoch in range(args.epochs):

            loss_log = {'C/loss': 0.0,
                        'C/cls_previous': 0.0,
                        'C/cls_current': 0.0,
                        'C/aug_blockORkd': 0.0,
                        'C/aug_dist': 0.0}
            # scheduler.step()
            for batch, data in enumerate(train_loader, 0):
                inputs1, labels1 = data
                inputs1, labels1 = inputs1.cuda(), labels1.cuda()
                inputs, labels = inputs1, labels1   #!
                size = inputs.shape[1:]
                inputs = torch.stack([torch.rot90(inputs, k, (2, 3)) for k in range(4)], 1).view(-1, *size)
                labels = torch.stack([labels * 4 + i for i in range(4)], 1).view(-1)

                embed_feat, ss_logits = model(inputs, grad=True)

                loss = torch.zeros(1).cuda()
                cls_current = torch.zeros(1).cuda()  # classification loss for the current task
                cls_previous = torch.zeros(1).cuda()  # classification loss for the previous tasks
                aug_blockORkd = torch.zeros(1).cuda()  # augmented loss for the block-wise feature map or the knowledge distillation
                aug_dist = torch.zeros(1).cuda()  # augmented loss for the distance of the last blocks between previous and current model

                optimizer.zero_grad()
                if current_task == 0:
                    soft_feat = model.backbone.embed(embed_feat)
                    for i in range(len(ss_logits)):
                        aug_blockORkd = aug_blockORkd + torch.nn.CrossEntropyLoss()(ss_logits[i], labels)
                    cls_current = torch.nn.CrossEntropyLoss()(soft_feat[0::4], labels1)
                    loss = cls_current + aug_blockORkd
                else:
                    embed_feat_old, ss_logits_old = model_old(inputs)
                    aug_dist = torch.dist(embed_feat, embed_feat_old, 2)
                    aug_dist = old_task_factor * aug_dist
                    for i in range(len(ss_logits)-1):
                        aug_blockORkd = aug_blockORkd + DistillKL(args.tau)(ss_logits[i], ss_logits_old[i])
                    aug_blockORkd = old_task_factor * aug_blockORkd

                    embed_sythesis = []
                    embed_label_sythesis = []
                    ind = list(range(len(pre_index)))
                    for _ in range(args.BatchSize):
                        np.random.shuffle(ind)
                        embed_label_sythesis.append(pre_index[ind[0]])
                    embed_label_sythesis = np.asarray(embed_label_sythesis)
                    embed_label_sythesis = torch.from_numpy(embed_label_sythesis).cuda()

                    z = torch.Tensor(np.random.normal(0, 1, (args.BatchSize, args.latent_dim))).cuda()
                    embed_sythesis = cvae.inference(z, c=embed_label_sythesis)
                    embed_sythesis = torch.cat((embed_feat[0::4],embed_sythesis))
                    embed_label_sythesis = torch.cat((labels1,embed_label_sythesis.cuda()))
                    soft_feat_syt = model.backbone.embed(embed_sythesis)
                    batch_size1 = inputs1.shape[0]
                    cls_current = torch.nn.CrossEntropyLoss()(soft_feat_syt[:batch_size1], embed_label_sythesis[:batch_size1])
                    cls_current = cls_current / (old_task_factor + 1)
                    cls_previous = torch.nn.CrossEntropyLoss()(soft_feat_syt[batch_size1:], embed_label_sythesis[batch_size1:])
                    cls_previous = (cls_previous * old_task_factor) / (old_task_factor + 1)
                    loss = cls_current + cls_previous + (aug_blockORkd + aug_dist) * args.tradeoff
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10000)
                optimizer.step()
                loss_log['C/loss'] += loss.item()
                loss_log['C/cls_current'] += cls_current.item()
                loss_log['C/cls_previous'] += cls_previous.item()
                loss_log['C/aug_blockORkd'] += aug_blockORkd.item()
                loss_log['C/aug_dist'] += aug_dist.item()
                if epoch == 0 and batch == 0:
                    print(50 * '#')
            scheduler.step()
            print('[CLS %05d]\t C/loss: %.3f \t C/cls_current: %.3f \t C/aug_blockORkd: %.3f \t C/cls_previous: %.3f \t C/aug_dist: %.3f'
                % (epoch + 1, loss_log['C/loss'], loss_log['C/cls_current'], loss_log['C/aug_blockORkd'], loss_log['C/cls_previous'], loss_log['C/aug_dist']))
            for k, v in loss_log.items():
                if v != 0:
                    tb_writer.add_scalar('Task {} - Classifier/{}'.format(current_task, k), v, epoch + 1)

            if epoch == args.epochs-1:
                torch.save(model, os.path.join(log_dir, 'task_' + str(current_task).zfill(2) + '_%d_model.pkl' % epoch))
    else:
        model = torch.load(os.path.join(log_dir, 'task_' + str(current_task).zfill(2) + '_%d_model.pkl' % (args.epochs-1)))
        
    ################################################################## CVAE Training stage####################################################
    model = model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in cvae.parameters():
        p.requires_grad = True
    if current_task != args.num_task:
        for epoch in range(args.epochs_vae):
            loss_log = {'V/loss': 0.0, 'V/var': 0.0, 'V/rec': 0.0, 'V/cls_current': 0.0, 'V/cls_previous': 0.0, 'V/aug': 0.0}
            # scheduler_VAE.step()
            for batch, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                real_feat = model.backbone(inputs)
                optimizer_cvae.zero_grad()
                fake_feat, mu, logvar, z = cvae(real_feat, labels)
                loss_rec = torch.nn.MSELoss(reduction='sum')(real_feat, fake_feat) / real_feat.size(0)
                loss_var = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / real_feat.size(0)
                if current_task == 0:
                    loss_aug, cls_previous = torch.zeros(1).cuda(), torch.zeros(1).cuda()
                    fake_feat_soft = model.backbone.embed(fake_feat)
                    cls_current = torch.nn.CrossEntropyLoss()(fake_feat_soft, labels)
                else:
                    # labels of pre-tasks
                    ind = list(range(len(pre_index)))
                    embed_label_sythesis = []
                    for _ in range(args.BatchSize):
                        np.random.shuffle(ind)
                        embed_label_sythesis.append(pre_index[ind[0]])
                    embed_label_sythesis = np.asarray(embed_label_sythesis)
                    embed_label_sythesis = torch.from_numpy(embed_label_sythesis).cuda()
                    z = torch.Tensor(np.random.normal(0, 1, (args.BatchSize, args.latent_dim))).cuda()
                    pre_feat = cvae.inference(z, embed_label_sythesis)
                    fake_feat_soft = model.backbone.embed(fake_feat)
                    cls_current = torch.nn.CrossEntropyLoss()(fake_feat_soft, labels.cuda())
                    cls_current = cls_current / (old_task_factor + 1)
                    fake_feat_soft_ = model.backbone.embed(pre_feat)
                    cls_previous = torch.nn.CrossEntropyLoss()(fake_feat_soft_, embed_label_sythesis.cuda()) * old_task_factor
                    cls_previous = cls_previous / (old_task_factor + 1)
                    pre_feat_old = cvae_old.inference(z, embed_label_sythesis)
                    loss_aug = loss_mse(pre_feat, pre_feat_old) * old_task_factor
                cvae_loss = loss_rec + loss_var + (cls_previous + cls_current) * args.tradeoff + loss_aug * args.vae_tradeoff
                loss_log['V/loss'] += cvae_loss.item()
                loss_log['V/var'] += loss_var.item()
                loss_log['V/rec'] += loss_rec.item()
                loss_log['V/cls_current'] += cls_current.item() * args.tradeoff
                loss_log['V/cls_previous'] += cls_previous.item() * args.tradeoff
                loss_log['V/aug'] += loss_aug.item() * args.vae_tradeoff
                cvae_loss.backward()
                optimizer_cvae.step()
            print('[CVAE %05d]\t V/loss: %.3f \t V/var: %.3f \t V/rec: %.3f \t V/cls_current: %.3f \t V/cls_previous: %.3f \t V/aug: %.3f' %
                  (epoch + 1, loss_log['V/loss'], loss_log['V/var'], loss_log['V/rec'], loss_log['V/cls_current'], loss_log['V/cls_previous'], loss_log['V/aug']))
            for k, v in loss_log.items():
                if v != 0:
                    tb_writer.add_scalar('Task {} - VAE/{}'.format(current_task, k), v, epoch + 1)
        torch.save(cvae, os.path.join(log_dir, 'task_' + str(current_task).zfill(2) + '_%d_model_vae.pkl' % (args.epochs_vae - 1)))
    tb_writer.close()


if __name__ == '__main__':

    args = get_train_args()

    # Data
    print('==> Preparing data..')
    
    if args.data == 'imagenet_sub' or args.data == 'imagenet_full':
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            #transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values)
        ])
        traindir = os.path.join(args.dir, 'ILSVRC12_256', 'train')

    if args.data == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        traindir = args.dir + '/cifar100'

    num_classes = args.num_class 
    num_task = args.num_task
    num_class_per_task = (num_classes-args.nb_cl_fg) // num_task
    
    random_perm = list(range(num_classes))      # multihead fails if random permutaion here
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        
    for i in range(args.start, num_task+1):
        print("-------------------Get started--------------- ")
        print("Training on Task " + str(i))
        if i == 0:
            pre_index = 0
            class_index = random_perm[:args.nb_cl_fg]
        else:
            pre_index = random_perm[:args.nb_cl_fg + (i-1) * num_class_per_task]
            class_index = random_perm[args.nb_cl_fg + (i-1) * num_class_per_task:args.nb_cl_fg + (i) * num_class_per_task]

        if args.data == 'cifar100':
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

            target_transform = np.random.permutation(num_classes)
            trainset = CIFAR100(root=traindir, train=True, download=True, transform=transform_train, target_transform = target_transform, index = class_index)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.BatchSize, shuffle=True, num_workers=args.nThreads,drop_last=True)
        else:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

            trainfolder = ImageFolder(traindir, transform_train, index=class_index)
            train_loader = torch.utils.data.DataLoader(
                trainfolder, batch_size=args.BatchSize,
                shuffle=True,
                drop_last=True, num_workers=args.nThreads)
        train_task(args, train_loader, i, pre_index=pre_index)