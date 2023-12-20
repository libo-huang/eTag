# coding=utf-8
from __future__ import absolute_import, print_function
from torch.backends import cudnn
from evaluations import extract_features_classification_aug
import torchvision.transforms as transforms
from ImageFolder import *
from utils import *
from CIFAR100 import CIFAR100
from utils import mkdir_if_missing
from opts_eTag import get_test_args

cudnn.benchmark = True
args = get_test_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
models = []
for i in os.listdir(args.log_dir):
    if i.endswith("_%d_model.pkl" % (args.epochs - 1)):  # 500_model.pkl
        models.append(os.path.join(args.log_dir, i))

models.sort() 

if 'cifar' in args.data:
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    testdir = args.dir + '/cifar100'

if args.data == 'imagenet_sub' or args.data == 'imagenet_full':
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
        transform_test = transforms.Compose([
            transforms.CenterCrop(224),  # TODO
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values)
        ])
        testdir = os.path.join(args.dir, 'ILSVRC12_256', 'val')

num_classes = args.num_class
num_task = args.num_task
num_class_per_task = (num_classes - args.nb_cl_fg) // num_task

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random_perm = list(range(num_classes))

print('Test starting -->\t')
acc_all = np.zeros((num_task+3, num_task+1), dtype = 'float') # Save for csv

for task_id in range(num_task+1):
    if task_id == 0:
        index = random_perm[:args.nb_cl_fg]
    else:
        index = random_perm[:args.nb_cl_fg + (task_id) * num_class_per_task]
    if 'imagenet' in args.data:
        testfolder = ImageFolder(testdir, transform_test, index=index)
        test_loader = torch.utils.data.DataLoader(testfolder, batch_size=128, shuffle=False, num_workers=4, drop_last=False)
    elif args.data =='cifar100':
        np.random.seed(args.seed)
        target_transform = np.random.permutation(num_classes)
        testset = CIFAR100(root=testdir, train=False, download=True, transform=transform_test, target_transform = target_transform, index = index)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2,drop_last=False)

    print('Test %d\t' % task_id)

    model_id = task_id
    model = torch.load(models[model_id])


    val_embeddings_cl, val_labels_cl = extract_features_classification_aug(model, test_loader, print_freq=32, metric=None)
    # Unknown task ID

    num_class = 0
    ave = 0.0
    weighted_ave = 0.0
    for k in range(task_id + 1):
        if k==0:
            tmp = random_perm[:args.nb_cl_fg]
        else:
            tmp = random_perm[args.nb_cl_fg + (k-1) * num_class_per_task:args.nb_cl_fg + (k) * num_class_per_task]
        gt = np.isin(val_labels_cl, tmp)
        if args.top5:
            estimate = np.argsort(val_embeddings_cl, axis=1)[:,-5:]
            estimate_label = estimate
            estimate_tmp = np.asarray(estimate_label)[gt]
            labels_tmp = np.tile(val_labels_cl[gt].reshape([len(val_labels_cl[gt]),1]),[1,5])
            acc = np.sum(estimate_tmp == labels_tmp) / float(len(estimate_tmp))
        else:
            estimate = np.argmax(val_embeddings_cl, axis=1)
            estimate_label = estimate
            estimate_tmp = np.asarray(estimate_label)[gt]
            acc = np.sum(estimate_tmp == val_labels_cl[gt]) / float(len(estimate_tmp))
        ave += acc
        weighted_ave += acc * len(tmp)
        num_class += len(tmp)
        print("Accuracy of Model %d on Task %d with unknown task boundary is %.3f" % (model_id, k, acc))
        acc_all[k, task_id] = acc
    print('Average: %.3f      Weighted Average: %.3f' %(ave / (task_id + 1), weighted_ave / num_class))
    acc_all[num_task + 1, task_id] = ave / (task_id + 1)
    acc_all[num_task + 2, task_id] = weighted_ave / num_class

np.savetxt(args.log_dir + '/epoch_%d.csv' % args.epochs, acc_all*100, delimiter=',')