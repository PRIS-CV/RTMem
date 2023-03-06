# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta
import os
from sklearn.cluster import DBSCAN
# from tensorboardX import SummaryWriter
from sklearn.cluster import KMeans
import faiss

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from RTMem import datasets
from RTMem import models
from RTMem.models.cm import ClusterMemory, HybridMemory
from RTMem.trainers import ClusterContrastTrainer
from RTMem.evaluators import Evaluator, extract_features
from RTMem.utils.data import IterLoader
from RTMem.utils.data import transforms as T
from RTMem.utils.data.sampler import RandomMultipleGallerySampler
from RTMem.utils.data.preprocessor import Preprocessor
from RTMem.utils.logging import Logger
from RTMem.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from RTMem.utils.faiss_rerank import compute_jaccard_distance


start_epoch = best_mAP = 0

# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)

    # initial_weights = load_checkpoint('/data1/yinjunhui/data/per-id/logs/market1501TOdukemtmc/resnet50-pretrain-1/model_best.pth.tar')
    # initial_weights = load_checkpoint('./pretrain/duke.pth.tar')
    
    # copy_state_dict(initial_weights['state_dict'], model)

    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # if args.enable_tb:
    # from tensorboardX import SummaryWriter
    # import pdb 
    # pdb.set_trace()
    # print(torch.cuda.is_available())
    # a=torch.Tensor([1,2])
    # a=a.cuda()
    # a

    # log_path = osp.join(args.save_path, args.exp_name)
    # # tensorboard
    # if args.enable_tb: 
    vis_log_dir = osp.join('./', 'vis_log')
    if not os.path.isdir(vis_log_dir):
        os.makedirs(vis_log_dir)
    # writer = SummaryWriter(vis_log_dir)

    main_worker(args)

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


def main_worker(args):
    global start_epoch, best_mAP
    # start_time = time.monotonic()

    cudnn.benchmark = True

    vis_log_dir = osp.join('./', 'vis_log')
    if not os.path.isdir(vis_log_dir):
        os.makedirs(vis_log_dir)
    # writer = SummaryWriter(vis_log_dir)


    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)


    # from torchvision.models import resnet50
    # from thop import profile
    # model = create_model(args)
    # dummy_input = torch.randn(1, 3, 256, 128).cuda()
    # macs, params = profile(model, inputs=(dummy_input, ))

    # #输出
    # from thop import clever_format
    # macs, params = clever_format([macs, params], "%.3f")
    
    # Create model
    model = create_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer

    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = ClusterContrastTrainer(model)

    memory_instance = HybridMemory(model.module.num_features, len(dataset.train),
                            temp=args.temp, momentum=args.momentum).cuda()
    #################################
    print("==> Initialize instance features in the hybrid memory")
    cluster_loader = get_test_loader(dataset, args.height, args.width,
                                    args.batch_size, args.workers, testset=sorted(dataset.train))
    features, _ = extract_features(model, cluster_loader, print_freq=50)
    features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
    memory_instance.features = F.normalize(features, dim=1).cuda()
    del features
    #########################################
    for epoch in range(args.epochs):

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])
            centers = [
                torch.stack(centers[idx], dim=0)[np.random.choice(len(centers[idx]))] for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers


        # # generate new dataset and calculate cluster centers
        def generate_pseudo_labels(cluster_id, num):
            labels = []
            outliers = 0
            for i, ((fname, _, cid), id) in enumerate(zip(sorted(dataset.train), cluster_id)):
                if id!=-1:
                    labels.append(id)
                else:
                    labels.append(num+outliers)
                    outliers += 1
            return torch.Tensor(labels).long()

        ##############################
        # del features, pseudo_labels, rerank_dist
        with torch.no_grad():
            features, _ = extract_features(model, cluster_loader, print_freq=50)
            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)

            if epoch == 0:
                # DBSCAN cluster
                eps = args.eps
                print('Clustering criterion: eps: {:.3f}'.format(eps))
                cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

            # select & cluster images as training set of this epochs
            pseudo_labels = cluster.fit_predict(rerank_dist)
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
            cluster_features = generate_cluster_features(pseudo_labels, features)

        # ## k-means ##################
        # features = memory_instance.features
        # num_cluster = 1000
        # cluster = faiss.Kmeans(
        #     features.size(-1), num_cluster, niter=300, verbose=True, gpu=True
        # )

        # cluster.train(to_numpy(features))
        # centers = to_torch(cluster.centroids).float()
        # _, pseudo_labels = cluster.index.search(to_numpy(features), 1)
        # pseudo_labels = to_torch(pseudo_labels)
        # cluster_features = generate_cluster_features(pseudo_labels, features)
        # ## k-means ##################

        # Create hybrid memory
        memory = ClusterMemory(model.module.num_features, num_cluster, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()

        memory.features = F.normalize(cluster_features, dim=1).cuda() # [196, 2048]

        pseudo_labeled_dataset_label = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset_label.append((fname, label.item(), cid))

        # print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

        train_loader_label = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_label)

        train_loader_label.new_epoch()

        #####################################
        pseudo_labels = generate_pseudo_labels(pseudo_labels, num_cluster)

        memory_instance.labels = pseudo_labels.cuda()  # [12936]
        # import pdb
        # pdb.set_trace()
        trainer.memory = memory
        trainer.memory_instance = memory_instance  # [12936, 2048]]

        print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

        trainer.train(epoch, torch.tensor(num_cluster).cuda(), train_loader_label, train_loader_label, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_label))

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            _, mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()
    start_time = time.monotonic()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    for i in range(1):
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,     # default=0.00035
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)     # default=20
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                         default= "data")
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join('./', 'logs'))
    parser.add_argument('--pooling-type', type=str, default='gem')  # gem
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--enable_tb', action='store_true', help='enable tensorboard logging')

    main()
