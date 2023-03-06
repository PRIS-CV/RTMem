from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch
from torch.autograd import Variable
# from tensorboardX import SummaryWriter
import os.path as osp
import os

class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None, memory_instance=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.memory_instance = memory_instance

    def train(self, epoch, num_cluster, data_loader, data_loader_label, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        loss_class_meter = AverageMeter()
        loss_instance_meter = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)
            # import pdb 
            # pdb.set_trace()
            # process inputs
            # inputs, labels, indexes = self._parse_data(inputs)

            inputs_label = data_loader_label.next()
            inputs_label, labels_label, indexes = self._parse_data(inputs_label)

            f_out_label = self._forward(inputs_label)
            
            # compute loss with the hybrid memory
            if epoch < 5:
                momentum = 0.1
            else:
                momentum = 0.1
            loss_class = self.memory(f_out_label, labels_label, momentum)
            loss_instance = self.memory_instance(f_out_label, indexes, momentum)
            loss_instance = loss_class
            loss = loss_class + (1.2) * loss_instance
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            loss_class_meter.update(loss_class.item())
            loss_instance_meter.update(loss_instance.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'loss {:.3f} ({:.3f})\t'
                      'loss_class {:.3f} ({:.3f})\t'
                      'loss_instance {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              loss_class_meter.val, loss_class_meter.avg,
                              loss_instance_meter.val, loss_instance_meter.avg))



    # if args.enable_tb:
        # writer.add_scalar('total_loss', losses.avg, epoch)
        # writer.add_scalar('class_loss', loss_class_meter.avg, epoch)
        # writer.add_scalar('instance_loss', loss_instance_meter.avg, epoch)
        # if args.method == 'full':
        #     writer.add_scalar('ic_loss', loss_ic_meter.avg, epoch)
        #     writer.add_scalar('dt_loss', loss_dt_meter.avg, epoch)
    # writer.close()

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

