import os, sys
import torch
import numpy as np
import scipy.io as sio
import argparse
from os.path import join, isfile #split, abspath, splitext, split, isdir, isfile


class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_pretrained(model, fname, optimizer=None):
    """
    resume training from previous checkpoint
    :param fname: filename(with path) of checkpoint file
    :return: model, optimizer, checkpoint epoch
    """
    if isfile(fname):
        print("=> loading checkpoint '{}'".format(fname))
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            return model, optimizer, checkpoint['epoch']
        else:
            return model, checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(fname))

###########################################
################## logger
        
class Logger(object):
  def __init__(self, fpath=None):
    self.console = sys.stdout
    self.file = None
    if fpath is not None:
      self.file = open(fpath, 'w')

  def __del__(self):
    self.close()

  def __enter__(self):
    pass

  def __exit__(self, *args):
    self.close()

  def write(self, msg):
    self.console.write(msg)
    if self.file is not None:
        self.file.write(msg)

  def flush(self):
    self.console.flush()
    if self.file is not None:
        self.file.flush()
        os.fsync(self.file.fileno())

  def close(self):
    self.console.close()
    if self.file is not None:
        self.file.close()


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

####################################################################
        ########## learning rate tuning
            
def arg_parser(dataset='../HED-BSDS', batch_size=1, lr=1e-6, momentum=0.9, 
               weight_decay=2e-4,stepsize=3, gamma=0.1, start_epoch=0, 
               maxepoch=10, itersize=10, print_freq=50, gpu='0',resume='', 
               tmp='tmp/HED'):
    
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--batch_size', default=batch_size, type=int, metavar='BT',
                        help='batch size')
    # =============== optimizer
    parser.add_argument('--lr', '--learning_rate', default=lr, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=momentum, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=weight_decay, type=float,
                        metavar='W', help='default weight decay')
    parser.add_argument('--stepsize', default=stepsize, type=int, 
                        metavar='SS', help='learning rate step size')
    parser.add_argument('--gamma', '--gm', default=gamma, type=float,
                        help='learning rate decay parameter: Gamma')
    parser.add_argument('--maxepoch', default=maxepoch, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--itersize', default=itersize, type=int,
                        metavar='IS', help='iter size')
    # =============== misc
    parser.add_argument('--start_epoch', default=start_epoch, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--print_freq', '-p', default=print_freq, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--gpu', default=gpu, type=str,
                        help='GPU ID')
    parser.add_argument('--resume', default=resume, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--tmp', help='tmp folder', default=tmp)
    # ================ dataset
    parser.add_argument('--dataset', help='root folder of dataset', default=dataset)
    
    return parser.parse_args()


def tune_lrs(net, lr, weight_decay):
    
    bias_params= [param for name,param in list(net.named_parameters()) if name.find('bias')!=-1]
    weight_params= [param for name,param in list(net.named_parameters()) if name.find('weight')!=-1]

    conv1_4_weights , conv1_4_bias  = weight_params[0:10]  , bias_params[0:10]
    conv5_weights   , conv5_bias    = weight_params[10:13] , bias_params[10:13]
    d1_5_weights    , d1_5_bias     = weight_params[13:18] , bias_params[13:18]
    fuse_weights , fuse_bias =weight_params[-1] , bias_params[-1]
    
    tuned_lrs=[
            {'params': conv1_4_weights, 'lr': lr*1    , 'weight_decay': weight_decay},
            {'params': conv1_4_bias,    'lr': lr*2    , 'weight_decay': 0.},
            {'params': conv5_weights,   'lr': lr*100  , 'weight_decay': weight_decay},
            {'params': conv5_bias,      'lr': lr*200  , 'weight_decay': 0.},
            {'params': d1_5_weights,    'lr': lr*0.01 , 'weight_decay': weight_decay},
            {'params': d1_5_bias,       'lr': lr*0.02 , 'weight_decay': 0.},
            {'params': fuse_weights,    'lr': lr*0.001, 'weight_decay': weight_decay},
            {'params': fuse_bias ,      'lr': lr*0.002, 'weight_decay': 0.},
            ]
    return  tuned_lrs