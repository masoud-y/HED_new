#!/user/bin/python
# coding=utf-8
import os, sys
import numpy as np
from PIL import Image
import time
import torch


#import torch.nn as nn
#import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision
#import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
from torch.utils.data import DataLoader #, sampler
from os.path import join, split, isdir, isfile, splitext,  abspath, dirname

from data_loader import BSDSLoader
from models import HED, convert_vgg, weights_init
from functions import   cross_entropy_loss # sigmoid_cross_entropy_loss
from utils import Logger, Averagvalue, save_checkpoint, load_vgg16pretrain, arg_parser, tune_lrs

from scipy.io import savemat

root='../..'
args= arg_parser(dataset=join(root,'HED-BSDS'), batch_size=1, lr=1e-6, momentum=0.9, weight_decay=2e-4,stepsize=3,
                 gamma=0.1, start_epoch=0, maxepoch=10, itersize=10, print_freq=50,
                 gpu='1',resume='', tmp=join('tmp','HED'))


args.model_path=join(root,'pretrained_models', 'vgg16.pth')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

THIS_DIR = abspath(dirname(__file__))
TMP_DIR = join(THIS_DIR, args.tmp)
if not isdir(TMP_DIR):
  os.makedirs(TMP_DIR)

def main():
    args.cuda = True
    # dataset
    train_dataset = BSDSLoader(root=args.dataset, dataSplit="train")
    test_dataset = BSDSLoader(root=args.dataset, dataSplit="test")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=8, drop_last=True,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=8, drop_last=True,shuffle=False)
    with open(join(args.dataset,'test.lst'), 'r') as f:
        test_list = f.readlines()
    test_list = [split(i.rstrip())[1] for i in test_list]
    assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))

    # model
    model = HED()
    
    model.apply(weights_init)
    
    pretrained_dict = torch.load(args.model_path)
    pretrained_dict = convert_vgg(pretrained_dict)
    
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    
    model.cuda()
    
    
    if args.resume:
        if isfile(args.resume): 
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    if args.resume:
        if isfile(args.resume): 
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    #tune lr
    tuned_lrs=tune_lrs(model,args.lr, args.weight_decay)


    optimizer = torch.optim.SGD(tuned_lrs, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    # log
    log = Logger(join(TMP_DIR, '%s-%d-log.txt' %('Adam',args.lr)))
    sys.stdout = log

    train_loss = []
    train_loss_detail = []
    for epoch in range(args.start_epoch, args.maxepoch):
        if epoch == 0:
            print("Performing initial testing...")
            validate(model, test_loader, epoch=epoch, test_list=test_list,
                 save_dir = join(TMP_DIR, 'initial-testing-record'))

        tr_avg_loss, tr_detail_loss = train(train_loader, model, optimizer, epoch, 
                                            save_dir = join(TMP_DIR, 'epoch-%d-training-record' % epoch))
        validate(model, test_loader, epoch=epoch, test_list=test_list,
            save_dir = join(TMP_DIR, 'epoch-%d-testing-record' % epoch))
        log.flush() # write log
        # Save checkpoint
        save_file = join(TMP_DIR, 'checkpoint_epoch{}.pth'.format(epoch))
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
                         }, filename=save_file)
        scheduler.step() # will adjust learning rate
        # save train/val loss/accuracy, save every epoch in case of early stop
        train_loss.append(tr_avg_loss)
        train_loss_detail += tr_detail_loss

def train(train_loader, model, optimizer, epoch, save_dir):
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    # switch to train mode
    model.train()
    end = time.time()
    epoch_loss = []
    counter = 0
    for i, (image, label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        image, label = image.cuda(), label.cuda()
        outputs = model(image)
        loss = torch.zeros(1).cuda()
        for o in outputs:
            loss = loss + cross_entropy_loss(o, label)
        counter += 1
        loss = loss / args.itersize
        loss.backward()
        if counter == args.itersize:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0
        # measure accuracy and record loss
        losses.update(loss.item(), image.size(0))
        epoch_loss.append(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        # display and logging
        if not isdir(save_dir):
            os.makedirs(save_dir)
        if i % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, i, len(train_loader)) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(
                       loss=losses)
            print(info)
            outputs.append(label)
            _, _, H, W = outputs[0].shape
            all_results = torch.zeros((len(outputs), 1, H, W))
            for j in range(len(outputs)):
                all_results[j, 0, :, :] = outputs[j][0, 0, :, :]
            torchvision.utils.save_image(all_results, join(save_dir, "iter-%d.jpg" % i))
        # save checkpoint
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
            }, filename=join(save_dir, "epoch-%d-checkpoint.pth" % epoch))

    return losses.avg, epoch_loss

def validate(model, test_loader, epoch, test_list, save_dir):
    model.eval()
    if not isdir(save_dir):
        os.makedirs(save_dir)
    for idx, image in enumerate(test_loader):
        image = image.cuda()
        # rescale image to [0, 255] and then substract the mean
        # https://github.com/pytorch/vision/blob/c74b79c83fc99d0b163d8381f7aa1296e4cb23d0/torchvision/transforms/functional.py#L51
        _, _, H, W = image.shape
        results = model(image)
        result = torch.squeeze(results[-1].detach()).cpu().numpy()
        results_all = torch.zeros((len(results), 1, H, W))
        for i in range(len(results)):
          results_all[i, 0, :, :] = results[i]
        filename = splitext(test_list[idx])[0]
        torchvision.utils.save_image(results_all, join(save_dir, "%s.jpg" % filename))
        result_b = Image.fromarray(((1- result) * 255).astype(np.uint8))
        result = Image.fromarray((result * 255).astype(np.uint8))
        result.save(join(save_dir, "%s.png" % filename))
        result_b.save(join(save_dir, "%s.jpg" % filename))
        print("Running validation [%d/%d]" % (idx + 1, len(test_loader)))
        
def test(restore_path, save_dir='test/', dataset=args.dataset):
    # model
    model = HED()
    #model = nn.DataParallel(model)
    model.cuda()
    print("=> loading checkpoint '{}'".format(restore_path))
    checkpoint = torch.load(restore_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}'".format(restore_path))
    # data
    test_dataset = BSDSLoader(root=dataset, dataSplit="test")
    test_loader = DataLoader(test_dataset, batch_size=1, 
                             num_workers=8, drop_last=True,shuffle=False)
    with open(join(dataset,'test.lst'), 'r') as f:
        test_list = f.readlines()
    test_list = [split(i.rstrip())[1] for i in test_list]
    assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))
    
    # Now run the testing
    model.eval()
    # setup the save directories
    dirs = ['side_1/', 'side_2/', 'side_3/', 'side_4/', 'side_5/', 'fuse/', 'merge/', 'jpg_out/']
    for idx, dir_path in enumerate(dirs):
        os.makedirs(save_dir + dir_path, exist_ok=True)
        if(idx < 6): os.makedirs(save_dir+ 'mat/' + dir_path, exist_ok=True)
    # run training
    for idx, image in enumerate(test_loader):
        print("\rRunning test [%d/%d]" % (idx + 1, len(test_loader)), end='')
        image = image.cuda()
        _, _, H, W = image.shape
        results = model(image)
        results_all = torch.zeros((len(results), 1, H, W))
        # make our result array
        for i in range(len(results)):
          results_all[i, 0, :, :] = results[i]
        filename = splitext(test_list[idx])[0]
        torchvision.utils.save_image(results_all, join(save_dir, '{}{}.jpg'.format(dirs[-1],filename)))
        
        # now go through and save all the results
        for i, r in enumerate(results): 
            
            img= torch.squeeze(r.detach()).cpu().numpy()
            savemat(join(save_dir,'mat',dirs[i],'{}.mat'.format(filename)), {'img': img})
            img = Image.fromarray((img * 255).astype(np.uint8))
            
            img.save('{}{}.png'.format(save_dir+dirs[i], filename))
            
        merge = sum(results) / 5
        torchvision.utils.save_image(torch.squeeze(merge), '{}{}.png'.format(save_dir+'merge/', filename))
        torchvision.transforms.transforms.ToPILImage(torch.squeeze(results[i]))
    print('')


if __name__ == '__main__':
    main()
