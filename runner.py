import torch
import os
import argparse
import train_hed
# Argument parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--mode', default='test', help='Set to either train or test, adding evaluate soon.')

args = parser.parse_args()

# # # # # # # # # # # # # # 
if __name__ == '__main__':
    # if train just use os to call the train script
    if(args.mode == 'train'):
        print(torch.cuda.current_device())
        os.system('python train_hed.py')
    
    # testing
    # - run the testing for this guy
    if(args.mode == 'test'):
        print("Current GPU device id:  %d" % torch.cuda.current_device())
        train_hed.run_single_test(restore_path='tmp/HED/checkpoint_epoch9.pth')
