from torch.utils import data
from os.path import join, split #, abspath, splitext, split, isdir, isfile
from PIL import Image
import numpy as np
import cv2


def prepare_image_cv2(im):
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im

class BSDSLoader(data.Dataset):
    """
    Dataloader HED-BSDS500
    """
    def __init__(self, root=join('..', 'HED-BSDS'), dataSplit='train', transform=False):
        self.root = root
        self.dataSplit = dataSplit
        self.transform = transform
        if self.dataSplit == 'train':
            self.filelist = join(self.root, 'train_pair.lst')
        elif self.dataSplit == 'test':
            self.filelist = join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid data split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        if self.dataSplit == "train":
            img_file, lb_file = self.filelist[index].split()
            lb = np.array(Image.open(join(self.root, lb_file)), dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            lb = lb[np.newaxis, :, :]
            lb[lb < 128] = 0.0
            lb[lb >= 128] = 1.0
        else:
            img_file = self.filelist[index].rstrip()
        img = np.array(cv2.imread(join(self.root, img_file)), dtype=np.float32)
        img = prepare_image_cv2(img)
        if self.dataSplit == "train":
            return img, lb
        else:
            return img

