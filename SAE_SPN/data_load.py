import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from torch.utils import data
from PIL import Image
import glob
class CLIC_Dataset(data.Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, root, data_folder, mirror, transform = None, crop_size=(321, 321)):
        self.root = root
        self.data_folder = data_folder
        self.transform = transform
        #self.img_list = sorted(os.listdir(osp.join(self.root, self.data_folder)))
        self.img_list = glob.glob(osp.join(self.root, self.data_folder) + '/*.png')
        
        self.img_path = [self.img_list[i] for i in range(0,len(self.img_list))]
        #self.img_path = [self.root + self.data_folder + self.img_list[i] for i in range(0,len(self.img_list))]
        self.crop_h, self.crop_w = crop_size
        self.is_mirror = mirror

    def __len__(self):
        return len(self.img_list)

    def hist_equal(self, img): #Return bgr if given bgr, returns rgb if given rgb
        gridsize = 8
        #img = cv2.imread("sample.png")[...,::-1]
        #plt.figure(3)
        #plt.imshow(img)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        #plt.figure(4)
        #plt.imshow(bgr)
        #plt.show()
        return bgr

    def __getitem__(self, index):

        img_file = self.img_path[index]
        #print(img_file)
        image = self.hist_equal(cv2.imread(img_file, cv2.IMREAD_COLOR))
        size = image.shape        
        image = np.asarray(image, np.float32)/255
        image = 2*image - 1 
        
        if self.transform is not None:
            image = self.transform(image)

        img_h, img_w, _ = image.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0))
        else:
            pass

        image = np.asarray(cv2.resize(image, dsize = (self.crop_h,self.crop_w), interpolation = cv2.INTER_NEAREST), np.float32)

        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]

        return image.copy()
