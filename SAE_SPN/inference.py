from network import  Generator, Discriminator, Encoder, Decoder, Quantizer
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import *
from torch.autograd import Variable
from data_load import *
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import time
import numpy
import torchvision.transforms as transforms

import torchvision.datasets as datasets

#File imports
from data_load import *
from config import config_train, directories
from model import *

def inference(checkpoint_enc_path, data_loader, net, device='cpu'):
    net.cuda(device)
    
    saved_state_dict = torch.load(checkpoint_enc_path)
    # for key,value in saved_state_dict.items():
    #     print(key,value.size())
    # print(saved_state_dict.items())
    saved_dict = torch.load('index_pred1.pth')
    saved_dict = {'decoder.ind_pred.'+k: v for k, v in saved_dict.state_dict().items() if 'conv1' in k }
    saved_state_dict1 = {k: v for k, v in saved_state_dict.items() if 'decoder.ind_pred' not in k }
    saved_state_dict1.update(saved_dict)
    net.load_state_dict(saved_state_dict1)
    
    #net.eval()
    net.cuda(device)
    #net.eval()
    global_count = 0
    for count,img in enumerate(data_loader):
        # print(img[1])
        # label = img[1]
        label = 0
        I_label = 1
        img = img.to(device)
        out = net(img,evaluate=True)
        print(out.size())
        for i in range(0,img.size(0)):
            show_img(img[i,:,:,:],global_count,I_label)
            show_img(out[i,:,:,:],global_count,label)
            global_count += 1
    return out

def to_img(x):
    x = (0.5 * (x + 1)) 
    # x = x.clamp(0, 1)
    #print(x)
    x = x.view(3, 512, 512)
    return x

def show_img(out,count,label):
    pic = to_img(out.detach().cpu().data)
    save_image(pic, './Results/image_{}_{}.png'.format(count,label))
    # img = out.detach().cpu().numpy()
    # img = np.reshape(img,(3,256,256))
    # img = np.swapaxes(img,1,0)
    # img = np.swapaxes(img,2,1)
    # img = img[:, :, ::-1]
    # print(img.shape)

    # cv2.imshow('Reconstructed image', img)
    # cv2.waitKey()

def main():
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device_id = 
    # torch.cuda.set_device(device_id)
    #device = 'cpu'
    print(torch.cuda.current_device())
    print('Running on device : {}'.format(device))

    start = time.time()
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=1, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        CLIC_Dataset(directories.root, directories.train, config_train.mirror, crop_size = (512,512)),
        batch_size=2, shuffle=True, num_workers=config_train.workers, pin_memory=True,drop_last=True)
    end = time.time()

    print("Time to create the training dataloaders = " ,(end - start))

    # Build network
    CompressNet = Model(config_train)
    # print(CompressNet)
    #CompressNet.load_state_dict(torch.load('./Checkpoints/Checkpoint_CompressNet_90.pth'))
    #CompressNet.eval()
    # inference('./Checkpoints/Checkpoint_CNet_CompressNet_withGD_noise_95.pth', train_loader, net = CompressNet,device=device)
    # Noise = False, Channel Bottleneck = 16, GAN, C = 8
    inference('./PreCheckpoints/Checkpoint_CNet_BN_ICNR_99.pth', train_loader, net = CompressNet,device=device)

if __name__ == "__main__":
    main()

