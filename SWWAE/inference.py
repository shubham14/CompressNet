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
import json
import pickle as pkl
import h5py

# def get_local(ind, img_size):
#     N,C,H,W = img_size
#     class_mask = torch.zeros(1,C,H*2,W*2)
#     for chan in range(len(ind)):
#        image_ind = ind[chan]
#        # index = 2*((image_ind / img_size)%2) + (image_ind % img_size)%2     # Using Row and Column indices to map it to {0,1,2,3}
#        row = image_ind/(H*2)
#        col = image_ind%(H*2)
#        class_mask[0,chan,row,col] = 1
#     return class_mask

def get_local(ind, img_size):
    N,C,H,W = img_size
    out_ind = torch.zeros_like(ind)
    for chan in range(len(ind)):
        image_ind = ind[chan]
        index = 2*((image_ind / H)%2) + (image_ind % W)%2     # Using Row and Column indices to map it to {0,1,2,3}
        out_ind[chan] = index
    return out_ind

def inference(checkpoint_enc_path, data_loader, net, device='cpu'):
    net.cuda(device)
    
    saved_state_dict = torch.load(checkpoint_enc_path)
    # for key,value in saved_state_dict.items():
    #     print(key,value.size())
    # print(saved_state_dict.items())
    net.load_state_dict(saved_state_dict)
    
    #net.eval()
    net.cuda(device)
    #net.eval()8
    global_count = 0
    for img in data_loader:
        size = img.size()
        break

    N,C,H,W = size
    print(size)
    layers = torch.zeros([60,256,256])
    C_mask = torch.zeros([60,256,256])
    # new_layer = torch.zeros([1,60,256,256])
    # new_C_mask = torch.zeros([1,60,256,256])
    for count,img in enumerate(data_loader):
        label = 0
        I_label = 1
        img = img.to(device)
        out = net(img,evaluate=True)
        index = net.indices[0]
        for i in range(0,img.size(0)):
            print('Working on image {}'.format(global_count))
            # class_mask = get_local(index[0], index.size())
            # # data[global_count] = [net.rec_intermediate_layers[3][i,:,:,:], class_mask]
            # print(net.rec_intermediate_layers[0][i,:,:,:].size())
            # hf = h5py.File('./SWWAE_dataset/'+str(global_count)+'.h5', 'w')
            # hf.create_dataset('Feature', data=net.rec_intermediate_layers[0][i,:,:,:].cpu().detach().numpy())
            # hf.create_dataset('label', data=class_mask.cpu().detach().numpy())
            # hf.close()
                # C_mask[0,:,:,:] = class_mask

            # else:
            #     new_layer[0,:,:,:] = net.rec_intermediate_layers[0][i,:,:,:]
            #     layers = torch.cat((layers,new_layer),dim = 0)
            #     new_C_mask[0,:,:,:] = class_mask
            #     C_mask = torch.cat((new_C_mask,C_mask),dim = 0)
                
            # show_img(img[i,:,:,:],global_count,I_label)
            show_img(out[i,:,:,:],global_count,label)
            global_count += 1

    # test_convert = torch.utils.data.TensorDataset(layers, C_mask)
    # test_loader = torch.utils.data.DataLoader(test_convert, batch_size=1, shuffle=False)
    # torch.save(test_loader, 'SWWAE_data.pkl')
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
    inference('./Checkpoints/Checkpoint_CNet_BN_ICNR_45.pth', train_loader, net = CompressNet,device=device)

if __name__ == "__main__":
    main()

