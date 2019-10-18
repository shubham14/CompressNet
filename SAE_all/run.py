import os, time, random, argparse
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
from torch.utils import data
from PIL import Image
from torch.optim.lr_scheduler import MultiStepLR
# from model_utils import Vgg16
#File imports
from data_load import *
from config import config_train, directories
from model import *

save_pred_every = 1
snapshot_dir = './Video_Checkpoints/'


def train(config, train_loader, net, optimizer_gen, optimizer_dis, device,scheduler):
	start_time = time.time()
	G_loss_best, D_loss_best = float('inf'), float('inf')
	
	G_loss_history = []
	D_loss_history = []
	R_loss_history = []
	P_loss_history = []
	running_loss_history = []
	for epoch in range(config.num_epochs):  # loop over the dataset multiple times
		scheduler.step()
		for param_group in optimizer_gen.param_groups:
			print("Generator Learning Rate = ",param_group['lr'])
		for param_group in optimizer_dis.param_groups:
			print("Discriminator Learning Rate = ",param_group['lr'])
		start = time.time()
		running_loss = 0.0
		for i, imgs in enumerate(train_loader):
			imgs = imgs.to(device)
			net = net.to(device)

			for param in net.discriminator.parameters():
				param.requires_grad = True

			# Update Discriminator
			
			optimizer_dis.zero_grad()
			net(imgs,evaluate=False)
			_, D_loss_real, D_loss_gen, _, _, gp_loss, _= net.loss()
			
			# seperate class training

			# real class
			D_loss_real.backward()

			# fake class
			D_loss_gen.backward()

			# gradient penalty 
			gp_loss.backward()

			D_loss = D_loss_real + D_loss_gen + gp_loss
			optimizer_dis.step()

			# # Update Generator

			# freeze discriminator parameters
			for param in net.discriminator.parameters():
				param.requires_grad = False
				
			for _ in range(5):
				optimizer_gen.zero_grad()
				net(imgs,evaluate=False)
				G_loss, _, _, R_loss, P_loss, _, _ = net.loss()
				G_loss.backward()
				optimizer_gen.step()


			G_loss_history.append(G_loss.item())
			D_loss_history.append(D_loss.item())
			R_loss_history.append(R_loss.item())
			P_loss_history.append(R_loss.item())

			running_loss += G_loss.item() + D_loss.item()
			#running_loss += G_loss.item()

			if True:#i % 100 == 99:    # print every 2000 mini-batches
				running_loss_history.append(running_loss)

			
			if True:    # print every 2000 mini-batches
				end = time.time()
				# print('[epoch %d, iter %5d] D_loss: %.3f, G_loss: %.3f,R_loss: %.3f ,P_loss: %.3f,style_loss: %.3f,loss: %.3f eplased time %.3f' %
				#       (epoch + 1, i + 1, D_loss.item(), G_loss.item(), R_loss.item(),P_loss.item(),style_loss.item(), running_loss / 100, end-start))
				print('[epoch %d, iter %5d] D_loss: %.3f, G_loss: %.3f,R_loss: %.3f ,P_loss: %.3f,loss: %.3f eplased time %.3f' %
					  (epoch + 1, i + 1, D_loss.item(), G_loss.item(), R_loss.item(),P_loss.item(),running_loss / 100, end-start))

			   
				start = time.time()
				running_loss = 0.0

			if epoch % 30 == 0:
				np.save('running_loss_history_CNet_BN_ICNR_{}.npz'.format(epoch), np.array(running_loss_history))
				np.save('G_loss_history_CNet_BN_ICNR_{}.npz'.format(epoch), np.array(G_loss_history))
				np.save('D_loss_history_CNet_BN_ICNR_{}.npz'.format(epoch), np.array(D_loss_history))
				np.save('R_loss_history_CNet_BN_ICNR_{}.npz'.format(epoch), np.array(R_loss_history))


		if epoch % save_pred_every == 0 and epoch!=0:
			print ('taking snapshot of the model...')
			torch.save(net.state_dict(),osp.join(snapshot_dir, 'Checkpoint_CNet_BN_ICNR_'+str(epoch)+'.pth'))

		print('Finished Training')
	 
def main():

	device = config_train.device
	#device = 'cpu'
	print('Running on device : {}'.format(device))
	print(torch.cuda.current_device())

	start = time.time()

	train_loader = torch.utils.data.DataLoader(
		CLIC_Dataset(directories.root, directories.train, config_train.mirror, crop_size = (512,512)),
		batch_size=config_train.batch_size, shuffle=True, num_workers=config_train.workers, pin_memory=True,drop_last=True)

	# train_loader = torch.utils.data.DataLoader(
	# 	datasets.MNIST('../data', train=True, download=True,
	# 					transform=transforms.Compose([
	# 						transforms.ToTensor(),
	# 						transforms.Normalize((0.1307,), (0.3081,))
	# 					])),
	# 	batch_size=128, shuffle=True)

	# val_loader = torch.utils.data.DataLoader(
	#     CLIC_Dataset(directories.root, directories.val, config_train.mirror, crop_size = (50,50)),
	#     batch_size=config_train.batch_size, shuffle=True, num_workers=config_train.workers, pin_memory=True)

	end = time.time()

	print("Time to create the training dataloaders = " ,(end - start))

	# Build network
	# CompressNet = Model(config_train)
	# print(CompressNet)
	# saved_state_dict = torch.load('TF_checkpoint.pth')
	# mod_saved_state_dict = saved_state_dict
	# for key,value in saved_state_dict.items():
	#     if 'dcgan_generator' in key:
	#         mod_saved_state_dict.pop(key)
	#     print(key)

	# model_dict = CompressNet.state_dict()
	# # 1. filter out unnecessary keys
	# pretrained_dict = {k: v for k, v in saved_state_dict.items() if 'dcgan_generator' not in k }
	# # 2. overwrite entries in the existing state dict
	# model_dict.update(pretrained_dict) 
	# # 3. load the new state dict
	# CompressNet.load_state_dict(model_dict)

	# CompressNet.load_state_dict(saved_state_dict)

	CompressNet = Model(config_train)
	saved_state_dict = torch.load('./Checkpoints/Checkpoint_CNet_BN_ICNR_46.pth')
	# model_dict = CompressNet.state_dict()
	# pretrained_dict = {k: v for k, v in saved_state_dict.items() if 'dcgan_generator' not in k }
	# pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'decoder' not in k }
	# model_dict.update(pretrained_dict) 
	CompressNet.load_state_dict(saved_state_dict)
	params_gen = list(CompressNet.encoder.parameters()) + list(CompressNet.decoder.parameters()) #+ list(CompressNet.dcgan_generator.parameters())
	optimizer_gen = optim.Adam(params_gen, lr=config_train.G_learning_rate, betas=(0.5, 0.999), eps=1e-08,weight_decay = config_train.weight_decay)
	# optimizer_dis = optim.Adam(CompressNet.discriminator.parameters(), lr=config_train.D_learning_rate, betas=(0.5, 0.999), eps=1e-08)
	optimizer_dis = optim.SGD(CompressNet.discriminator.parameters(), lr=config_train.D_learning_rate, momentum=0.9)
	milestones=[i for i in range(5,config_train.num_epochs,5)]
	# print(milestones)
	scheduler = MultiStepLR(optimizer_gen, milestones = [5,15,25,35] , gamma=0.5)

	train(config_train, train_loader, CompressNet, optimizer_gen, optimizer_dis, device,scheduler)

if __name__ == '__main__':
	main()
