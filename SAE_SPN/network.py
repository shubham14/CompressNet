'''
Contains the architecture for Generative Compression using GANs
'''

import torchvision.models as models
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
from model_utils import *
import torch.nn.init as init


# Xavier Initialization for layer weights
def init_weights(m):
	if type(m) == nn.Linear:
		torch.nn.init.xavier_uniform(m.weight)
		m.bias.data.fill_(0.01)

# Generator based on DCGAN 
class Generator(nn.Module):
	# noise dim can be subject to change
	def __init__(self, dim=64, upsample_dim=256, noise_dim=128):
		super(Generator, self).__init__()
		self.linear = nn.Linear(noise_dim, 3 * 3 * upsample_dim)
		self.relu_bn_reshape = nn.Sequential(nn.ReLU(),
											nn.BatchNorm1d(3 * 3 * upsample_dim),
											Unflatten(-1, upsample_dim, 3, 3)
											)
		C = 8 # bottleneck dimension
		self.inv_cnn_block = nn.Sequential(
			nn.ConvTranspose2d(upsample_dim, upsample_dim // 2, 5, stride=2, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(upsample_dim//2),
			nn.ConvTranspose2d(upsample_dim // 2 , upsample_dim // 4, 5, stride=2, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(upsample_dim//4),
			nn.ConvTranspose2d(upsample_dim // 4, upsample_dim // 8, 5, stride=2, padding=1, output_padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(upsample_dim//8))
		self.conv = nn.Conv2d(upsample_dim // 8, C, 7)
		self.pad = nn.modules.padding.ReflectionPad2d(3)    

	def forward(self, inp):
		'''
		input is noise of noise_dim  
		'''
		out = self.linear(inp)
		# print(out.shape)
		out = self.relu_bn_reshape(out)
		out = self.inv_cnn_block(out)
		out = self.pad(out)
		out = self.conv(out)
		return out

class Discriminator(nn.Module):
	def __init__(self, img_size, dim=64):
		super(Discriminator, self).__init__()
		self.img_size = img_size
		self.cnn_block = nn.Sequential(
			nn.Conv2d(self.img_size, dim, 4, stride=2, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(dim, 2 * dim, 4, stride=2, padding=1),
			nn.LeakyReLU(0.2),
			nn.InstanceNorm2d(dim * 2),
			nn.Conv2d(2 * dim, 4 * dim, 4, stride=2, padding=1),
			nn.LeakyReLU(0.2),
			nn.InstanceNorm2d(dim * 4),
			nn.Conv2d(4 * dim, 8 * dim, 4, stride=2, padding=1),
			nn.LeakyReLU(0.2),
			nn.InstanceNorm2d(dim * 8),
			nn.Conv2d(8 * dim, 1, 4, padding=1)            
		)
		# self.sigmoid = nn.Sigmoid()

	def forward(self, input):
		out = self.cnn_block(input)
		# out = self.sigmoid(out)
		return out

# class Quantizer(nn.Module):
#     '''
#     Quantizer architecture implemented
#     '''
#     def __init__(self, temperature=1.0):
#         super(Quantizer, self).__init__()
#         self.centers = torch.arange(-2, 3)
#         self.softmax = nn.Softmax(dim=1)
#         self.temperature = temperature

#     def forward(self, w):
#         w_stack = torch.stack([w for _ in self.centers], dim=1)
#         self.centers = self.centers.float().cuda()
#         self.centers1 = self.centers[None, :, None, None, None] 
#         w_hard = torch.argmin(torch.abs(w_stack - self.centers1).float().cuda() + torch.min(self.centers))
#         smx = self.softmax(-1.0/self.temperature * torch.abs(w_stack - self.centers1))
#         w_soft = torch.einsum('imjkl,m->ijkl', smx, self.centers) 
#         int_w = w_hard - w_soft
#         int_w = int_w.detach()
#         int_w.requires_grad = False

#         w_bar = torch.round(int_w + w_soft)
#         return w_bar

class Quantizer(nn.Module):
	def __init__(self, levels=[i for i in range(-2, 3)], sigma=1.0):
		super(Quantizer, self).__init__()
		self.levels = levels
		self.sigma = sigma

	def forward(self, input):
		levels = input.data.new(self.levels)
		xsize = list(input.size())

		# Compute differentiable soft quantized version
		input = input.view(*(xsize + [1]))
		level_var = Variable(levels, requires_grad=False)
		dist = torch.pow(input-level_var, 2)
		output = torch.sum(level_var * nn.functional.softmax(-self.sigma*dist, dim=-1), dim=-1)

		# Compute hard quantization (invisible to autograd)
		_, symbols = torch.min(dist.data, dim=-1, keepdim=True)
		for _ in range(len(xsize)): levels.unsqueeze_(0)
		levels = levels.expand(*(xsize + [len(self.levels)]))

		quant = levels.gather(-1, symbols.long()).squeeze_(dim=-1)

		# Replace activations in soft variable with hard quantized version
		output.data = quant

		return output
	
class Encoder_orig(nn.Module):
	'''
	latent_dim to which the encoder projects is same dimensional 
	as Generator noise dim
	Can switch instance norm to Batch Norm
	The parameters of Encoder will be dependent on the dataset
	C : bottleneck dimension
	input_dim : channels of input image
	'''
	def __init__(self, input_dim, C, filters=[60, 120, 240, 480, 960]):
		super(Encoder, self).__init__()
		self.input_dim = input_dim
		self.filters = filters
		self.cnn1 = nn.Sequential(nn.Conv2d(input_dim, filters[0], 7, stride=1),
					nn.ReLU(),
					nn.InstanceNorm2d(filters[0]))
		self.cnn2 = nn.Sequential(nn.Conv2d(filters[0], filters[1], 3, stride=2, padding=1),
					nn.ReLU(),
					nn.InstanceNorm2d(filters[1]))
		self.cnn3 = nn.Sequential(nn.Conv2d(filters[1], filters[2], 3, stride=2, padding=1),
					nn.ReLU(),
					nn.InstanceNorm2d(filters[2]))
		self.cnn4 = nn.Sequential(nn.Conv2d(filters[2], filters[3], 3, stride=2, padding=1),
					nn.ReLU(),
					nn.InstanceNorm2d(filters[3]))
		self.cnn5 = nn.Sequential(nn.Conv2d(filters[3], filters[4], 3, stride=2, padding=1),
					nn.ReLU(),
					nn.InstanceNorm2d(filters[4]))
		self.cnn6 = nn.Sequential(nn.Conv2d(filters[4], C, 3, stride=1),
					nn.ReLU(),
					nn.InstanceNorm2d(filters[2]))
		self.pad2 = nn.modules.padding.ReflectionPad2d(1)  
		self.pad1 = nn.modules.padding.ReflectionPad2d(3)    
		
	def forward(self, x):
		out = self.pad1(x)
		out = self.cnn1(out)
		out = self.cnn2(out)
		out = self.cnn3(out)
		out = self.cnn4(out)
		out = self.cnn5(out)
		out = self.pad2(out)
		out = self.cnn6(out)
		return out

class Encoder(nn.Module):
	'''
	latent_dim to which the encoder projects is same dimensional 
	as Generator noise dim
	Can switch instance norm to Batch Norm
	The parameters of Encoder will be dependent on the dataset
	C : bottleneck dimension
	input_dim : channels of input image
	'''
	def __init__(self, input_dim, C, filters=[60, 120, 240, 480, 960]):
		super(Encoder, self).__init__()
		self.input_dim = input_dim
		self.filters = filters
		self.cnn1 = nn.Sequential(nn.Conv2d(input_dim, filters[0], 7, stride=1),
					nn.ReLU(),
					nn.InstanceNorm2d(filters[0]))
		self.cnn2 = nn.Sequential(nn.Conv2d(filters[0], filters[1], 3, padding=1),
					nn.ReLU(),
					nn.InstanceNorm2d(filters[1]))
		self.cnn3 = nn.Sequential(nn.Conv2d(filters[1], filters[2], 3, padding=1),
					nn.ReLU(),
					nn.InstanceNorm2d(filters[2]))
		self.cnn4 = nn.Sequential(nn.Conv2d(filters[2], filters[3], 3, padding=1),
					nn.ReLU(),
					nn.InstanceNorm2d(filters[3]))
		self.cnn5 = nn.Sequential(nn.Conv2d(filters[3], filters[4], 3, padding=1),
					nn.ReLU(),
					nn.InstanceNorm2d(filters[4]))
		self.cnn6 = nn.Sequential(nn.Conv2d(filters[4], C, 3, stride=1),
					nn.ReLU(),
					nn.InstanceNorm2d(filters[2]))
		self.pad2 = nn.modules.padding.ReflectionPad2d(1)  
		self.pad1 = nn.modules.padding.ReflectionPad2d(3)   
		self.maxpool = nn.MaxPool2d(2)
		
	def forward(self, x):
		out = self.pad1(x)
		out1 = self.cnn1(out)
		out = self.maxpool(out1)
		out2 = self.cnn2(out)
		out = self.maxpool(out2)
		out3 = self.cnn3(out)
		out = self.maxpool(out3)
		out4 = self.cnn4(out)
		out = self.maxpool(out4)
		out = self.cnn5(out)
		out = self.pad2(out)
		out = self.cnn6(out)
		return out, [out1, out2, out3, out4]

class SpaceToDepth(nn.Module):
	def __init__(self, block_size):
		super(SpaceToDepth, self).__init__()
		self.block_size = block_size
		self.block_size_sq = block_size*block_size

	def forward(self, input):
		output = input.permute(0, 2, 3, 1)
		(batch_size, s_height, s_width, s_depth) = output.size()
		d_depth = s_depth * self.block_size_sq
		d_width = int(s_width / self.block_size)
		d_height = int(s_height / self.block_size)
		t_1 = output.split(self.block_size, 2)
		stack = [t_t.contiguous().view(batch_size, d_height, d_depth) for t_t in t_1]
		output = torch.stack(stack, 1)
		output = output.permute(0, 2, 1, 3)
		output = output.permute(0, 3, 1, 2)
		return output

class Index_pred(nn.Module):
    def __init__(self, in_channels=60, out_channels=60):
        super(Index_pred, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        out = self.conv1(x)
        # min_val, _ = torch.min(out, dim = 1)
        # min_val, _ = torch.min(min_val, dim = 1)
        # min_val = -min_val.unsqueeze(1).unsqueeze(1)
        max_val, _ = torch.max(out, dim = 1)
        max_val, _ = torch.max(max_val, dim = 1) 
        # max_val = max_val.unsqueeze(1).unsqueeze(1)
        # out = torch.div(torch.add(out, min_val),torch.add(max_val,min_val)) 
        # print(out)
        out = F.sigmoid(out)
        return out

class Decoder(nn.Module):
	'''
	Decoder architecture with Upsampling + Conv2d to avoid checkerboard artefacts
	Decoder architecture for the paper
	filters to mirror the encoder structure
	'''
	def __init__(self, in_channels, filters=[960, 480, 240, 120, 60]):
		super(Decoder, self).__init__()
		res_block = ResidualNet(filters[0], filters[0], kernel_size=3)
		self.res = nn.Sequential(res_block,
								nn.ReLU())
		self.res_iter = nn.ModuleList([self.res for _ in range(9)])
		self.conv1 = nn.Sequential(nn.Conv2d(in_channels, filters[0], 3),
								   nn.BatchNorm2d(filters[0]),
								   nn.ReLU())
		self.conv_tr1 = nn.Sequential(nn.PixelShuffle(2),
									  nn.Conv2d(filters[2], filters[1], 3, stride=1, padding=1),
									  nn.BatchNorm2d(filters[1]),
									  nn.ReLU()
									  )
		self.conv_tr2 = nn.Sequential(nn.PixelShuffle(2),
									  nn.Conv2d(filters[3], filters[2], 3, stride=1, padding=1),
									  nn.BatchNorm2d(filters[2]),
									  nn.ReLU())        
		self.conv_tr3 = nn.Sequential(nn.PixelShuffle(2),
									  nn.Conv2d(filters[4], filters[3], 3, stride=1, padding=1),
									  nn.BatchNorm2d(filters[3]),
									  nn.ReLU())
		self.conv_tr4 = nn.ConvTranspose2d(filters[3], filters[4], 3, padding=1)
		self.MaxUnpool4 = nn.MaxUnpool2d(2)
		self.BatchNorm4 = nn.BatchNorm2d(filters[4]) 
		
		self.tanh = nn.Tanh()
		self.pad1 = nn.modules.padding.ReflectionPad2d(1)      
		self.pad2 = nn.modules.padding.ReflectionPad2d(3) 
		self.conv2 = nn.Conv2d(filters[4], 3, 7) 
		self.relu = nn.ReLU()
		# self._initialize_weights()
		self._initialize_weights_ICNR()
		self.ind_pred = Index_pred()
		saved_dict = torch.load('index_pred1.pth')
		self.ind_pred.load_state_dict(saved_dict.state_dict())

	def ICNR(self, tensor, upscale_factor=2, inizializer=nn.init.kaiming_normal):
	
		new_shape = [int(tensor.shape[0] / (upscale_factor ** 2))] + list(tensor.shape[1:])
		subkernel = torch.zeros(new_shape)
		subkernel = inizializer(subkernel)
		subkernel = subkernel.transpose(0, 1)

		subkernel = subkernel.contiguous().view(subkernel.shape[0],
												subkernel.shape[1], -1)

		kernel = subkernel.repeat(1, 1, upscale_factor ** 2)

		transposed_shape = [tensor.shape[1]] + [tensor.shape[0]] + list(tensor.shape[2:])
		kernel = kernel.contiguous().view(transposed_shape)

		kernel = kernel.transpose(0, 1)

		return kernel

	def inverse_map(self, I, P = 2): 
	    """
	    Transforms the local indices to global indices
	    Inputs:
	    - I: local indices must be torch.float
	    - P: pool_kernel_size
	    """
	    W = I.size(-1)*2
	    I_f = torch.floor((I/P).float()).long().to(config_train.device)
	    I_r = torch.remainder(I, P).long().to(config_train.device)
	    h, w = I.size(-2), I.size(-1)

	    col_indices = torch.arange(w).view(-1,1).to(config_train.device)
	    A_j = col_indices.expand(-1,h)
	    A_i = A_j.transpose(1,0)
	    global_ind = W*A_j*P + A_i*P + I_f*W + I_r
	    return global_ind

	def _initialize_weights(self):
		init.orthogonal_(self.conv_tr1.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.conv_tr2.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.conv_tr3.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.conv_tr4.weight)

	def _initialize_weights_ICNR(self):
		kernel = self.ICNR(self.conv_tr1[1].weight, upscale_factor=2, inizializer=nn.init.kaiming_normal)
		self.conv_tr1[1].weight.data.copy_(kernel)
		kernel = self.ICNR(self.conv_tr2[1].weight, upscale_factor=2, inizializer=nn.init.kaiming_normal)
		self.conv_tr2[1].weight.data.copy_(kernel)
		kernel = self.ICNR(self.conv_tr3[1].weight, upscale_factor=2, inizializer=nn.init.kaiming_normal)
		self.conv_tr3[1].weight.data.copy_(kernel)
		kernel = self.ICNR(self.conv_tr4.weight, upscale_factor=2, inizializer=nn.init.kaiming_normal)
		self.conv_tr4.weight.data.copy_(kernel)
		print("Initialized ICNR weights")

	def forward(self, x):  
		x_pad = self.pad1(x)
		out = self.conv1(x_pad)
		for layer in self.res_iter:
			out = layer(out)
		out4 = self.conv_tr1(out)
		out3 = self.conv_tr2(out4)
		out2 = self.conv_tr3(out3)
		# Fourth stage decoder conv block
		out = self.conv_tr4(out2)
		temp_ind = self.inverse_map(self.ind_pred(out))
		temp_ind = Variable(temp_ind, requires_grad=False)
		out1 = self.MaxUnpool4(out, temp_ind)
		out = self.BatchNorm4(out1)
		out = self.relu(out)
		out = self.pad2(out)
		out = self.conv2(out)
		out = self.tanh(out)
		return out, [out1,out2,out3,out4]

class Decoder_new(nn.Module):
	'''
	Decoder architecture with Upsampling + Conv2d to avoid checkerboard artefacts
	Decoder architecture for the paper
	filters to mirror the encoder structure
	'''
	def __init__(self, in_channels, filters=[960, 480, 240, 120, 60]):
		super(Decoder_new, self).__init__()
		res_block = ResidualNet(filters[0], filters[0], kernel_size=3)
		self.res = nn.Sequential(res_block,
								nn.ReLU())
		self.res_iter = nn.ModuleList([self.res for _ in range(9)])
		self.conv1 = nn.Sequential(nn.Conv2d(in_channels, filters[0], 3),
								   nn.ReLU(),
								   nn.BatchNorm2d(filters[0]))
		self.conv_tr1 = nn.ConvTranspose2d(filters[0], filters[1], 3, padding=1)
		self.BatchNorm1 = nn.BatchNorm2d(filters[1])
		self.relu = nn.ReLU()
		self.conv_tr2 = nn.ConvTranspose2d(filters[1], filters[2], 3, padding=1)
		self.BatchNorm2 = nn.BatchNorm2d(filters[2])      
		self.conv_tr3 = nn.ConvTranspose2d(filters[2], filters[3], 3, padding=1)
		self.BatchNorm3 = nn.BatchNorm2d(filters[3])
		self.conv_tr4 = nn.ConvTranspose2d(filters[3], filters[4], 3, padding=1)
		self.BatchNorm4 = nn.BatchNorm2d(filters[4]) 
		
		self.tanh = nn.Tanh()
		self.pad1 = nn.modules.padding.ReflectionPad2d(1)      
		self.pad2 = nn.modules.padding.ReflectionPad2d(3) 
		self.conv2 = nn.Conv2d(filters[4], 3, 7) 
		# self._initialize_weights()

	def forward(self, x, ind):  
		x_pad = self.pad1(x)
		out = self.conv1(x_pad)
		for layer in self.res_iter:
			out = layer(out)
			
		# First stage decoder conv block
		out4 = F.max_unpool2d(self.conv_tr1(out), ind[3], 2)
		out = self.BatchNorm1(out4)
		out = self.relu(out)

		# Second stage decoder conv block
		out3 = F.max_unpool2d(self.conv_tr2(out), ind[2], 2)
		out = self.BatchNorm2(out3)
		out = self.relu(out)

		# Third stage decoder conv block
		out2 = F.max_unpool2d(self.conv_tr3(out), ind[1], 2)
		out = self.BatchNorm3(out2)
		out = self.relu(out)

		# Fourth stage decoder conv block
		out1 = F.max_unpool2d(self.conv_tr4(out), ind[0], 2)
		out = self.BatchNorm4(out1)
		out = self.relu(out)

		out = self.pad2(out)
		out = self.conv2(out)
		out = self.tanh(out)
		return out, [out1, out2, out3, out4]
