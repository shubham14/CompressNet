'''
Contains the architecture for Generative Compression
'''
from collections import namedtuple
from network import  Generator, Discriminator, Encoder, Decoder, Quantizer
import torch
import torch.nn as nn
from torch.autograd import Variable
from data_load import *
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from model_utils import perceptual_loss, StyleLoss, Vgg16, gradient_penalty
from config import *
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        input_dim = 3
        filter_dims = [60, 120, 240, 480, 960]
        #C = 16#{2,4,8,16}
        reversed_filter_dims = filter_dims[::-1]
        self.config = config
        self.encoder = Encoder(input_dim, C = self.config.channel_bottleneck, filters= filter_dims)
        self.quantizer = Quantizer()
        self.dcgan_generator = Generator(dim=64, upsample_dim=256)
        self.decoder = Decoder(in_channels = self.config.channel_bottleneck, filters=reversed_filter_dims) #CHANGE THIS 960 to run it temporarily
        self.discriminator = Discriminator(img_size=input_dim, dim=64)
        #self.multiscale_discriminator = Multiscale_discriminator()

    def forward(self, input_data, evaluate=False):
        if self.config.use_conditional_GAN:
            self.example, semantic_map = input_data
        else:
            self.example = input_data

        # Global generator: Encode -> quantize -> reconstruct
        # =======================================================================================================>>>
        feature_map, self.intermediate_layers, self.indices = self.encoder(self.example)#, config, self.training_phase, self.config.channel_bottleneck)
        # print("Feature_Map Size: ",(feature_map.size()))
        w_hat = self.quantizer(feature_map)
        # w_hat = feature_map
        # if self.config.use_conditional_GAN:
        #     semantic_feature_map = self.encoder(semantic_map)#, config, self.training_phase,self.config.channel_bottleneck, scope='semantic_map')
        #     w_hat_semantic = self.quantizer(semantic_feature_map)#, config, scope='semantic_map')
        #     w_hat = torch.cat((w_hat, w_hat_semantic), 1) 

        if self.config.sample_noise:
            # print('Sampling noise...',self.example.size())
            noise_prior = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros((self.config.noise_dim)), torch.eye(self.config.noise_dim))
            v = noise_prior.sample([self.example.size(0)])
            # print(v.size())
            v = v.to(self.config.device)
            Gv = self.dcgan_generator(v)#, config, self.training_phase, C=self.config.channel_bottleneck, upsample_dim=self.config.upsample_dim)
            # print(Gv.size())
            # print(w_hat.size())

            z = torch.cat((w_hat, Gv), 1)
        else:
            z = w_hat
        # print("Quatized Size: ",(z.size()))

        self.reconstruction, self.rec_intermediate_layers = self.decoder(z, self.indices)#, config, self.training_phase, C=self.config.channel_bottleneck)
        # print('Real image shape:', self.example.size())
        # print('Reconstruction shape:', self.reconstruction.size())
        # self.reconstruction = 0.5 * (self.reconstruction + 1)

        if evaluate:
            return self.reconstruction

        if self.config.multiscale:
            self.D_x, self.D_x2, self.D_x4, *self.Dk_x = self.multiscale_discriminator(self.example)#, config, self.training_phase, use_sigmoid=self.config.use_vanilla_GAN, mode='real')
            self.D_Gz, self.D_Gz2, self.D_Gz4, *self.Dk_Gz = self.multiscale_discriminator(self.reconstruction)#, config, self.training_phase, use_sigmoid=self.config.use_vanilla_GAN, mode='reconstructed', reuse=True)
        else:
            self.D_x = self.discriminator(self.example)#, config, self.training_phase, use_sigmoid=self.config.use_vanilla_GAN)
            self.D_Gz = self.discriminator(self.reconstruction)#, config, self.training_phase, use_sigmoid=self.config.use_vanilla_GAN, reuse=True)
         
        # Loss terms 
        # =======================================================================================================>>>
    def loss(self):
        if self.config.use_vanilla_GAN is True:
        #     # Minimize JS divergence
            D_loss_real = torch.mean(nn.BCELoss()(F.sigmoid(self.D_x), torch.ones_like(self.D_x)))
            D_loss_gen = torch.mean(nn.BCELoss()(F.sigmoid(self.D_Gz), torch.zeros_like(self.D_Gz)))
            D_loss = 0.5 * (D_loss_real + D_loss_gen)
            # G_loss = max log D(G(z))
            G_loss = torch.mean(nn.BCELoss()(F.sigmoid(self.D_Gz), torch.ones_like(self.D_Gz)))
        # elif self.config_use_WGAN is True:
        #     D_loss_real = -(torch.mean(self.D_x) 
        #     D_loss_gen = - torch.mean(self.D_Gz))
        #     G_loss = -torch.mean(self.D_Gz)
        else:
            # Minimize $\chi^2$ divergence
            D_loss_real = -(torch.mean(self.D_x))
            D_loss_gen = - (torch.mean(self.D_Gz))
            G_loss = -torch.mean(self.D_Gz)

        criterion = nn.MSELoss()

        intermediate_loss = 0
        for i, encoder_layer in enumerate(self.intermediate_layers):
            intermediate_loss += criterion(self.rec_intermediate_layers[i], encoder_layer)
       

        vgg_model = Vgg16().to(config_train.device)
        p_loss = perceptual_loss(self.reconstruction,self.example)
        style_loss = StyleLoss(self.reconstruction,self.example)
        gp_loss = gradient_penalty(self.discriminator, self.example, self.reconstruction)
        distortion_penalty = self.config.lambda_X * criterion(self.example,self.reconstruction) + self.config.lambda_P * p_loss #+ style_loss #CHANGE this to self.example to run it temporarily
        G_loss += distortion_penalty
        G_loss += intermediate_loss
        R_loss = self.config.lambda_X * criterion(self.example,self.reconstruction)
        return G_loss, D_loss_real, D_loss_gen, R_loss, self.config.lambda_P * p_loss, gp_loss, intermediate_loss#, style_loss
