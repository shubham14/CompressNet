'''
Contains util functions for models for the 
final generative compression architecture
'''
from collections import namedtuple
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
import torch.autograd as autograd
from config import config_train, directories
import torch.nn.functional as F

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        reqd_list = [0,2,4,5,7,9,10,12,14,16,17,19,21,23,24,26,28,30]
        self.slice = nn.ModuleList([nn.Sequential() for _ in range(len(reqd_list))])
        self.slice[0].add_module(str(0), vgg_pretrained_features[0])
        for i in range(1,len(reqd_list)):
            for j in range(reqd_list[i-1],reqd_list[i]):
                self.slice[i].add_module(str(j), vgg_pretrained_features[j])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        reqd_list = [0, 2, 4, 5, 7, 9, 10, 12, 14, 16, 17, 19, 21, 23, 24, 26, 28, 30]
        out = [X]*len(reqd_list)
        print("Output Type",type(out[0]))
        out[0] = self.slice[0](X)
        for i in range(1,len(reqd_list)):
            out[i] = self.slice[i](out[i - 1])

        return out

class AlNetFeatureExtractor(nn.Module):
    '''
    New model constructed till the 4th Conv layer
    of the pretrained AlexNet
    '''
    def __init__(self, original_model):
        super(AlNetFeatureExtractor, self).__init__()
        self.model = nn.Sequential(*list(original_model.features.children())[:7])
    
    def forward(self, inp):
        out = self.model(inp)
        return out

alexnet = models.alexnet(pretrained=True)
device = config_train.device

alnet = AlNetFeatureExtractor(alexnet).to(device)

# creating data iterator
class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.
    """
    ###########################
    ######### TO DO ###########
    ###########################
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3, betas=(0.5, 0.999))
    return optimizer

def perceptual_loss(input, target):
    input_features = alnet(input).float().cuda()
    target_features = alnet(target).float().cuda()
    perceptual_loss = F.mse_loss(input_features,target_features)
    return perceptual_loss

# def perceptual_loss(vgg_model, input, target):
#     input_features = vgg_model(input)
#     target_features = vgg_model(target)
#     loss = F.mse_loss(input_features.relu2_2, target_features.relu2_2)
#     return loss


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    """
    neg_abs = - input.abs()
    z = torch.zeros_like(neg_abs)
    loss1 = torch.max(input, z) - input * target + (1 + neg_abs.exp()).log()
    loss = loss1.mean()
    return loss


def discriminator_loss(logits_real, logits_fake, dtype):
    """
    Computes the discriminator loss
    """
    dis_targets = torch.ones_like(logits_real)
    loss1 = bce_loss(logits_real, dis_targets)
    
    gen_targets = torch.zeros_like(logits_fake)
    loss2 = bce_loss(1 - logits_fake, gen_targets)
    loss = loss1 + loss2
    return loss


def generator_loss(logits_fake, dtype):
    """
    Computes the generator loss 
    """
    loss = None
    dis_targets = torch.ones_like(logits_fake)
    loss = bce_loss(logits_fake, dis_targets)
    return loss

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

    
class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

def Gen_AE_Loss(x, x_hat, Alex_model):
    '''
    Custom loss for Generative Decoder (in the AE part)
    Alex_model : sliced pretrained AlexNet model till 4th Conv layer
    '''
    loss_1 = torch.sqrt(torch.sum((x - x_hat) ** 2))
    conv_x = Alex_model(x)
    conv_x_hat = Alex_model(x_hat)
    loss_2 = torch.sqrt(torch.sum((conv_x - conv_x_hat) ** 2))
    loss = loss_1 + loss_2
    return loss

# class StyleLoss(nn.Module):
#     '''
#     target_layer can take values from [1, 3, 6]\
#     alnet_feat : alexnet feature extractor
#     '''
#     def __init__(self, alnet_feat):
#         super(StyleLoss, self).__init__()
#         self.alnet_feat = alnet_feat
        

#     def forward(self, input, target_image):
#         G = gram_matrix(self.alnet_feat(input))
#         target = gram_matrix(self.alnet(target_image)).detach()
#         loss = F.mse_loss(G, target)
#         return loss

def StyleLoss(input, target):
    input_features = alnet(input).float().cuda()
    target_features = alnet(target).float().cuda()
    Gram_input_features =  gram_matrix(input_features)
    Gram_target_features =  gram_matrix(target_features)
    Gram_target_features.detach()
    Gram_input_features.detach()
    style_loss = F.mse_loss(Gram_input_features, Gram_target_features)
    return style_loss

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(b*c*d)



class ResidualNet(nn.Module):
    def __init__(self, in_channel, n_filters, kernel_size=3):
        super(ResidualNet, self).__init__()
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.pad = (kernel_size - 1)//2
        self.cnn1 = nn.Sequential(nn.Conv2d(in_channel, n_filters, kernel_size),
                                  nn.BatchNorm2d(n_filters),
                                  nn.ReLU()) #in_channel depends on the image size
        self.cnn2 = nn.Sequential(nn.Conv2d(n_filters, n_filters, kernel_size),
                                  nn.BatchNorm2d(n_filters),
                                  nn.ReLU())
    
    def forward(self, x):
        x_pad = nn.modules.padding.ReflectionPad2d(self.pad)(x)
        cnn1_out = self.cnn1(x_pad)
        cnn1_out = nn.modules.padding.ReflectionPad2d(self.pad)(cnn1_out)
        cnn2_out = self.cnn2(cnn1_out)
        out = cnn2_out + x
        return out


def model_load_G_D(gen, dis):
    '''
    gen and dis are Generator and Discriminator models
    load pretrained cifar-10 GAN weights 
    '''
    pretrained_G = torch.load('netG_epoch_199.pth', map_location='cpu')
    pretrained_D = torch.load('netD_epoch_199.pth', map_location='cpu')
    
    model_dict_gen = gen.state_dict()
    model_dict_dis = dis.state_dict()

    pretrained_dict_G = {k: v for k, v in pretrained_G.items() if k != 'main.12.weight'}
    pretrained_dict_D = {k: v for k, v in pretrained_D.items()}
    
    model_dict_gen.update(pretrained_dict_G)
    model_dict_dis.update(pretrained_dict_D)
    
    dis.load_state_dict(model_dict_dis)
    gen.load_state_dict(model_dict_gen)
    
    return gen, dis
    

def gradient_penalty(netD, real_data, fake_data, lamda =1):
    
    alpha = torch.rand(real_data.size())
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(config_train.device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(config_train.device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs = torch.ones(disc_interpolates.size()).to(config_train.device),
                              create_graph = True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lamda
    return gradient_penalty

# if __name__ == "__main__":
#     NUM_TRAIN = 50000
#     NUM_VAL = 5000

#     NOISE_DIM = 96
#     batch_size = 128

#     mnist_train = dset.MNIST('./data', train=True, download=True,
#                             transform=T.ToTensor())
#     loader_train = DataLoader(mnist_train, batch_size=batch_size,
#                             sampler=ChunkSampler(NUM_TRAIN, 0))
