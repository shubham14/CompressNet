#!/usr/bin/env python3
import torch 
import os


class config_train(object):
    mode = 'gan-train'
    num_epochs = 100
    batch_size = 2
    ema_decay = 0.999
    G_learning_rate = 5e-3
    D_learning_rate = 5e-4
    lr_decay_rate = 2e-5
    momentum = 0.9
    weight_decay = 5e-4
    noise_dim = 128
    optimizer = 'adam'
    kernel_size = 3
    diagnostic_steps = 256
    workers = 1
    mirror = True
    
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # WGAN
    gradient_penalty = True
    lambda_gp = 1
    weight_clipping = False
    max_c = 1e-2
    n_critic_iterations = 20

    # Compression
    lambda_X = 1
    lambda_P = 5
    channel_bottleneck = 8
    sample_noise = False
    use_vanilla_GAN = True
    use_feature_matching_loss = False
    upsample_dim = 256
    multiscale = False
    feature_matching_weight = 10
    use_conditional_GAN = False

class config_test(object):
    mode = 'gan-test'
    num_epochs = 512
    batch_size = 1
    ema_decay = 0.999
    G_learning_rate = 2e-4
    D_learning_rate = 2e-4
    lr_decay_rate = 2e-5
    momentum = 0.9
    weight_decay = 5e-4
    noise_dim = 128
    optimizer = 'adam'
    kernel_size = 3
    diagnostic_steps = 256
    mirror = False
    workers = 1

    # WGAN
    gradient_penalty = True
    lambda_gp = 10
    weight_clipping = False
    max_c = 1e-2
    n_critic_iterations = 5

    # Compression
    lambda_X = 1
    channel_bottleneck = 8
    sample_noise = False
    use_vanilla_GAN = True
    use_feature_matching_loss = True
    upsample_dim = 256
    multiscale = False
    feature_matching_weight = 10
    use_conditional_GAN = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class directories(object):
    data_type = 'Professional' # Professional Or Mobile
    root = './Dataset/'
    if data_type == 'Professional':
        train = 'train/'
        test = 'test/'
        val = 'professional_valid/valid/'
    else:
        train = 'train/'
        val = 'valid/'
        test = 'test/'
