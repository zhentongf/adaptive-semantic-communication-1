#!/usr/bin/env python
# encoding: utf-8
"""
This script implements a deep learning pipeline for CIFAR-10 image classification 
with compression and noise addition. It uses:
- A GoogleNet classifier
- A custom RED-CNN autoencoder for compression/decompression
- Training with both classification and reconstruction losses
- SNR-controlled noise addition in the compressed domain
"""

import os
# Import necessary libraries
import torch
print(torch.__version__)            # Check PyTorch version
print(torch.cuda.is_available())    # Should return True if GPU is available
print(f"CUDA available: {torch.cuda.is_available()}")  # Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import copy
from torch import nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import warnings
import argparse
import imageio

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def get_argparser():
    """Define command line arguments for the script"""
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--alpha", type=float, default=0.7,
                      help="parameter in loss function")
    parser.add_argument("--random_seed", type=int, default=99,
                      help='seed of random sequence')

    return parser

# Constants for model configuration
raw_dim = 32 * 32  # shape of the raw image
manualSeed = 999
batch_size = 32
image_size = 64
nc = 3  # number of channels (RGB)
nz = 100
ngf = 64
ndf = 64
num_epochs = 100  # number of epochs
lr = 0.0002  # learning rate
beta1 = 0.5
ngpu = 4

def data_tf(x):
    """Image transformation function for preprocessing"""
    x = x.resize((96, 96), 2)  # Resize to 96x96 using bicubic interpolation
    x = np.array(x, dtype='float32') / 255  # Normalize to [0,1]
    x = x.transpose((2, 0, 1))  # Change from HWC to CHW format
    x = torch.from_numpy(x)
    return x

def conv_relu(in_channels, out_channels, kernel, stride=1, padding=0):
    """Helper function to create a conv-BN-ReLU block"""
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
        nn.BatchNorm2d(out_channels, eps=1e-3),
        nn.ReLU(True)
    )
    return layer

class inception(nn.Module):
    """Inception module as used in GoogleNet"""
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
        super(inception, self).__init__()
        # Four parallel pathways
        self.branch1x1 = conv_relu(in_channel, out1_1, 1)  # 1x1 conv
        
        self.branch3x3 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1)  # 1x1 followed by 3x3
        )
        
        self.branch5x5 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2)  # 1x1 followed by 5x5
        )
        
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 1)  # Pooling followed by 1x1
        )

    def forward(self, x):
        f1 = self.branch1x1(x)
        f2 = self.branch3x3(x)
        f3 = self.branch5x5(x)
        f4 = self.branch_pool(x)
        output = torch.cat((f1, f2, f3, f4), dim=1)  # Concatenate all branches
        return output

class googlenet(nn.Module):
    """GoogleNet architecture for CIFAR-10 classification"""
    def __init__(self, in_channel, num_classes, verbose=False):
        super(googlenet, self).__init__()
        self.verbose = verbose
        
        # Define the network blocks
        self.block1 = nn.Sequential(
            conv_relu(in_channel, out_channels=64, kernel=7, stride=2, padding=3),
            nn.MaxPool2d(3, 2)
        )
        self.block2 = nn.Sequential(
            conv_relu(64, 64, kernel=1),
            conv_relu(64, 192, kernel=3, padding=1),
            nn.MaxPool2d(3, 2)
        )
        self.block3 = nn.Sequential(
            inception(192, 64, 96, 128, 16, 32, 32),
            inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2)
        )
        self.block4 = nn.Sequential(
            inception(480, 192, 96, 208, 16, 48, 64),
            inception(512, 160, 112, 224, 24, 64, 64),
            inception(512, 128, 128, 256, 24, 64, 64),
            inception(512, 112, 144, 288, 32, 64, 64),
            inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2)
        )
        self.block5 = nn.Sequential(
            inception(832, 256, 160, 320, 32, 128, 128),
            inception(832, 384, 182, 384, 48, 128, 128),
            nn.AvgPool2d(2)
        )
        
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Forward pass through each block
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.classifier(x)  # Final classification layer
        return x

def get_acc(output, label):
    """Calculate accuracy given model outputs and true labels"""
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def merge_images(sources, targets):
    """Merge source and target images side-by-side for visualization."""
    sources = to_data(sources)
    targets = to_data(targets)
    
    B, C, H, W = sources.shape
    merged = np.zeros((B * H, W * 2, C), dtype=np.float32)

    for idx in range(B):
        s = sources[idx].transpose(1, 2, 0)  # (H, W, C)
        t = targets[idx].transpose(1, 2, 0)
        merged[idx * H:(idx + 1) * H, :W, :] = s
        merged[idx * H:(idx + 1) * H, W:, :] = t

    # Scale and convert to uint8
    merged = np.clip(merged * 255.0, 0, 255).astype(np.uint8)
    return merged


def to_data(x):
    """Convert variable to numpy array, moving to CPU if needed"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x

# Generate 10 random SNR values between 0 and 20 dB
random.seed(42)  # For reproducibility
snr_values = [random.uniform(0, 20) for _ in range(10)]
print("SNR values to test:", snr_values)

# Outer loop for different SNR values
for snr_idx, snr in enumerate(snr_values):
    print(f"\n=== Testing SNR: {snr:.2f} dB ===")

    # Load CIFAR-10 dataset
    train_set = datasets.CIFAR10('./datasets/cifar10', train=True, transform=data_tf, download=True)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_set = datasets.CIFAR10('./datasets/cifar10', train=False, transform=data_tf, download=True)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    # Initialize and load pretrained GoogleNet
    opts = get_argparser().parse_args()
    torch.manual_seed(opts.random_seed)
    
    classifier = googlenet(3, 10)
    classifier.load_state_dict(torch.load('saved_model/google_net.pkl'))
    classifier.to(device)
    optimizer_classifier = torch.optim.SGD(classifier.parameters(), lr=0.0001)
    criterion_classifier = nn.CrossEntropyLoss()

    # Test different compression rates
    for rate in range(6, 10):
        compression_rate = min((rate + 1) * 0.1, 1)
        channel = max(np.sqrt(96 * (1 - compression_rate) / 3), 1)
        channel = int(channel)
        print('channel:', channel)

        dimension = int(96 * 96 * 3 * compression_rate / (8 * 8))
        size_recover = int(96 * np.sqrt(compression_rate))

        class RED_CNN(nn.Module):
            """Autoencoder for compression/decompression with noise injection"""
            def __init__(self, out_ch=16):
                super(RED_CNN, self).__init__()
                # Encoder layers
                self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=0)
                self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=0)
                self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0)
                self.conv4 = nn.Conv2d(256, 384, kernel_size=5, stride=1, padding=0)
                self.conv5 = nn.Conv2d(384, 512, kernel_size=5, stride=1, padding=0)
                self.conv6 = nn.Conv2d(512, dimension, kernel_size=3, stride=1, padding=0)
                
                # Decoder layers
                self.tconv1 = nn.ConvTranspose2d(dimension, 512, kernel_size=3, stride=1, padding=0)
                self.tconv2 = nn.ConvTranspose2d(512, 384, kernel_size=5, stride=1, padding=0)
                self.tconv3 = nn.ConvTranspose2d(384, 256, kernel_size=5, stride=1, padding=0)
                self.tconv4 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1, padding=0)
                self.tconv5 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=0)
                self.tconv6 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=0)

            def forward(self, x, snr_value):
                # Encoder
                out = F.relu(self.conv1(x))
                out = F.relu(self.conv2(out))
                out = F.relu(self.conv3(out))
                out = F.relu(self.conv4(out))
                out = F.relu(self.conv5(out))
                out = F.relu(self.conv6(out))
                
                # Add noise in compressed domain based on SNR
                out_tmp = out.detach().cpu().numpy()
                out_square = np.square(out_tmp)
                aver = np.sum(out_square) / np.size(out_square)
                
                aver_noise = aver / 10 ** (snr_value / 10)
                noise = torch.randn(size=out.shape) * np.sqrt(aver_noise)
                noise = noise.to(device)
                out = torch.add(out, noise)
                
                # Decoder
                out = F.relu(self.tconv1(out))
                out = F.relu(self.tconv2(out))
                out = F.relu(self.tconv3(out))
                out = F.relu(self.tconv4(out))
                out = F.relu(self.tconv5(out))
                out = F.relu(self.tconv6(out))
                return out

        # Initialize and load pretrained autoencoder if available
        mlp_encoder = RED_CNN().to(device)
        model_path = ("./saved_model/CIFAR_encoder_%.6f_snr_%.2f.pkl" % (compression_rate, snr))
        pre_model_exist = os.path.isfile(model_path)
        
        if pre_model_exist:
            print('load model parameters ...')
            mlp_encoder.load_state_dict(torch.load(model_path))
        else:
            print('No Well-Trained Model!')

        def criterion(x_in, y_in, raw_in):
            """Combined loss function for reconstruction and classification"""
            out_tmp1 = nn.CrossEntropyLoss()
            out_tmp2 = nn.MSELoss()
            z_in = classifier(x_in)
            loss_channel = out_tmp2(x_in, raw_in)  # MSE loss for reconstruction
            return loss_channel

        # Training variables
        losses = []
        acces = []
        eval_losses = []
        eval_acces = []
        psnr_all = []
        psnr = None
        acc_real = None

        print('Training Start')
        print('Compression Rate:', compression_rate)
        print('SNR:', snr)
        epoch_len = 20
        
        # Training loop
        for e in range(epoch_len):
            # Adjust learning rate based on epoch
            if e < 5:
                optimizer = torch.optim.Adam(mlp_encoder.parameters(), 1e-3)
            elif e < 10:
                optimizer = torch.optim.Adam(mlp_encoder.parameters(), 1e-4)
            else:
                optimizer = torch.optim.Adam(mlp_encoder.parameters(), 2e-5)

            train_loss = 0
            train_acc = 0
            psnr_aver = 0
            mlp_encoder.train()
            counter = 0
            
            # Batch training
            for im, label in train_data:
                im = im.to(device)
                label = label.to(device)
                
                # Adaptive transmission based on SNR
                if snr < 10:
                    # Use neural network for SNR < 10
                    out = mlp_encoder(im, snr)
                else:
                    # Copy raw image for SNR >= 10
                    out = im.clone()
                    out_np = out.detach().cpu().numpy()
                    out_square = np.square(out_np)
                    aver = np.sum(out_square) / np.size(out_square)
                    aver_noise = aver / 10 ** (snr / 10)
                    noise = np.random.random(size=out_np.shape) * np.sqrt(aver_noise)
                    noise = torch.from_numpy(noise).to(device).to(torch.float32)
                    # Add Gaussian noise based on SNR
                    out = torch.add(out, noise)
                
                out_mnist = classifier(out)
                out_real = classifier(im)
                
                # Calculate loss and metrics
                loss = criterion(out, label, im)
                cr1 = nn.MSELoss()
                mse = cr1(out, im)
                psnr = 10 * (np.log(1. / mse.item()) / np.log(10))
                psnr_aver += psnr
                
                # Only update encoder if using neural network
                if snr < 10:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                counter += 1
                train_loss += loss.item()
                
                # Calculate accuracy
                _, pred = out_mnist.max(1)
                num_correct = (pred == label).sum().item()
                acc = num_correct / im.shape[0]
                train_acc += acc

                # Save sample images periodically
                if e % 10 == 0 and counter == 1:
                    im_data = to_data(im)
                    out_data = to_data(out)
                    merged = merge_images(im_data, out_data)

                    # save the images
                    path = os.path.join('./image_recover_combing/cifar10/merged_images/sample-epoch-%d-compre-%.2f-snr-%.2f.png' % (
                        e, compression_rate, snr))
                    # scipy.misc.imsave(path, merged)
                    print(f"merged shape: {merged.shape}, dtype: {merged.dtype}")
                    imageio.imwrite(path, merged)
                    print('saved %s' % path)

            # Epoch statistics
            losses.append(train_loss / counter)
            acces.append(train_acc / counter)
            psnr_all.append(psnr_aver / counter)
            train_acc = train_acc / counter
            train_loss = train_loss / counter
            psnr_aver = psnr_aver / counter

            # Validation
            eval_loss = 0
            eval_acc = 0
            eval_psnr = 0
            mlp_encoder.eval()
            counter = 0
            
            with torch.no_grad():
                for im, label in test_data:
                    im = im.to(device)
                    label = label.to(device)
                    
                    # Adaptive transmission based on SNR
                    if snr < 10:
                        # Use neural network for SNR < 10
                        out = mlp_encoder(im, snr)
                    else:
                        # Copy raw image for SNR >= 10
                        out = im.clone()
                        out_np = out.detach().cpu().numpy()
                        out_square = np.square(out_np)
                        aver = np.sum(out_square) / np.size(out_square)
                        aver_noise = aver / 10 ** (snr / 10)
                        noise = np.random.random(size=out_np.shape) * np.sqrt(aver_noise)
                        noise = torch.from_numpy(noise).to(device).to(torch.float32)
                        # Add Gaussian noise based on SNR
                        out = torch.add(out, noise)
                    
                    out_mnist = classifier(out)
                    
                    loss = criterion(out, label, im)
                    eval_loss += loss.item()
                    
                    _, pred = out_mnist.max(1)
                    num_correct = (pred == label).sum().item()
                    acc = num_correct / im.shape[0]
                    eval_acc += acc
                    
                    cr1 = nn.MSELoss()
                    mse = cr1(out, im)
                    psnr = 10 * (np.log(1. / mse.item()) / np.log(10))
                    eval_psnr += psnr
                    counter += 1

            # Print epoch results
            print('epoch: {}, Test Acc: {:.6f}, Test Loss: {:.6f},'
                 'Test PSNR: {:.6f}, Train Acc: {:.6f}, Train Loss: {:.6f}, Train PSNR: {:.6f}'
                 .format(e, eval_acc / counter, eval_loss / counter,
                         eval_psnr / counter, train_acc, train_loss,
                         psnr_aver))

            # Save model and results (only if using neural network)
            if snr < 10:
                torch.save(mlp_encoder.state_dict(), ('saved_model/CIFAR_encoder_%f_snr_%.2f.pkl' % (compression_rate, snr)))
            
            # Save accuracy and PSNR results with SNR in filename
            file = ('./results/MLP_sem_CIFAR/acc_semantic_combining_%.2f_snr_%.2f.csv' % (
                compression_rate, snr))
            data = pd.DataFrame(acces)
            data.to_csv(file, index=False)

            eval_psnr = np.array(psnr_all)
            file = ('./results/MLP_sem_CIFAR/psnr_semantic_combining_%.2f_snr_%.2f.csv' % (
                compression_rate, snr))
            data = pd.DataFrame(eval_psnr)
            data.to_csv(file, index=False)
            # save the recovered image (only if using neural network)
            if snr < 10 and out is not None:
                for ii in range(min(len(out), 10)):  # Save only first 10 images to avoid too many files
                    pil_img = Image.fromarray(np.uint8(out[ii].detach().cpu().numpy().transpose(1, 2, 0) * 255))
                    pil_img.save(
                        "./image_recover_combing/cifar10/cifar_train_%d_%f_snr_%.2f.jpg" % (ii, compression_rate, snr))

print("All experiments completed!")
