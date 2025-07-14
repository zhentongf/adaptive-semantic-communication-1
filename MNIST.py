# encoding: utf-8

import os
import torch
from torchvision.datasets import mnist
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import copy
from torch import nn
import scipy
from torch.autograd import Variable
from PIL import Image
import pdb
import random

# Define the raw dimension of the MNIST image (28x28)
raw_dim = 28 * 28

# Function to normalize and flatten image data
def data_transform(x):
    x = np.array(x, dtype='float32') / 255  # Normalize to [0, 1]
    # x = (x - 0.5) / 0.5  # Optional: normalize to [-1, 1]
    x = x.reshape((-1,))  # Flatten to 1D
    x = torch.from_numpy(x)
    return x

# Function to inverse the normalization and reshape the image
def data_inv_transform(x):
    recover_data = x * 0.5 + 0.5  # Optional: inverse [-1, 1] normalization
    recover_data = recover_data * 255  # Denormalize to [0, 255]
    recover_data = recover_data.reshape((28, 28))  # Reshape to original image shape
    recover_data = recover_data.detach().cpu().numpy()
    return recover_data

# Load MNIST dataset with transformation
trainset = mnist.MNIST('./datasets/mnist', train=True, transform=data_transform, download=True)
testset = mnist.MNIST('./datasets/mnist', train=False, transform=data_transform, download=True)
train_data = DataLoader(trainset, batch_size=64, shuffle=True)
test_data = DataLoader(testset, batch_size=128, shuffle=False)

# Generate 10 random SNR values between 0 and 20 dB
random.seed(42)  # For reproducibility
snr_values = [random.uniform(0, 20) for _ in range(10)]
print("SNR values to test:", snr_values)

# Outer loop for different SNR values
for snr_idx, snr in enumerate(snr_values):
    print(f"\n=== Testing SNR: {snr:.2f} dB ===")
    
    # Loop through different compression rates
    for rate in range(10):
        compression_rate = min((rate + 1) * 0.1, 1)
        channel = int(compression_rate * raw_dim)  # Size of compressed representation

        # Coefficients for loss balancing
        lambda1 = 0.4 - compression_rate * 0.2
        lambda2 = 0.6 + compression_rate * 0.2

        # Define the Encoder-Decoder network
        class MLP(nn.Module):
            def __init__(self):
                super(MLP, self).__init__()
                self.fc1_1 = nn.Linear(28 * 28, 1024)  # Encoder layer 1
                self.fc1_2 = nn.Linear(1024, channel)  # Encoder layer 2 (compressed)
                self.fc2_1 = nn.Linear(channel, 1024)  # Decoder layer 1
                self.fc2_2 = nn.Linear(1024, 28 * 28)  # Decoder layer 2

            def forward(self, x):
                x = x.view(-1, 28 * 28)
                x = F.relu(self.fc1_1(x))  # Encoder forward pass
                x = F.relu(self.fc1_2(x))

                # Add Gaussian noise based on SNR
                x_np = x.detach().cpu().numpy()
                out_square = np.square(x_np)
                aver = np.sum(out_square) / np.size(out_square)
                aver_noise = aver / 10 ** (snr / 10)
                noise = np.random.random(size=x_np.shape) * np.sqrt(aver_noise)
                noise = torch.from_numpy(noise).cuda().to(torch.float32)

                x = torch.add(x, noise)  # Add noise to latent vector

                # Decoder forward pass
                x = F.relu(self.fc2_1(x))
                x = F.relu(self.fc2_2(x))
                return x

        # Define the classifier network for MNIST
        class MLP_MNIST(nn.Module):
            def __init__(self):
                super(MLP_MNIST, self).__init__()
                self.fc1 = nn.Linear(28 * 28, 500)
                self.fc2 = nn.Linear(500, 250)
                self.fc3 = nn.Linear(250, 125)
                self.fc4 = nn.Linear(125, 10)

            def forward(self, x):
                x = x.view(-1, 28 * 28)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = self.fc4(x)
                return x

        # Initialize encoder and classifier
        mlp_encoder = MLP().cuda()
        mlp_mnist = MLP_MNIST().cuda()

        # Loss function combining MSE and classification loss
        def criterion(x_in, y_in, raw_in):
            out_tmp1 = nn.CrossEntropyLoss()
            out_tmp2 = nn.MSELoss()
            z_in = mlp_mnist(x_in)
            mse_in = out_tmp2(x_in, raw_in)
            loss_channel = lambda2 * mse_in  # Only use reconstruction loss
            return loss_channel

        # Lists for recording performance
        losses = []
        acces = []
        eval_losses = []
        eval_acces = []
        psnr_all = []
        psnr = None

        # Load pretrained classifier
        mlp_mnist.load_state_dict(torch.load('saved_model/MLP_MNIST.pkl'))

        # Load encoder if exists
        model_path = ('saved_model/MLP_MNIST_coder_%.6f_snr_%.2f.pkl' % (compression_rate, snr))
        pre_model_exist = os.path.isfile(model_path)

        if pre_model_exist:
            print('load model parameters ...')
            mlp_encoder.load_state_dict(torch.load(model_path))
        else:
            print('No Well-Trained Model!')

        model_dict = mlp_mnist.state_dict()

        print('Training Start')
        print('Compression Rate:', compression_rate)
        print('SNR:', snr)
        epoch_len = 150
        out = None

        # Start training loop
        for e in range(epoch_len):
            torch.cuda.empty_cache()

            # Adjust optimizer based on epoch range
            if epoch_len < 80:
                optimizer = torch.optim.Adam(mlp_encoder.parameters(), 1e-3)
            elif epoch_len < 120:
                optimizer = torch.optim.Adam(mlp_encoder.parameters(), 1e-4)
            else:
                optimizer = torch.optim.Adam(mlp_encoder.parameters(), 1e-5)

            train_loss = 0
            train_acc = 0
            psnr_aver = 0
            counter = 0
            mlp_encoder.train()

            # Training pass
            for im, label in train_data:
                counter += 1
                im = im.cuda()
                label = label.cuda()

                # Adaptive transmission based on SNR
                if snr < 10:
                    # Use neural network for SNR < 10
                    out = mlp_encoder(im)
                else:
                    # Copy raw image for SNR >= 10
                    out = im.clone()
                    out_np = out.detach().cpu().numpy()
                    out_square = np.square(out_np)
                    aver = np.sum(out_square) / np.size(out_square)
                    aver_noise = aver / 10 ** (snr / 10)
                    noise = np.random.random(size=out_np.shape) * np.sqrt(aver_noise)
                    noise = torch.from_numpy(noise).cuda().to(torch.float32)
                    # Add Gaussian noise based on SNR
                    out = torch.add(out, noise) 


                out_mnist = mlp_mnist(out)

                loss = criterion(out, label, im)
                cr1 = nn.MSELoss()
                mse = cr1(out, im)
                out_np = out.detach().cpu().numpy()
                psnr = 10 * (np.log(1. / mse.item()) / np.log(10))  # Compute PSNR
                psnr_aver += psnr

                # Only update encoder if using neural network
                if snr < 10:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()

                _, pred = out_mnist.max(1)
                num_correct = (pred == label).sum().item()
                acc = num_correct / im.shape[0]
                train_acc += acc

            losses.append(train_loss / counter)
            acces.append(train_acc / counter)
            psnr_all.append(psnr_aver / counter)

            # Evaluation on test set
            eval_loss = 0
            eval_acc = 0
            counter = 0
            mlp_encoder.eval()
            for im, label in test_data:
                counter += 1
                im = im.cuda()
                label = label.cuda()

                # Adaptive transmission based on SNR
                if snr < 10:
                    # Use neural network for SNR < 10
                    out = mlp_encoder(im)
                else:
                    # Copy raw image for SNR >= 10
                    out = im.clone()
                    out_np = out.detach().cpu().numpy()
                    out_square = np.square(out_np)
                    aver = np.sum(out_square) / np.size(out_square)
                    aver_noise = aver / 10 ** (snr / 10)
                    noise = np.random.random(size=out_np.shape) * np.sqrt(aver_noise)
                    noise = torch.from_numpy(noise).cuda().to(torch.float32)
                    # Add Gaussian noise based on SNR
                    out = torch.add(out, noise)

                out_mnist = mlp_mnist(out)

                loss = criterion(out, label, im)
                eval_loss += loss.item()

                _, pred = out_mnist.max(1)
                num_correct = (pred == label).sum().item()
                acc = num_correct / im.shape[0]
                eval_acc += acc

            eval_losses.append(eval_loss / counter)
            eval_acces.append(eval_acc / counter)

            print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}, PSNR: {:.6f}'
                  .format(e, train_loss / len(train_data), train_acc / len(train_data),
                          eval_loss / len(test_data), eval_acc / len(test_data), psnr))

            # Save encoder model (only if using neural network)
            if snr < 10:
                torch.save(mlp_encoder.state_dict(), ('saved_model/MLP_MNIST_coder_%f_snr_%.2f.pkl' % (compression_rate, snr)))

            # Save metrics to CSV with SNR in filename
            file = ('./results/MLP_sem_MNIST/loss_semantic_combining_%f_snr_%.2f.csv' % (compression_rate, snr))
            data = pd.DataFrame(eval_losses)
            data.to_csv(file, index=False)

            file = ('./results/MLP_sem_MNIST/acc_semantic_combining_%f_snr_%.2f.csv' % (compression_rate, snr))
            data = pd.DataFrame(eval_acces)
            data.to_csv(file, index=False)

            eval_psnr = np.array(psnr_all)
            file = ('./results/MLP_sem_MNIST/psnr_semantic_combining_%f_snr_%.2f.csv' % (compression_rate, snr))
            data = pd.DataFrame(eval_psnr)
            data.to_csv(file, index=False)

            # Optional: save reconstructed images (only if using neural network)
            if snr < 10 and out is not None:
                for ii in range(min(len(out), 10)):  # Save only first 10 images to avoid too many files
                    image_recover = data_inv_transform(out[ii])
                    pil_img = Image.fromarray(np.uint8(image_recover))
                    pil_img.save("image_recover_combing/mnist/mnist_train_%d_%f_snr_%.2f.jpg" % (ii, compression_rate, snr))

print("All experiments completed!")
