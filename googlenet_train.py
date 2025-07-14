# encoding: utf-8
# Train and save a CIFAR-10 classifier using a simplified GoogLeNet-like architecture

import sys
import os

sys.path.append("...")  # Extend path for external module imports if needed

import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torchvision.datasets import CIFAR10

# Define a helper function to create a convolutional layer with ReLU and BatchNorm
def conv_relu(in_channels, out_channels, kernel, stride=1, padding=0):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
        nn.BatchNorm2d(out_channels, eps=1e-3),
        nn.ReLU(True)
    )
    return layer

# Define the Inception module
class inception(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
        super(inception, self).__init__()
        # First branch: 1x1 conv
        self.branch1x1 = conv_relu(in_channel, out1_1, 1)

        # Second branch: 1x1 conv followed by 3x3 conv
        self.branch3x3 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1)
        )

        # Third branch: 1x1 conv followed by 5x5 conv
        self.branch5x5 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2)
        )

        # Fourth branch: 3x3 max pool followed by 1x1 conv
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 1)
        )

    def forward(self, x):
        # Compute outputs of all branches and concatenate them along the channel dimension
        f1 = self.branch1x1(x)
        f2 = self.branch3x3(x)
        f3 = self.branch5x5(x)
        f4 = self.branch_pool(x)
        output = torch.cat((f1, f2, f3, f4), dim=1)
        return output

# Test the inception module
test_net = inception(3, 64, 48, 64, 64, 96, 32)
test_x = Variable(torch.zeros(1, 3, 96, 96))
print('input shape: {} x {} x {}'.format(test_x.shape[1], test_x.shape[2], test_x.shape[3]))
test_y = test_net(test_x)
print('output shape: {} x {} x {}'.format(test_y.shape[1], test_y.shape[2], test_y.shape[3]))

# Define the full GoogLeNet-like architecture
class googlenet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(googlenet, self).__init__()
        self.verbose = verbose

        # Initial conv and pool layers
        self.block1 = nn.Sequential(
            conv_relu(in_channel, out_channels=64, kernel=7, stride=2, padding=3),
            nn.MaxPool2d(3, 2)
        )

        # Second block of conv layers
        self.block2 = nn.Sequential(
            conv_relu(64, 64, kernel=1),
            conv_relu(64, 192, kernel=3, padding=1),
            nn.MaxPool2d(3, 2)
        )

        # Third block: two inception modules + max pooling
        self.block3 = nn.Sequential(
            inception(192, 64, 96, 128, 16, 32, 32),
            inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2)
        )

        # Fourth block: multiple inception modules + max pooling
        self.block4 = nn.Sequential(
            inception(480, 192, 96, 208, 16, 48, 64),
            inception(512, 160, 112, 224, 24, 64, 64),
            inception(512, 128, 128, 256, 24, 64, 64),
            inception(512, 112, 144, 288, 32, 64, 64),
            inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2)
        )

        # Fifth block: two inception modules + average pooling
        self.block5 = nn.Sequential(
            inception(832, 256, 160, 320, 32, 128, 128),
            inception(832, 384, 182, 384, 48, 128, 128),
            nn.AvgPool2d(2)
        )

        # Final classifier
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Forward pass through all blocks
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

        # Flatten and apply classifier
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

# Test the full GoogLeNet network
test_net = googlenet(3, 10, True)
test_x = Variable(torch.zeros(1, 3, 96, 96))
test_y = test_net(test_x)
print('output: {}'.format(test_y.shape))

# Data transformation function for CIFAR-10 images
def data_tf(x):
    x = x.resize((96, 96), 2)  # Resize image to 96x96
    x = np.array(x, dtype='float32') / 255  # Normalize pixel values
    # x = (x - 0.5) / 0.5  # Optional: standardize to [-1, 1]
    x = x.transpose((2, 0, 1))  # Convert to channel-first format
    x = torch.from_numpy(x)  # Convert to tensor
    return x

# Load CIFAR-10 dataset
train_set = CIFAR10('./datasets/cifar10', train=True, transform=data_tf, download=True)
train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_set = CIFAR10('./datasets/cifar10', train=False, transform=data_tf, download=True)
test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

# Instantiate GoogLeNet model and loss function
net = googlenet(3, 10)
criterion = nn.CrossEntropyLoss()

model_path = ("./saved_model/google_net.pkl")
pre_model_exist = os.path.isfile(model_path)

if pre_model_exist:
    print('load model parameters ...')
    net.load_state_dict(torch.load(model_path))
else:
    print('No Well-Trained Model!')

from datetime import datetime

# Accuracy calculation helper
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)  # Get predicted class
    num_correct = (pred_label == label).sum().item()
    return num_correct / total

# Training function
def train(net, train_data, valid_data, num_epochs, criterion):
    if torch.cuda.is_available():
        net = net.cuda()  # Move model to GPU if available
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        # Adjust learning rate based on epoch
        if epoch < 20:
            optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        elif epoch < 25:
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            if torch.cuda.is_available():
                im = Variable(im.cuda())
                label = Variable(label.cuda())
            else:
                im = Variable(im)
                label = Variable(label)

            # Forward pass
            output = net(im)
            loss = criterion(output, label)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += get_acc(output, label)

        # Time calculation
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)

        # Validation evaluation
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = Variable(im.cuda(), volatile=True)
                    label = Variable(label.cuda(), volatile=True)
                else:
                    im = Variable(im, volatile=True)
                    label = Variable(label, volatile=True)
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)

            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))

        prev_time = cur_time
        print(epoch_str + time_str)

        # Save model checkpoint
        torch.save(net.state_dict(), 'saved_model/google_net.pkl')

# Train the model for 30 epochs
train(net, train_data, test_data, 30, criterion)
