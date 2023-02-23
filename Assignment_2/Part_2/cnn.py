import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.LazyConv2d(out_channels=10, kernel_size=5, padding=2)
        self.conv2 = nn.LazyConv2d(out_channels=20, kernel_size=2)

        self.fc1 = nn.LazyLinear(out_features=50)
        self.out = nn.LazyLinear(out_features=10)

    def forward(self, inputs):
        conv_output = self.conv1(inputs)    # 28 x 28
        # print(conv_output.shape)
        pooled_output = F.max_pool2d(conv_output, kernel_size=2, padding = 1, stride=1)    # 29 x 29
        # print(pooled_output.shape)
        activation_output_1 = F.relu(pooled_output)            # 29 x 29

        conv_output = self.conv2(activation_output_1)   # 28 x 28
        pooled_output_2 = F.max_pool2d(conv_output, kernel_size=2, padding = 1, stride=1)
        # print(conv_output.shape)

        pooled_output = F.max_pool2d(pooled_output_2, kernel_size=2, stride=2)

        activation_output_2 = F.relu(pooled_output)
        linear_input = activation_output_2.reshape(-1, activation_output_2.shape[-1]*activation_output_2.shape[-2]*activation_output_2.shape[-3])
        linear_output = self.fc1(linear_input)

        activation_output = F.relu(linear_output)

        logits = self.out(activation_output)
        softmax_outputs = F.softmax(logits, dim=1)

        return softmax_outputs

def visualize_data(batch):
    # Generate a 5x5 grid of images for 5 random images in the batch per digit class
    fig, ax = plt.subplots(5, 5, figsize=(10, 10))
    for i in range(5):
        for j in range(5):
            ax[i, j].imshow(batch[0][i*5+j][0].reshape(28, 28), cmap='gray')
            ax[i, j].axis('off')
    plt.show()
