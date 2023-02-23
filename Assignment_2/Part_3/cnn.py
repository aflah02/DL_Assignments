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

        self.conv1 = nn.LazyConv2d(out_channels=10, kernel_size=5)
        self.conv2 = nn.LazyConv2d(out_channels=20, kernel_size=2)

        self.fc1 = nn.LazyLinear(out_features=50)
        self.out = nn.LazyLinear(out_features=10)

    def forward(self, inputs):
        conv_output = self.conv1(inputs)
        pooled_output = F.max_pool2d(conv_output, kernel_size=2, stride=2)
        activation_output = F.relu(pooled_output)

        conv_output = self.conv2(activation_output)
        pooled_output = F.max_pool2d(conv_output, kernel_size=4, stride=2)
        activation_output = F.relu(pooled_output)

        linear_input = activation_output.reshape(-1, 320)
        linear_output = self.fc1(linear_input)
        activation_output = F.relu(linear_output)

        logits = self.out(activation_output)
        softmax_outputs = F.softmax(logits, dim=1)

        return softmax_outputs

def visualize_data(batch):
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 3))
    fig.suptitle("Randomly selected images from the training set", fontsize=16)
    for ax in axes:
        index = random.randint(0, len(batch[0]))
        ax.imshow(batch[0][index].view(28, 28))
        ax.set_title("Label: " + str(batch[1][index].item()))
    plt.show()
