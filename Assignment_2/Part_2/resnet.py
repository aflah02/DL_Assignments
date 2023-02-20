import torch
import torch.nn as nn
import torch.nn.functional as F

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.conv1 = nn.LazyConv2d(out_channels=10, kernel_size=5)
        self.conv2 = nn.LazyConv2d(out_channels=20, kernel_size=2)

        self.downsample = nn.LazyConv2d(out_channels=20, kernel_size=7, stride=2, bias=False)

        self.fc1 = nn.LazyLinear(out_features=50)
        self.out = nn.LazyLinear(out_features=10)

    def forward(self, inputs):
        conv_output = self.conv1(inputs)
        pooled_output = F.max_pool2d(conv_output, kernel_size=2, stride=2)
        activation_output_1 = F.relu(pooled_output)

        conv_output = self.conv2(activation_output_1)

        conv_output += self.downsample(inputs)

        pooled_output = F.max_pool2d(conv_output, kernel_size=4, stride=2)
        activation_output_2 = F.relu(pooled_output)
        
        linear_input = activation_output_2.reshape(-1, 320)
        linear_output = self.fc1(linear_input)

        activation_output = F.relu(linear_output)

        logits = self.out(activation_output)
        softmax_outputs = F.softmax(logits, dim=1)

        return softmax_outputs

