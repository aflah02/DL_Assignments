import torch

class PositionalAugmentation:
    def __init__(self) -> None:
        pass
    
    def augment_data(self, data, labels, augment_type):
        if augment_type == "left_flip":
            data = self.left_flip_data(data)
        elif augment_type == "rotate":
            data = self.rotate_data(data)
        elif augment_type == "gaussian_noise":
            data = self.gaussian_noise_data(data)
        else:
            print("Invalid Augmentation Type")
        return data, labels

    def left_flip_data(self, data):             # flips the image horizontally
        data = torch.flip(data, [2])
        return data

    def rotate_data(self, data):            # rotates the image by 90 degrees
        data = torch.rot90(data, 1, [1, 2])
        return data

    def gaussian_noise_data(self, data):            # adds gaussian noise of mean 0 and variance 0.1
        data = data + torch.randn(data.size()) * 0.1
        return data