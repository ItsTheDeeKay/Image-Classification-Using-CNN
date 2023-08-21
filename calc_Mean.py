import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor

# Set your dataset path here
dataset_path = "./cnn_dataset/vehicles"

image_paths = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path)]

means = torch.zeros(3)
stds = torch.zeros(3)
n_pixels = 0

for img_path in image_paths:
    img = Image.open(img_path)
    img_tensor = ToTensor()(img)

    # Update the mean and standard deviation
    means += torch.mean(img_tensor, dim=[1, 2])
    stds += torch.std(img_tensor, dim=[1, 2])
    n_pixels += img_tensor.size(1) * img_tensor.size(2)

# Calculate the final mean and standard deviation
means /= len(image_paths)
stds /= len(image_paths)

print("Mean values: ", means)
print("Standard deviation values: ", stds)
