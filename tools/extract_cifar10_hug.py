import os
from datasets import load_dataset
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()
os.makedirs(args.output, exist_ok=True)

# Define the path to the folder where the images will be saved
save_path = os.path.join(args.output, 'cifar10_images')
os.makedirs(save_path, exist_ok=True)

# Load CIFAR10 from HuggingFace
dataset = load_dataset("uoft-cs/cifar10", split="train")

# Loop through the dataset and save each image to the folder
for i, example in enumerate(tqdm(dataset)):
    image = example["img"]
    image_name = f'{i}.png'
    image_path = os.path.join(save_path, image_name)
    image.save(image_path)
