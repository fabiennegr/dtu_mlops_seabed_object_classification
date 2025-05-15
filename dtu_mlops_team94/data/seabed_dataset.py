import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import transforms
import hydra
import os
import numpy as np
import matplotlib.pyplot as plt

class SeabedDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = os.path.join(hydra.utils.get_original_cwd(), data_dir)
        self.image_files = []
        self.labels = []
        self.class_mapping = {}
        self.id_to_class_mapping = {}
        self.num_classes = 0

        # Load the labels from the labels.csv file
        labels_file = os.path.join(self.data_dir, "labels/labels.csv")
        labels_df = pd.read_csv(labels_file)

        # Create a class mapping dictionary
        unique_classes = labels_df["seafloor"].unique()
        for i, class_name in enumerate(unique_classes):
            self.class_mapping[class_name] = i
            self.id_to_class_mapping[i] = class_name

        # Get the number of classes
        self.num_classes = len(unique_classes)

        # Get the list of images and their labels based on iteration of the labels.csv file
        for index, row in labels_df.iterrows():
            self.image_files.append(os.path.join(self.data_dir, "imgs", row["id"]))
            self.labels.append(self.class_mapping[row["seafloor"]])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the image
        image = Image.open(self.image_files[idx])

        # resize image to 224x224
        image = image.resize((224, 224))

        # Convert the image to a numpy array
        image = np.array(image)

        # Convert the single-channel image to 3-channel (for resnet)
        image = np.expand_dims(image, axis=0)
        image = np.repeat(image, 3, axis=0)

        # Convert the image to a torch tensor
        image = torch.from_numpy(image).float()

        # Normalize the image
        image = image / 255.0

        # # Transforms
        # image = transforms.ToTensor()(image)
        # image = transforms.Normalize([0.485, 0.456, 0.406],
        #                             [0.229, 0.224, 0.225])(image)

        # # Add a batch dimension
        # image = image.unsqueeze(0)
        # logger.info(image.shape)

        # Get the label
        label = self.labels[idx]

        return image, label


def preview_images(dataset, num_images=5):
    """Preview a few images from the dataset.

    Args:
        dataset: dataset object
        num_images: number of images to preview
    """
    # Get a random sample of images and targets
    indices = torch.randperm(len(dataset))[:num_images]
    images = [dataset[i][0] for i in indices]
    targets = [dataset[i][1] for i in indices]

    # Plot the images with their classes
    fig, axes = plt.subplots(1, num_images, figsize=(12, 4))
    for i, ax in enumerate(axes):
        img = images[i][0]  # use only the first channel (single-channel image)
        ax.imshow(img)
        ax.set_title(f"Class: {targets[i]} ({dataset.id_to_class_mapping[targets[i]]})")
        ax.axis("off")

    plt.show()