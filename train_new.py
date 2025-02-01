#!pip install tensorflow==2.12.0

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import os
import glob
# from fuzzywuzzy import process

from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
import os
import cv2
import json
import numpy as np
from skimage.draw import polygon
from skimage import io
import glob
from tqdm import tqdm
import albumentations as A
import matplotlib.pyplot as plt

# Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# Import TensorFlow and other necessary libraries
import tensorflow as tf
import os
import cv2

# Convert JSON label to image mask
def fill_mask(blobs, mask, label):
    for l in blobs:
        fill_row_coords, fill_col_coords = polygon(l[1], l[0], l[2])
        mask[fill_row_coords, fill_col_coords] = label

def convert_annotation_json_to_mask(path_to_annotation_json, path_to_masks_folder, classes, label_value, pixel_value):
    f = open(path_to_annotation_json)
    train = []
    objects = json.load(f)
    annotations = objects['shapes']
    h = objects['imageHeight']
    w = objects['imageWidth']
    mask = np.zeros((h, w)).astype('uint8')
    for annot in annotations:
        label = annot['label']
        points = annot['points']
        x_coord = []
        y_coord = []
        l = []
        for p in points:
            x_coord.append(int(p[0]))
            y_coord.append(int(p[1]))
        shape = (h, w)
        l.append(np.array(x_coord))
        l.append(np.array(y_coord))
        l.append(shape)
        if not label_value.get(label):
            label_value[label] = pixel_value[0]
            pixel_value[0] += 100
        if not classes.get(label):
            classes[label] = [l]
        else:
            classes[label].append(l)
    for label in classes:
        fill_mask(classes[label], mask, label_value[label])
        classes[label] = []
    io.imsave(path_to_masks_folder, mask)

def normalize_class_name(class_name):
    known_classes = {"deciduous": ["deciduous", "Deciduous", "decidious", "Decidious"],
                     "coniferous": ["coniferous", "Coniferous", "coniferus", "Coniferus"]}

    for correct_name, variations in known_classes.items():
        if class_name in variations:
            return correct_name

    return class_name  # Return as is if no match found

def get_classes(dataset_path):
    classes = dict()
    count = 0
    pixel_value = [100]
    label_value = dict()

    for dirpath, dirname, filename in os.walk(dataset_path):
        path_to_annotation_json = glob.glob(dirpath + "/*.json")
        for json_file in path_to_annotation_json:
            path_to_mask_png = json_file[0:-5] + "_mask.png"
            convert_annotation_json_to_mask(json_file, path_to_mask_png, classes, label_value, pixel_value)
            count += 1

    assert count, "Dataset folder path does not contain any json mask file"
    print(".png file of all json mask files saved in respective folders!")

    labels = [1] * len(label_value)
    for label, value in label_value.items():
        normalized_label = normalize_class_name(label)
        labels[(value - 100) // 100] = normalized_label

    class_list = ''
    for clss in labels:
        class_list += clss + "$"

    return class_list[:-1]

# Get classes
classes = get_classes(r'/content/train')
print('classes names: ', classes.split('$'))

# Data preprocessing
class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, dataset_path, n_classes, augmentation=None):
        self.images_fps = []
        self.mask_fps = []
        self.n_classes = n_classes
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            mask_src = glob.glob(dirpath + '/*.png')
            mask_src.sort()
            img_src = glob.glob(dirpath + '/*.JPG') + glob.glob(dirpath + '/*.jpg')
            img_src.sort()
            if len(img_src):
                assert len(mask_src) != 0, f"{dirpath} does not contain any annotation mask file for the images."
                if len(mask_src) != 1:
                    assert len(mask_src) == len(img_src), (
                        f"{dirpath} contains {len(mask_src)} mask files but {len(img_src)} images."
                    )
                    self.mask_fps += mask_src
                else:
                    self.mask_fps += [mask_src[0]] * len(img_src)
                self.images_fps += img_src
            else:
                if len(mask_src):
                    print(f"Skipping {dirpath} as it contains masks but no images.")
                continue  # Skip directories with no images or masks
        assert len(self.images_fps) != 0, f"{dataset_path} does not contain any images."
        self.actaul_size = len(self.mask_fps)
        self.augmentation = augmentation

    def __getitem__(self, i):
        j = i
        if i >= self.actaul_size:
            j = i - self.actaul_size
        image = cv2.imread(self.images_fps[j])
        h, w, _ = image.shape
        h = 256
        w = 256
        image = cv2.resize(image, (w, h))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32') / 255.0  # Convert to float32 and normalize
        mask1 = cv2.imread(self.mask_fps[j], 0)
        mask1 = cv2.resize(mask1, (w, h), interpolation=cv2.INTER_NEAREST)  # Resize without interpolation artifacts
        mask = [(mask1 == ((i + 1) * 100)) for i in range(self.n_classes)]
        mask = np.stack(mask, axis=-1)
        mask = mask.astype('float32')
        # add background to mask
        mask = np.zeros((h, w), dtype=np.uint8)  # Single-channel mask

        mask[mask1 == 100] = 1  # Deciduous
        mask[mask1 == 200] = 2  # Coniferous
        mask[mask1 == 0] = 0    # Background (sky)

        print("Unique values in mask:", np.unique(mask))


        # Convert to categorical one-hot encoding
        mask = np.eye(self.n_classes, dtype=np.float32)[mask]
        mask = 1.0 * (mask > 0.9)
        image = np.asarray(image, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        # apply augmentations
        if i >= self.actaul_size:
            sample = self.augmentation(image=image.astype('float32'), mask=mask.astype('float32'))
            image, mask = sample['image'], sample['mask']
            image = np.asarray(image, dtype=np.float32)
            mask = np.asarray(mask, dtype=np.float32)
        return (image, mask)

    def __len__(self):
        if self.augmentation:
            return self.actaul_size * 2
        return self.actaul_size

# DataLoader
class Dataloder(tf.keras.utils.Sequence):
    def __init__(self, indexes, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = indexes
        self.on_epoch_end()

    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[self.indexes[j]])
        # transpose list of lists
        batch = tuple(np.stack(samples, axis=0) for samples in zip(*data))
        return batch

    def __len__(self):
        """batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

# Data Augmentation
def aug():
    train_transform = [
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=5, shift_limit=0.1, p=0.9, border_mode=0),
        A.HorizontalFlip(p=0.5),
        A.OneOf(
            [
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)

# Function for visualizing multiple images together
def visualize(**images):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title("Image")

    # Display mask with correct colormap
    ax[1].imshow(mask, cmap="jet")  # "jet" gives different colors for classes
    ax[1].set_title("Mask")

    plt.show()

# Importing Dataset
dataset_path = r'/content/train'
dataset = CustomDataset(dataset_path, n_classes=3, augmentation=aug())

# Visualization of Data
for i in range(5):
    image, mask = dataset[i]
    mask = np.argmax(mask, axis=-1)
    visualize(image=image, mask=mask[..., 0].squeeze())

import torch.nn.functional as F





train_losses = []
iou_scores = []

def jaccard_index(preds, targets, num_classes=3):
    preds = preds.view(-1)
    targets = targets.view(-1)

    iou_per_class = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = targets == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou_per_class.append(torch.tensor(1.0))  # If no instances of the class, IoU = 1
        else:
            iou_per_class.append(intersection / union)

    return torch.mean(torch.stack(iou_per_class))

# Initialize dataset
dataset_path = '/content/train'
train_dataset = CustomDataset(dataset_path, n_classes=3, augmentation=aug())
test_dataset = CustomDataset(dataset_path, n_classes=3, augmentation=aug())

train_indexes = np.arange(len(train_dataset))  # Generate indexes
test_indexes = np.arange(len(test_dataset))

# Create dataloaders
train_dataloader = Dataloder(train_indexes, train_dataset, batch_size=8, shuffle=True)
test_dataloader = Dataloder(test_indexes, test_dataset, batch_size=8, shuffle=False)

# Check dataloader sizes
print(f"Train dataloader size: {len(train_dataloader)}")
print(f"Test dataloader size: {len(test_dataloader)}")

# Define the model using PyTorch
class SimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Fix: Final convolution layer should output 'n_classes' channels
        self.final_conv = nn.Conv2d(128, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.final_conv(x)  # Ensure final shape (batch_size, n_classes, H, W)
        return x

# Initialize model, loss function, and optimizer
model = SimpleCNN(n_classes=3)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 1.0]))
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 40
num_batches = len(train_dataloader)
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_dataloader:
        print("Image shape before model:", images.shape)  # Debugging
        images, masks = torch.tensor(images).to(device), torch.tensor(masks).to(device)

        # Ensure correct shape
        images = images.permute(0, 3, 1, 2)  # (batch_size, C, H, W)

        optimizer.zero_grad()
        outputs = model(images)

        masks = F.interpolate(masks.float().permute(0, 3, 1, 2), size=(32, 32), mode="nearest").squeeze(1)
        masks = torch.argmax(masks, dim=1).long()

        # Ensure values are within valid range
        masks = torch.clamp(masks, 0, 3 - 1)

        # Debugging
        print("Outputs shape:", outputs.shape)  # Expected (batch_size, n_classes, 32, 32)
        print("Target shape:", masks.shape)     # Expected (batch_size, 32, 32)
        print("Unique values in masks:", torch.unique(masks))  # Ensure only valid classes

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_dataloader)}")

        # Calculate IoU
        preds = torch.argmax(outputs, dim=1)
        iou = jaccard_index(preds, masks)  # Implement IoU function below
        total_iou = 0.0
        total_iou += iou

    # Store epoch loss and IoU
    avg_loss = running_loss / num_batches
    avg_iou = total_iou / num_batches
    train_losses.append(avg_loss)
    iou_scores.append(avg_iou)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in test_dataloader:
            images, masks = torch.tensor(images).to(device), torch.tensor(masks).to(device)
            images = images.permute(0, 3, 1, 2)
            outputs = model(images)
            masks = F.interpolate(masks.float().permute(0, 3, 1, 2), size=(32, 32), mode="nearest").squeeze(1)
            masks = masks.argmax(dim=1).long()

            # Ensure values are within valid range
            masks = torch.clamp(masks, 0, 2 - 1)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    print(f"Validation Loss: {val_loss/len(test_dataloader)}")

model_save_path = "/content/train/model.pth"  # Specify a file path with .pth extension
torch.save(model.state_dict(), model_save_path)
print(f"Model saved at {model_save_path}")
files.download(model_save_path)

import matplotlib.pyplot as plt
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, marker='o', label="Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()
plt.grid()

# Plot IoU Score
plt.subplot(1, 2, 2)
plt.plot(epochs, iou_scores, marker='s', label="IoU Score", color="blue")
plt.xlabel("Epochs")
plt.ylabel("IoU Score")
plt.title("IoU Score vs Epochs")
plt.legend()
plt.grid()

plt.show()

from matplotlib.colors import ListedColormap

custom_cmap = ListedColormap(["purple", "yellow", "green"])

predictions = torch.argmax(outputs, dim=1)  # Shape: (batch_size, H, W)

# Move tensors to CPU and convert to numpy
orig_img = images[0].cpu().permute(1, 2, 0).numpy()  # (H, W, C)
pred_img = predictions[0].cpu().numpy()  # (H, W)

# Plot the original image
plt.figure(figsize=(6, 6))
plt.imshow(orig_img)
plt.title("Original Image")
plt.axis("off")
plt.show()

# Plot the segmentation mask with yellow and purple
plt.figure(figsize=(6, 6))
plt.imshow(pred_img, cmap=custom_cmap)
plt.title("Predicted Segmentation Mask")
plt.axis("off")
plt.show()

for images, masks in train_dataloader:
        print("Image shape before model:", images.shape)  # Debugging
        images, masks = torch.tensor(images).to(device), torch.tensor(masks).to(device)

        # Ensure correct shape
        images = images.permute(0, 3, 1, 2)  # (batch_size, C, H, W)

        optimizer.zero_grad()
        outputs = model(images)

        masks = F.interpolate(masks.float().permute(0, 3, 1, 2), size=(32, 32), mode="nearest").squeeze(1)
        masks = masks.argmax(dim=1).long()

        # Ensure values are within valid range
        masks = torch.clamp(masks, 0, 2 - 1)

        # Debugging
        print("Outputs shape:", outputs.shape)  # Expected (batch_size, n_classes, 32, 32)
        print("Target shape:", masks.shape)     # Expected (batch_size, 32, 32)
        print("Unique values in masks:", torch.unique(masks))  # Ensure only valid classes

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_dataloader)}")

