import os
from datetime import datetime
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.cluster import KMeans
import random
import pandas as pd
from config import *
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


# Define and register the FixedDropout layer
@register_keras_serializable()
class FixedDropout(Layer):
    def __init__(self, rate, seed=None, noise_shape=None, **kwargs):
        super(FixedDropout, self).__init__(**kwargs)
        self.rate = rate
        self.seed = seed
        self.noise_shape = noise_shape

    def build(self, input_shape):
        super(FixedDropout, self).build(input_shape)

    def call(self, inputs, training=None):
        if training:
            return K.dropout(inputs, self.rate, seed=self.seed, noise_shape=self.noise_shape)
        return inputs

# Register custom activation function
@register_keras_serializable()
def swish(x):
    return x * K.sigmoid(x)

def main(img_dir_path,str_time,end_tim,filtrs,mask_name):

    # Define required variables
    # image_dir = img_dir_path  
    start_time = str_time
    end_time = end_tim
    filters = filtrs
    filtered_images = []
    mask_need=mask_name  

    def get_image_files(img_dir_path):
      image_dir = []
      try:
          for dirpath, _, filenames in os.walk(img_dir_path):
            for file in filenames:
                if file.lower().endswith('.jpg'):  # Check for .jpg files
                    file_path = os.path.join(dirpath, file)
                    image_dir.append(file_path)
          return image_dir
      except Exception as e:
        print(f"Error while traversing directories: {e}")
        raise
      
    image_dir = get_image_files(img_dir_path)

    # Apply time filter logic
    for file_name in os.listdir(image_dir):
        try:
            # Extract time from the file name
            file_time = datetime.strptime(file_name[4:], "%Y_%m_%d_%H%M%S.jpg")
            if start_time <= file_time <= end_time:
                filtered_images.append(os.path.join(image_dir, file_name))
        except ValueError:
            continue

    def filter_images_by_user_filters(filtered_images, filters):

        retained_images = []

        for image_path in filtered_images:
            image = cv2.imread(image_path)
            remove = False

            # Detect blurred images
            if 'blurry' in filters:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                laplacian_var = laplacian.var()
                is_blurry = laplacian_var < 1000
                if is_blurry:
                    remove = True

            # Detect darkened images
            if 'darkened' in filters:
                height = image.shape[0]
                bottom_third = image[int(height * (2 / 3)):, :]
                gray = cv2.cvtColor(bottom_third, cv2.COLOR_BGR2GRAY)
                mean_intensity = np.mean(gray)
                std_intensity = np.std(gray)
                is_darkened = mean_intensity < 30 and std_intensity < 15
                if is_darkened:
                    remove = True

            # Detect snowy images
            if 'snowy' in filters:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                lower_snow = np.array([0, 0, 180], dtype=np.uint8)
                upper_snow = np.array([180, 25, 255], dtype=np.uint8)
                snow_mask = cv2.inRange(hsv, lower_snow, upper_snow)
                snowy_percentage = np.sum(snow_mask == 255) / snow_mask.size
                is_snowy = snowy_percentage > 0.0053
                if is_snowy:
                    remove = True

            # Retain the image if it passed all specified filters
            if not remove:
                retained_images.append(image_path)

        return retained_images

    # Apply user-specified filters
    retained_images = filter_images_by_user_filters(filtered_images, filters)


# Load model with the registered swish and FixedDropout
    model_path = MODEL_PATH

    model = load_model(model_path, custom_objects={'swish': swish, 'FixedDropout': FixedDropout})


    # selected_masks = []  # List to store the selected masks based on mask_name

    for image_path in retained_images:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize and prepare the image for prediction
        img = cv2.resize(image, (256, 256))
        img = np.expand_dims(img, axis=0)

        # Predict the masks
        mask = model.predict(img, verbose=0) > 0.5

        # Select the mask based on mask_name
        if mask_need == "coniferous":
            selected_mask = mask[..., 1].squeeze()  # Coniferous tree mask
        elif mask_need == "deciduous":
            selected_mask = mask[..., 0].squeeze()  # Deciduous tree mask
        else:
            raise ValueError(f"Invalid mask_name: {mask_need}. Must be 'coniferous' or 'deciduous'.")

        
        points = []

        for i in range(255):
            for j in range(255):
                if selected_mask[i, j] > 0.9:  # You can change this condition to deciduous_mask for other mask
                    points.append([i, j])
        points = np.array(points)

        # Scale factor and threshold area calculation
        H, W, _ = image.shape
        scale_h = H / 255
        scale_w = W / 255
        threshold_area = 20000 * (H * W / 1200000)
        ROI_area = int(threshold_area / (scale_h * scale_w))
        n_ROIs = points.shape[0] // ROI_area

        k_means = KMeans(n_clusters=n_ROIs, max_iter=300)
        k_means.fit(points)
        # centroids = k_means.cluster_centers_
        labels = k_means.labels_

        clustering_image = np.zeros((256, 256))
        for i in range(n_ROIs):
            p = points[labels == i]
            clustering_image[p[:, 0], p[:, 1]] = 255 * (i + 1) // (n_ROIs)

        ROIs = []
        for i in range(len(labels)):
            if len(points[labels == i]) < ROI_area * 0.7:
                continue
            x1 = min(points[labels == i][:, 1])
            y1 = min(points[labels == i][:, 0])
            x2 = max(points[labels == i][:, 1])
            y2 = max(points[labels == i][:, 0])

            ROI_IMG = np.zeros((y2 - y1 + 1, x2 - x1 + 1))
            ROI_IMG[points[labels == i][:, 0] - y1, points[labels == i][:, 1] - x1] = 1

            h, w = ROI_IMG.shape
            h, w = int(scale_h * h), int(scale_w * w)
            ROI_image = cv2.resize(ROI_IMG, (w, h))
            ROI = []
            for i in range(h):
                for j in range(w):
                    if ROI_image[i, j] > 0.5:
                        ROI.append([i, j])
            ROI = np.array(ROI)
            ROI[:, 0] += int(y1 * scale_h)
            ROI[:, 1] += int(x1 * scale_w)

            ROIs.append(ROI)

        # Randomly sample 4 ROIs
        rois = random.sample(ROIs,ROI_TO_SAMPLE)
        ROI_image = image.copy()

        # Create a list to store the rows of data for the Excel file
        data = []

        for r in rois:
            ROI_image[(r[:, 0]), (r[:, 1]), 1] += 100
            ROI_image[(r[:, 0]), (r[:, 1]), 0] = 0
            ROI_image[(r[:, 0]), (r[:, 1]), 2] = 0

            # Calculate the required indices for the given ROI
            red = np.mean(ROI_image[r[:, 0], r[:, 1], 0])
            green = np.mean(ROI_image[r[:, 0], r[:, 1], 1])
            blue = np.mean(ROI_image[r[:, 0], r[:, 1], 2])

            # Calculate the rcc, gcc, bcc, and percentiles
            rcc = red / (red + green + blue)
            gcc = green / (red + green + blue)
            bcc = blue / (red + green + blue)

            # Calculate brightness, darkness, contrast
            brightness = (np.max(ROI_image[r[:, 0], r[:, 1], :]) + np.min(ROI_image[r[:, 0], r[:, 1], :])) / 2
            darkness = np.min(ROI_image[r[:, 0], r[:, 1], :])
            contrast = np.max(ROI_image[r[:, 0], r[:, 1], :]) - np.min(ROI_image[r[:, 0], r[:, 1], :])

            # Calculate GRVI (Green-Red Vegetation Index)
            grR = (green - red) / (green + red)

            # Calculate RGB ratios
            rbR = red / green
            gbR = green / blue

            # Create the row of data
            row = [
                'file',  # Placeholder for file name or time
                'time',  # Placeholder for timestamp
                'Date',  # Placeholder for Date
                'DoY',  # Placeholder for Day of Year
                red, green, blue, rcc, gcc, bcc,
                np.std(rcc), np.std(gcc), np.std(bcc),
                np.percentile(rcc, 5), np.percentile(gcc, 5), np.percentile(bcc, 5),
                np.percentile(rcc, 10), np.percentile(gcc, 10), np.percentile(bcc, 10),
                np.percentile(rcc, 25), np.percentile(gcc, 25), np.percentile(bcc, 25),
                np.percentile(rcc, 50), np.percentile(gcc, 50), np.percentile(bcc, 50),
                np.percentile(rcc, 75), np.percentile(gcc, 75), np.percentile(bcc, 75),
                np.percentile(rcc, 90), np.percentile(gcc, 90), np.percentile(bcc, 90),
                np.percentile(rcc, 95), np.percentile(gcc, 95), np.percentile(bcc, 95),
                brightness, darkness, contrast, grR, rbR, gbR, grR, 0, 0  # VCI not calculated
            ]
            data.append(row)

        # Convert the list of rows into a DataFrame
    df = pd.DataFrame(data, columns=[
            'file', 'time', 'Date', 'DoY', 'red', 'green', 'blue', 'rcc', 'gcc', 'bcc',
            'rcc.std', 'gcc.std', 'bcc.std', 'rcc05', 'gcc05', 'bcc05', 'rcc10', 'gcc10', 'bcc10',
            'rcc25', 'gcc25', 'bcc25', 'rcc50', 'gcc50', 'bcc50', 'rcc75', 'gcc75', 'bcc75', 
            'rcc90', 'gcc90', 'bcc90', 'rcc95', 'gcc95', 'bcc95', 'brightness', 'darkness', 
            'contrast', 'grR', 'rbR', 'gbR', 'GRVI', 'exG', 'VCI'
    ])

    # Save the data to an Excel file
    df.to_excel("output.xlsx", index=False)


