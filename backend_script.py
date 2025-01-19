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

def main(img_dir_path, str_time, end_tim, filtrs, mask_name):

    # Define required variables
    start_time = str_time
    end_time = end_tim
    filters = filtrs
    filtered_images = []
    mask_need = mask_name  

    print("CHECKPOINT1: Initializing variables")

    def get_image_files(img_dir_path):
        image_dir = []
        try:
            for dirpath, _, filenames in os.walk(img_dir_path):
                for file in filenames:
                    if file.lower().endswith('.jpg'):  # Check for .jpg files
                        file_path = os.path.join(dirpath, file)
                        image_dir.append(file_path)
            print(f"CHECKPOINT2: Found {len(image_dir)} image files")
            return image_dir
        except Exception as e:
            print(f"Error while traversing directories: {e}")
            raise

    image_dir = get_image_files(img_dir_path)
    print("CHECKPOINT3: Image files retrieved")

    # Apply time filter logic
    for file_name in image_dir:
        try:
            datetime_str = os.path.basename(file_name)[4:-11]  # Remove the first 4 characters and '.JPG'
            file_time = datetime.strptime(datetime_str, "%Y_%m_%d")
    
    # Compare the extracted file time with the start and end time
            if start_time <= file_time <= end_time:
                filtered_images.append(file_name)
        except ValueError:
            continue   

    print(f"CHECKPOINT4: Filtered {len(filtered_images)} images by time")

    def filter_images_by_user_filters(filtered_images, filters):
        retained_images = []
        for image_path in filtered_images:
            image = cv2.imread(image_path)
            if image is None:
                print(f"CHECKPOINT4.1: Failed to read image {image_path}")
                continue
            remove = False

            # Detect blurred images
            if 'blurry' in filters:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                laplacian_var = laplacian.var()
                if laplacian_var < 1000:
                    remove = True

            # Detect darkened images
            if 'darkened' in filters:
                height = image.shape[0]
                bottom_third = image[int(height * (2 / 3)):, :]
                gray = cv2.cvtColor(bottom_third, cv2.COLOR_BGR2GRAY)
                mean_intensity = np.mean(gray)
                if mean_intensity < 30:
                    remove = True

            # Detect snowy images
            if 'snowy' in filters:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                lower_snow = np.array([0, 0, 180], dtype=np.uint8)
                upper_snow = np.array([180, 25, 255], dtype=np.uint8)
                snow_mask = cv2.inRange(hsv, lower_snow, upper_snow)
                snowy_percentage = np.sum(snow_mask == 255) / snow_mask.size
                if snowy_percentage > 0.0053:
                    remove = True

            if not remove:
                retained_images.append(image_path)

        print(f"CHECKPOINT5: Retained {len(retained_images)} images after user filters")
        return retained_images

    retained_images = filter_images_by_user_filters(filtered_images, filters)
    print("CHECKPOINT6: Filters applied, starting model loading")

    model = load_model(MODEL_PATH)
    print("CHECKPOINT7: Model loaded successfully")

    data = []

    for idx, image_path in enumerate(retained_images):
        print(f"CHECKPOINT8: Processing image {idx + 1}/{len(retained_images)}: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"CHECKPOINT8.1: Failed to read image {image_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(image, (256, 256))
        img = np.expand_dims(img, axis=0)

        mask = model.predict(img, verbose=0) > 0.5

        if mask_need == "coniferous":
            selected_mask = mask[..., 1].squeeze()
        elif mask_need == "deciduous":
            selected_mask = mask[..., 0].squeeze()
        else:
            raise ValueError(f"Invalid mask_name: {mask_need}")

        points = np.argwhere(selected_mask > 0.9)
        print(f"CHECKPOINT9: Found {len(points)} points in the selected mask")

        H, W, _ = image.shape
        scale_h = H / 255
        scale_w = W / 255
        threshold_area = 20000 * (H * W / 1200000)
        ROI_area = int(threshold_area / (scale_h * scale_w))
        n_ROIs = points.shape[0] // ROI_area

        k_means = KMeans(n_clusters=n_ROIs, max_iter=300)
        k_means.fit(points)
        labels = k_means.labels_

        ROIs = []
        for i in range(len(labels)):
            roi_points = points[labels == i]
            if len(roi_points) < ROI_area * 0.7:
                continue

            ROIs.append(roi_points)

        print(f"CHECKPOINT10: Generated {len(ROIs)} ROIs for image {image_path}")

        for r in random.sample(ROIs, min(len(ROIs), ROI_TO_SAMPLE)):
            red = np.mean(image[r[:, 0], r[:, 1], 0])
            green = np.mean(image[r[:, 0], r[:, 1], 1])
            blue = np.mean(image[r[:, 0], r[:, 1], 2])
            rcc = red / (red + green + blue)
            gcc = green / (red + green + blue)
            bcc = blue / (red + green + blue)

            row = ['file', 'time', 'Date', 'DoY', red, green, blue, rcc, gcc, bcc]
            data.append(row)

    print(f"CHECKPOINT11: Completed processing all images, saving data")

    df = pd.DataFrame(data, columns=[
        'file', 'time', 'Date', 'DoY', 'red', 'green', 'blue', 'rcc', 'gcc', 'bcc'])

    # df = pd.DataFrame(data, columns=[
    #     'file', 'time', 'Date', 'DoY', 'red', 'green', 'blue', 'rcc', 'gcc', 'bcc',
    #     'rcc.std', 'gcc.std', 'bcc.std', 'rcc05', 'gcc05', 'bcc05', 'rcc10', 'gcc10', 'bcc10',
    #     'rcc25', 'gcc25', 'bcc25', 'rcc50', 'gcc50', 'bcc50', 'rcc75', 'gcc75', 'bcc75', 
    #     'rcc90', 'gcc90', 'bcc90', 'rcc95', 'gcc95', 'bcc95', 'brightness', 'darkness', 
    #     'contrast', 'grR', 'rbR', 'gbR', 'GRVI', 'exG', 'VCI'
    # ])
    df.to_excel("output.xlsx", index=False)

    print("CHECKPOINT12: Data saved to output.xlsx successfully")
