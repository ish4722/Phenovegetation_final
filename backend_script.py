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
                    if file.lower().endswith('.jpg'):
                        file_path = os.path.join(dirpath, file)
                        image_dir.append(file_path)
            print(f"CHECKPOINT2: Found {len(image_dir)} image files")
            return image_dir
        except Exception as e:
            print(f"Error while traversing directories: {e}")
            raise

    image_dir = get_image_files(img_dir_path)
    print("CHECKPOINT3: Image files retrieved")

    for file_name in image_dir:
        try:
            # Extract date and time
            datetime_str = os.path.basename(file_name)[4:-11]  # Extracts the date (YYYY_MM_DD)
            time_str = os.path.basename(file_name)[-10:-4]     # Extracts the time (HHMMSS)

            # Convert date and time to datetime
            file_date = datetime.strptime(datetime_str, "%Y_%m_%d")
            file_time = datetime.strptime(time_str, "%H%M%S").time()  # Extract time part

            # Combine date and time into one datetime object
            combined_datetime = datetime.combine(file_date, file_time)

            # Apply the time filter
            if start_time <= file_time <= end_time:
                filtered_images.append((file_name,file_time))
        except ValueError:
            continue
    filtered_images.sort(key=lambda x: x[1])

    print(f"CHECKPOINT4: Filtered {len(filtered_images)} images by time")

    def filter_images_by_user_filters(filtered_images, filters):
        retained_images = []
        for image_path, file_time in filtered_images:
            image = cv2.imread(image_path)
            if image is None:
                print(f"CHECKPOINT4.1: Failed to read image {image_path}")
                continue
            remove = False

            if 'blurry' in filters:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                laplacian_var = laplacian.var()
                if laplacian_var < 1000:
                    remove = True

            if 'darkened' in filters:
                height = image.shape[0]
                bottom_third = image[int(height * (2 / 3)):, :]
                gray = cv2.cvtColor(bottom_third, cv2.COLOR_BGR2GRAY)
                mean_intensity = np.mean(gray)
                if mean_intensity < 30:
                    remove = True

            if 'snowy' in filters:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                lower_snow = np.array([0, 0, 180], dtype=np.uint8)
                upper_snow = np.array([180, 25, 255], dtype=np.uint8)
                snow_mask = cv2.inRange(hsv, lower_snow, upper_snow)
                snowy_percentage = np.sum(snow_mask == 255) / snow_mask.size
                if snowy_percentage > 0.0053:
                    remove = True

            if not remove:
                retained_images.append((image_path, file_time))

        print(f"CHECKPOINT5: Retained {len(retained_images)} images after user filters")
        return retained_images

    retained_images = filter_images_by_user_filters(filtered_images, filters)
    print("CHECKPOINT6: Filters applied, starting model loading")

    model = load_model(MODEL_PATH)
    print("CHECKPOINT7: Model loaded successfully")
    data = []

    for idx, (image_path, file_time) in enumerate(retained_images):
        print(f"CHECKPOINT8: Processing image {idx + 1}/{len(retained_images)}: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"CHECKPOINT8.1: Failed to read image {image_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0  # Normalize the image to [0, 1]

        # Reinitialize variables for each image
        brightness = darkness = contrast = grR = rbR = gbR = GRVI = exG = VCI = None
        red = green = blue = rcc = gcc = bcc = None

        # Calculate brightness, darkness, and contrast
        brightness = np.mean(image)
        darkness = np.min(image)
        contrast = np.max(image) - np.min(image)

        # Calculate ratios
        grR = np.mean(image[:, :, 1]) / np.mean(image[:, :, 0])
        rbR = np.mean(image[:, :, 0]) / np.mean(image[:, :, 2])
        gbR = np.mean(image[:, :, 1]) / np.mean(image[:, :, 2])

        # Vegetation indices
        GRVI = (np.mean(image[:, :, 1]) - np.mean(image[:, :, 0])) / (np.mean(image[:, :, 1]) + np.mean(image[:, :, 0]) + 1e-6)
        exG = 2 * np.mean(image[:, :, 1]) - np.mean(image[:, :, 0]) - np.mean(image[:, :, 2])
        VCI = (np.mean(image[:, :, 1]) - np.min(image[:, :, 1])) / (np.max(image[:, :, 1]) - np.min(image[:, :, 1]) + 1e-6)

        # Process image for the mask
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

        if len(points) < ROI_area:
            print(f"CHECKPOINT: Not enough points for clustering in {image_path}")
            continue

        n_ROIs = max(1, points.shape[0] // ROI_area)
        points = np.array(points, dtype=np.float32)

        k_means = KMeans(n_clusters=n_ROIs, max_iter=300, n_init=10)
        k_means.fit(points)
        labels = k_means.labels_

        ROIs = []
        for i in range(n_ROIs):
            roi_points = points[labels == i]
            if len(roi_points) < ROI_area * 0.7:
                continue
            ROIs.append(roi_points)

        print(f"CHECKPOINT10: Generated {len(ROIs)} ROIs for image {image_path}")

        if ROIs:
            for r in random.sample(ROIs, min(len(ROIs), ROI_TO_SAMPLE)):
                r = r.astype(int)
                red = np.mean(image[r[:, 0], r[:, 1], 0])
                green = np.mean(image[r[:, 0], r[:, 1], 1])
                blue = np.mean(image[r[:, 0], r[:, 1], 2])
                rcc = red / (red + green + blue)
                gcc = green / (red + green + blue)
                bcc = blue / (red + green + blue)
                rcc_std = np.std(red / (red + green + blue))
                gcc_std = np.std(green / (red + green + blue))
                bcc_std = np.std(blue / (red + green + blue))

        else:
            print(f"CHECKPOINT: No valid ROIs found in {image_path}")
            red = green = blue = rcc = gcc = bcc = None  # Default to None if no ROIs are found

        # Correctly calculate DoY and use image-specific datetime values
        
        file_date = combined_datetime.date()
        doy = combined_datetime.timetuple().tm_yday

        clean_file_name = image_path.replace("/content/08/", "")
        row = [
            clean_file_name,
            combined_datetime.strftime('%H:%M:%S'),
            file_date.strftime('%Y-%m-%d'),
            doy,
            red, green, blue,
            rcc, gcc, bcc,
            rcc_std, gcc_std, bcc_std,
            np.percentile(rcc, 5), np.percentile(gcc, 5), np.percentile(bcc, 5),
            np.percentile(rcc, 10), np.percentile(gcc, 10), np.percentile(bcc, 10),
            np.percentile(rcc, 25), np.percentile(gcc, 25), np.percentile(bcc, 25),
            np.percentile(rcc, 50), np.percentile(gcc, 50), np.percentile(bcc, 50),
            np.percentile(rcc, 75), np.percentile(gcc, 75), np.percentile(bcc, 75),
            np.percentile(rcc, 90), np.percentile(gcc, 90), np.percentile(bcc, 90),
            np.percentile(rcc, 95), np.percentile(gcc, 95), np.percentile(bcc, 95),
            brightness, darkness, contrast,
            grR, rbR, gbR, GRVI, exG, VCI
        ]
        data.append(row)

    print(f"CHECKPOINT11: Completed processing all images, saving data")
    columns = [
        'file', 'time', 'Date', 'DoY', 'red', 'green', 'blue', 'rcc', 'gcc', 'bcc',
        'rcc.std', 'gcc.std', 'bcc.std', 'rcc05', 'gcc05', 'bcc05', 'rcc10', 'gcc10',
        'bcc10', 'rcc25', 'gcc25', 'bcc25', 'rcc50', 'gcc50', 'bcc50', 'rcc75', 'gcc75',
        'bcc75', 'rcc90', 'gcc90', 'bcc90', 'rcc95', 'gcc95', 'bcc95', 'brightness',
        'darkness', 'contrast', 'grR', 'rbR', 'gbR', 'GRVI', 'exG', 'VCI'
    ]

    df = pd.DataFrame(data, columns=columns)

    df.to_excel("output.xlsx", index=False)

    print("CHECKPOINT12: Data saved to output.xlsx successfully")
