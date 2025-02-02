import os
from datetime import datetime
import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
import random
import pandas as pd
from config import *
from architecture import SimpleCNN

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

            if start_time <= file_time <= end_time:
                filtered_images.append((file_name, combined_datetime))  # Store full datetime
        except ValueError:
            continue
    filtered_images.sort(key=lambda x: x[1])  # Sort by full datetime

    print(f"CHECKPOINT4: Filtered {len(filtered_images)} images by time")

    def filter_images_by_user_filters(filtered_images, filters):
        retained_images = []
        for image_path, combined_datetime in filtered_images:
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
                
                # Get the dimensions of the image
                height, width = image.shape[:2]
                
                # Define the region of interest (ROI)
                # Bottom half of the image (height-wise)
                start_row = int(height / 2)  # Start from the middle of the image height
                end_row = height  # End at the bottom of the image
                
                # Right one-third of the image (width-wise)
                start_col = int(2 * width / 3)  # Start from 2/3 of the image width
                end_col = width  # End at the right edge of the image
                
                # Create a mask for the bottom-right corner
                mask_bottom_right = np.zeros((height, width), dtype=np.uint8)
                mask_bottom_right[start_row:end_row, start_col:end_col] = 255  # Set the ROI to white (255)
                
                # Apply the snow detection only to the bottom-right corner
                snow_mask = cv2.inRange(hsv, lower_snow, upper_snow)
                snow_mask = cv2.bitwise_and(snow_mask, mask_bottom_right)  # Apply the ROI mask
                
                # Calculate the percentage of snow in the bottom-right corner
                snowy_percentage = np.sum(snow_mask == 255) / np.sum(mask_bottom_right == 255)

                if snowy_percentage > 0.00093:
                    remove = True

            if not remove:
                retained_images.append((image_path, combined_datetime))

        print(f"CHECKPOINT5: Retained {len(retained_images)} images after user filters")
        return retained_images

    retained_images = filter_images_by_user_filters(filtered_images, filters)
    print("CHECKPOINT6: Filters applied, starting model loading")

    model = SimpleCNN()
    if(mask_need == "rice" or "wheat"):
        model.load_state_dict(torch.load(MODEL_PATH_crop, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(MODEL_PATH_forest, map_location=torch.device('cpu')))

    model.eval()

    print("Model loaded successfully!")
    data = []

    for idx, (image_path, combined_datetime) in enumerate(retained_images):
        print(f"CHECKPOINT8: Processing image {idx + 1}/{len(retained_images)}: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"CHECKPOINT8.1: Failed to read image {image_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0

        # Initialize variables
        brightness = np.mean(image)
        darkness = np.min(image)
        contrast = np.max(image) - np.min(image)

        grR = np.mean(image[:, :, 1]) / (np.mean(image[:, :, 0]) + 1e-10)
        rbR = np.mean(image[:, :, 0]) / (np.mean(image[:, :, 2]) + 1e-10)
        gbR = np.mean(image[:, :, 1]) / (np.mean(image[:, :, 2]) + 1e-10)

        GRVI = (np.mean(image[:, :, 1]) - np.mean(image[:, :, 0])) / (np.mean(image[:, :, 1]) + np.mean(image[:, :, 0]) + 1e-10)
        exG = 2 * np.mean(image[:, :, 1]) - np.mean(image[:, :, 0]) - np.mean(image[:, :, 2])
        VCI = (np.mean(image[:, :, 1]) - np.min(image[:, :, 1])) / (np.max(image[:, :, 1]) - np.min(image[:, :, 1]) + 1e-10)

        img = cv2.resize(image, (256, 256))
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = torch.tensor(img, dtype=torch.float32)

        with torch.no_grad():
            output = model(img)
            probs = torch.softmax(output, dim=1)
            mask = torch.argmax(probs, dim=1).squeeze(0)

        if mask_need == "coniferous":
            selected_mask = (mask == 1).numpy()
        elif mask_need == "deciduous":
            selected_mask = (mask == 0).numpy()
        else:
            raise ValueError(f"Invalid mask_name: {mask_need}")

        points = np.argwhere(selected_mask>0.5)
        print(f"CHECKPOINT9: Found {len(points)} points in the selected mask")

        H, W, _ = image.shape
        scale_h = H / 255
        scale_w = W / 255
        threshold_area = 2000 * (H * W / 1200000)
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

        # Initialize color metrics
        red = green = blue = rcc = gcc = bcc = None
        rcc_std = gcc_std = bcc_std = None
        rcc_percentiles = gcc_percentiles = bcc_percentiles = None

        if ROIs:
            # Process a random sample of ROIs
            sampled_ROIs = random.sample(ROIs, min(len(ROIs), ROI_TO_SAMPLE))
            
            # Lists to store values across all sampled ROIs
            all_rcc = []
            all_gcc = []
            all_bcc = []
            
            for r in sampled_ROIs:
                r = r.astype(int)
                
                # Get RGB values for all pixels in the ROI
                roi_red = image[r[:, 0], r[:, 1], 0]
                roi_green = image[r[:, 0], r[:, 1], 1]
                roi_blue = image[r[:, 0], r[:, 1], 2]
                
                # Calculate chromatic coordinates for each pixel
                total = roi_red + roi_green + roi_blue + 1e-10
                roi_rcc = roi_red / total
                roi_gcc = roi_green / total
                roi_bcc = roi_blue / total
                
                # Append values to lists
                all_rcc.extend(roi_rcc)
                all_gcc.extend(roi_gcc)
                all_bcc.extend(roi_bcc)
            
            # Convert lists to numpy arrays for calculations
            all_rcc = np.array(all_rcc)
            all_gcc = np.array(all_gcc)
            all_bcc = np.array(all_bcc)
            
            # Calculate means
            red = np.mean(roi_red)
            green = np.mean(roi_green)
            blue = np.mean(roi_blue)
            rcc = np.mean(all_rcc)
            gcc = np.mean(all_gcc)
            bcc = np.mean(all_bcc)
            
            # Calculate standard deviations
            rcc_std = np.std(all_rcc)
            gcc_std = np.std(all_gcc)
            bcc_std = np.std(all_bcc)
            
            # Calculate percentiles
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            rcc_percentiles = np.percentile(all_rcc, percentiles)
            gcc_percentiles = np.percentile(all_gcc, percentiles)
            bcc_percentiles = np.percentile(all_bcc, percentiles)

        else:
            print(f"CHECKPOINT: No valid ROIs found in {image_path}")
            # All metrics remain None

        file_date = combined_datetime.date()
        file_time = combined_datetime.time()
        doy = combined_datetime.timetuple().tm_yday
        
        clean_file_name = os.path.basename(image_path)

        row = [
            clean_file_name,
            file_time.strftime('%H:%M:%S'),
            file_date.strftime('%Y-%m-%d'),
            doy,
            red, green, blue,
            rcc, gcc, bcc,
            rcc_std, gcc_std, bcc_std
        ]
        
        # Add percentiles if they exist
        if rcc_percentiles is not None:
            row.extend([
                rcc_percentiles[0], gcc_percentiles[0], bcc_percentiles[0],  # 5th
                rcc_percentiles[1], gcc_percentiles[1], bcc_percentiles[1],  # 10th
                rcc_percentiles[2], gcc_percentiles[2], bcc_percentiles[2],  # 25th
                rcc_percentiles[3], gcc_percentiles[3], bcc_percentiles[3],  # 50th
                rcc_percentiles[4], gcc_percentiles[4], bcc_percentiles[4],  # 75th
                rcc_percentiles[5], gcc_percentiles[5], bcc_percentiles[5],  # 90th
                rcc_percentiles[6], gcc_percentiles[6], bcc_percentiles[6]   # 95th
            ])
        else:
            row.extend([None] * 21)  # Add None for all percentiles
        
        # Add remaining metrics
        row.extend([
            brightness, darkness, contrast,
            grR, rbR, gbR, GRVI, exG, VCI
        ])
        
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
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['time'])
    df = df.sort_values('datetime')
    df = df.drop('datetime', axis=1)
    df.to_excel("output.xlsx", index=False)

    print("CHECKPOINT12: Data saved to output.xlsx successfully")
