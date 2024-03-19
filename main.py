import os
import cv2

# Define paths
input_dir = "D:/ML Projects/Dataset/training"  # Original input directory path
output_dir = "D:/ML Projects/aug_train"  # New output directory path

# Define the rotation degree
rotation_degree = 15

# Function to perform rotation on an image
def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

# Function to perform data augmentation
def augment_data(input_dir, output_dir):
    total_images = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(input_path, input_dir)
            output_subdir = os.path.join(output_dir, os.path.dirname(relative_path))
            os.makedirs(output_subdir, exist_ok=True)

            img = cv2.imread(input_path)

            # Determine the class (healthy or parkinson's)
            class_name = os.path.basename(os.path.dirname(input_path))

            # Save original image
            output_path = os.path.join(output_subdir, f"original_{file}")
            cv2.imwrite(output_path, img)
            total_images += 1

            # Perform rotation
            for angle in range(0, 360, rotation_degree):
                rotated_img = rotate_image(img, angle)
                output_path = os.path.join(output_subdir, f"rotated_{angle}_{file}")
                cv2.imwrite(output_path, rotated_img)
                total_images += 1

            # Perform flipping
            flipped_img = cv2.flip(img, 1)
            output_path = os.path.join(output_subdir, f"flipped_{file}")
            cv2.imwrite(output_path, flipped_img)
            total_images += 1

    return total_images

# Perform data augmentation
total_images = augment_data(input_dir, output_dir)
print(f"Total images after augmentation: {total_images}")