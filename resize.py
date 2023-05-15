import cv2
import numpy as np
import os

# Set the directory containing the images
image_directory = 'C:/Users/NS/Desktop/not_run/test'

# Set the target size for resizing
target_size = (100, 100)  # Adjust the size as per your requirements

# Set the directory to store resized images
output_directory = 'C:/Users/NS/Desktop/not_run/resized_images/brid/cat'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Iterate over the image files in the directory
for filename in os.listdir(image_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust the file extensions as per your image types
        # Read the image
        image_path = os.path.join(image_directory, filename)
        image = cv2.imread(image_path)

        # Resize the image to the target size
        resized_image = cv2.resize(image, target_size)

        # Save the resized image to the output directory
        output_path = os.path.join(output_directory, filename)
        cv2.imwrite(output_path, resized_image)

# Print a message upon completion
print('Resized images saved to:', output_directory)
