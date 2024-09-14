import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import argparse

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help=r"C:\Users\saad_\OneDrive\Bureau\TruckClassification-v2\TruckClassification-v2\trainnig\dataset\chargement_citerne")
ap.add_argument("-o", "--output", required=True, help=r"C:\Users\saad_\OneDrive\Bureau\TruckClassification-v2\TruckClassification-v2\trainnig\dataset\chargement citerne")
args = vars(ap.parse_args())

# Get input and output paths from arguments
input_folder = args["input"]
output_folder = args["output"]

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get all files in the input folder
onlyfiles = [f for f in listdir(input_folder) if isfile(join(input_folder, f))]

# Iterate through every image and resize them to 512x512
for n, file in enumerate(onlyfiles):
    # Construct full input path
    input_path = join(input_folder, file)
    
    # Read the image
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"Error reading file {file}. Skipping.")
        continue

    # Define the new size
    resize_width = 224
    resize_height = 224
    resized_dimensions = (resize_width, resize_height)
    
    # Resize the image
    resized_image = cv2.resize(img, resized_dimensions, interpolation=cv2.INTER_AREA)
    
    # Construct output file path
    output_filename = f'truck_{n}_resized.jpg'
    output_path = join(output_folder, output_filename)

    # Save the resized image
    cv2.imwrite(output_path, resized_image)
    print(f"Resized image saved at: {output_path}")

print("All images resized successfully.")
