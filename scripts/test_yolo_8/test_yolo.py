# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:33:14 2023

@author: heubu
"""

#pip install ultralytics

from ultralytics import YOLO
import os
from PIL import Image
import argparse

model_name='yolov8n.pt'

def load_model(model_name):
    model = YOLO(model_name)
    return model

def process_images(image_dir, model):
    # Load files from the directory
    image_files = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
    print("Number of images: " + str(len(image_files)))

    # Run batched inference on a list of images
    results = model(image_files)  # return a list of Results objects
    return results

def save_results(results, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the results
    for i, r in enumerate(results):
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im_file = os.path.join(output_dir, f'result_{i}.jpg')
        im.save(im_file)
        print("saved results")

def main(args):
    # Load a model
    model = load_model(args.model_name)
    print("Model loaded successfully")

    # Process images
    results = process_images(args.input_dir, model)
    print("Image processing and inference completed. Number of results:", len(results))

    # Save results
    save_results(results, args.output_dir)
    print("Results saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection using YOLOv8")
    parser.add_argument("--input_dir", required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", required=True, help="Directory for saving results")
    parser.add_argument("--model_name", required=False, default='yolov8n.pt', help="Path to pretrained YOLOv8n model")
    
    args = parser.parse_args()
    main(args)
