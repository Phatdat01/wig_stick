from flask import Flask, request, jsonify, Response, render_template

import io
import os
import torch
import subprocess
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from torchvision.utils import save_image
from werkzeug.utils import secure_filename

from hair_swap import HairFast, get_parser

model_args = get_parser()
hair_fast = HairFast(model_args.parse_args([]))
# model_parser = get_parser()
# model_args, _ = model_parser.parse_known_args()
# hair_fast = HairFast(model_args)

def resize_image(image, target_size=(1024, 1024)):
    """Resize the image to the target size (e.g., 1024x1024)."""
    image = image.resize(target_size, Image.LANCZOS)
    return image

def ensure_rgb(image):
    """Ensure the image has 3 channels (RGB). If the image has an alpha channel (RGBA), it will be converted to RGB."""
    if image.mode == 'RGBA':  # Check if the image has an alpha channel
        image = image.convert('RGB')  # Remove alpha channel and convert to RGB
    return image

app = Flask(__name__)

@app.route('/',  methods=['POST','GET'])
def hello_world():
    return 'Hello, World!'
import logging

@app.route('/wig_stick', methods=['POST'])
def wig_stick():
    # Check if the post request has the files
    if 'face' not in request.files or 'shape' not in request.files:
        return jsonify({"message": "Missing one or more required files (face, shape, color)"}), 400

    # Get the files from the request
    face_file = request.files['face']
    shape_file = request.files['shape']

    # If color is part of the input, handle it
    color_file = shape_file  # Using shape as color if color is not provided

    # Check if files are valid (you can add other checks here like file extensions)
    if face_file.filename == '' or shape_file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    try:
        # Log the file size and content type
        logging.info(f"Received face file: {face_file.filename}, MIME type: {face_file.content_type}, size: {len(face_file.read())} bytes")
        face_file.stream.seek(0)  # Reset the pointer after logging the file size

        logging.info(f"Received shape file: {shape_file.filename}, MIME type: {shape_file.content_type}, size: {len(shape_file.read())} bytes")
        shape_file.stream.seek(0)  # Reset the pointer

        if color_file:
            logging.info(f"Received color file: {color_file.filename}, MIME type: {color_file.content_type}, size: {len(color_file.read())} bytes")
            color_file.stream.seek(0)  # Reset the pointer

        # Try opening the files
        face_image = Image.open(io.BytesIO(face_file.read()))
        shape_image = Image.open(io.BytesIO(shape_file.read()))
        color_image = shape_image

        # Resize all images to a consistent size (1024x1024)
        target_size = (1024, 1024)  # Resize to a fixed size like 1024x1024
        face_image = resize_image(ensure_rgb(face_image), target_size)
        shape_image = resize_image(ensure_rgb(shape_image), target_size)
        if color_file:
            color_image = resize_image(ensure_rgb(color_image), target_size)

        # Log the resized image dimensions
        logging.info(f"Resized Face image size: {face_image.size}, Shape image size: {shape_image.size}")
        if color_file:
            logging.info(f"Resized Color image size: {color_image.size}")

        # Process the images here (e.g., hair swapping)
        # final_tensor = hair_fast.swap(face_image, shape_image, color_image)
        final_tensor = hair_fast.swap(face_image, shape_image, color_image, align=True)

        # Convert tensor to PIL Image
        final_image = final_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        final_image = (final_image * 255).astype(np.uint8)
        final_pil_image = Image.fromarray(final_image)

        # Convert the output image to a byte stream
        img_byte_arr = io.BytesIO()
        final_pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Return the final image as a response
        return Response(img_byte_arr, mimetype='image/png')

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"message": "Error in hair swapping", "error": str(e)}), 500
    
@app.route('/get_wig', methods=['POST'])
def get_wig():
    # Only check for 'face' since shape is loaded from path
    if 'face' not in request.files:
        return jsonify({"message": "Missing required file: face"}), 400

    face_file = request.files['face']
    choose_file = request.form.get('shape', '1')
    files = os.listdir("static/wig")
    if face_file.filename == '' or f"{choose_file}.png" not in files:
        return jsonify({"message": "No selected face file"}), 400

    try:
        # Open input images
        face_image = Image.open(io.BytesIO(face_file.read()))
        shape_image = Image.open(f"static/wig/{choose_file}.png")

        # Resize images
        target_size = (1024, 1024)
        face_image = resize_image(ensure_rgb(face_image), target_size)
        shape_image = resize_image(ensure_rgb(shape_image), target_size)
        color_image = shape_image  # Use shape as color too

        # Call your hair swap function
        final_tensor = hair_fast.swap(face_image, shape_image, color_image)

        # Convert tensor to PIL
        final_image = final_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        final_image = (final_image * 255).astype(np.uint8)
        final_pil_image = Image.fromarray(final_image)

        # Prepare response
        img_byte_arr = io.BytesIO()
        final_pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return Response(img_byte_arr, mimetype='image/png')

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"message": "Error in hair swapping", "error": str(e)}), 500
    
@app.route('/web')
def index():
    wig_dir = 'static/wig'
    files = [os.path.splitext(f)[0] for f in os.listdir(wig_dir) if f.lower().endswith(('.png'))]
    return render_template('index.html', wig_files=files)


if __name__ == '__main__':
    app.run(debug=True)