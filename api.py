from flask import Flask, request, jsonify, Response

import io
import numpy as np
import subprocess
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import torch
from torchvision.utils import save_image
from werkzeug.utils import secure_filename

from hair_swap import HairFast, get_parser

model_parser = get_parser()
model_args, _ = model_parser.parse_known_args()
hair_fast = HairFast(model_args)

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
    color_file = request.files.get('color')  # Optional, if you want to handle it

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

        # Debugging: Check if the images are opened successfully
        logging.info(f"Face image size: {face_image.size}, Shape image size: {shape_image.size}")
        if color_file:
            logging.info(f"Color image size: {color_image.size}")

        # Process the images here (e.g., hair swapping)
        final_tensor = hair_fast.swap(face_image, shape_image, color_image)

        # Convert the tensor to a PIL Image
        # Assuming the output tensor is in the format [C, H, W] (channels, height, width)
        final_image = final_tensor.squeeze(0)  # Remove batch dimension (if any)
        final_image = final_image.permute(1, 2, 0).cpu().detach().numpy()  # Convert tensor to numpy (H, W, C)

        # If the image has been processed in range [0, 1] or [0, 255], you may need to scale it:
        final_image = (final_image * 255).astype(np.uint8)

        # Convert to PIL Image
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


if __name__ == '__main__':
    app.run(debug=True)