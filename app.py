from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import os
import uuid  # Import for generating unique filenames

app = Flask(__name__)

# Load the model
model = load_model('model/forge_mobilnet_50epochs.keras')

# Define upload directory
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None  # Default value for prediction
    original_image_path = None
    preprocessed_image_path = None
    result = None

    if request.method == 'POST':
        try:
            # Check if file is uploaded
            if 'imagefile' not in request.files or not request.files['imagefile'].filename:
                return "No file uploaded", 400  # Bad request if no file provided
            
            # Save uploaded file with a unique name
            imagefile = request.files['imagefile']
            unique_filename = f"{uuid.uuid4()}_{imagefile.filename}"
            image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            original_image_path = unique_filename  # Save the unique filename
            imagefile.save(image_path)

            # Step 1: Open and resize the image
            image = Image.open(image_path)
            resized_image = image.resize((224, 224), Image.Resampling.LANCZOS)

            # Step 2: Apply binary thresholding
            x = np.array(resized_image)  # Convert to numpy array
            x = cv2.threshold(x, 100, 255, cv2.THRESH_BINARY)[1]  # Apply binary thresholding
            preprocessed_image = Image.fromarray(x)

            # Save the preprocessed image with a unique name
            preprocessed_image_filename = f"preprocessed_{unique_filename}"
            preprocessed_image_path = os.path.join(UPLOAD_FOLDER, preprocessed_image_filename)
            preprocessed_image.save(preprocessed_image_path)

            # Step 3: Prepare the image for the model
            final_image = np.array(preprocessed_image) / 255.0  # Normalize to [0, 1]
            final_image = final_image.reshape((1, 224, 224, 3))  # Add batch dimension

            # Step 4: Predict using the model
            result = model.predict(final_image)
            prediction = 'Forged' if result[0][0] > 0.5 else 'Genuine'
            print(result)

        except Exception as e:
            prediction = f"Error: {str(e)}"  # Handle errors gracefully

    # Render the template and pass the prediction and image paths
    return render_template(
        'index.html',
        prediction=prediction,
        original_image_path=original_image_path,
        preprocessed_image_path=preprocessed_image_filename if preprocessed_image_path else None,
        result=result
    )

# Serve files from the uploads folder
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    print(f"Uploads directory: {os.path.abspath(UPLOAD_FOLDER)}")  # Debugging info
    app.run(port=3000, debug=True)
