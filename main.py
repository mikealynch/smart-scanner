import os
import easyocr
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import json
import requests
from dotenv import load_dotenv

# Set the environment variable before importing easyocr
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load API key from environment variables
load_dotenv()
api_key = os.getenv("API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Header section for the Streamlit app
st.title("Let's Scan a Business Card")

# File uploader for image input
uploaded_file = st.file_uploader("Upload a business card image for OCR analysis", type=["jpg", "jpeg", "png", "bmp"])

# Initialize session state for field values
if "field_values" not in st.session_state:
    st.session_state.field_values = {}

if uploaded_file is not None:
    try:
        # Convert the uploaded file to a NumPy array (OpenCV format)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Decode into OpenCV image format

        if image is None:
            raise ValueError("Failed to load image. Ensure the file is a valid image.")

        # Convert to HSV color space for segmentation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0, 0, 180])  # Light color lower HSV bound
        upper_bound = np.array([180, 50, 255])  # Light color upper HSV bound
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # Find contours and crop the largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found; ensure the color range includes the business card.")

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y:y+h, x:x+w]

        # Initialize EasyOCR and perform OCR
        reader = easyocr.Reader(['en', 'es'])
        result = reader.readtext(cropped_image)

        # Extract detected text and structure for the API prompt
        text_values = [detection[1] for detection in result]
        prompt = (
            "The information in this data set was pulled from a single business card. "
            "Using this information, create valid JSON with the fields: first name, last name, position, email, "
            "phone number, country, and company name: " + " ".join(text_values)
        )

        # Call OpenAI API to process the text
        data = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": prompt,
            "temperature": 0.9,
            "max_tokens": 150
        }
        response = requests.post(
            "https://api.openai.com/v1/completions",
            headers=headers,
            data=json.dumps(data)
        )
        clean_response = response.json()["choices"][0]["text"].strip()
        categorized_data = json.loads(clean_response)

        # Save the categorized data in session state
        st.session_state.field_values = categorized_data

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Display editable fields for user review
if st.session_state.field_values:
    st.write("Please review and edit the extracted data below:")
    for key, value in st.session_state.field_values.items():
        st.session_state.field_values[key] = st.text_input(key, value=value)

# Button to confirm data and display final values
if st.button("The information above is accurate."):
    st.success("Thank you!")
    st.write("Final Data Submitted:")
    st.json(st.session_state.field_values)
