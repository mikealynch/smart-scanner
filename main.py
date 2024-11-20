import os
import easyocr
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import openai
from dotenv import load_dotenv
import requests
import json

# Set the environment variable for easyocr
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load API key from environment variables
load_dotenv()
api_key = os.getenv("API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Streamlit app header
st.title('Let\'s Scan a Business Card')

# File uploader
uploaded_file = st.file_uploader("Upload a business card image for OCR analysis", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file:
    try:
        # Convert uploaded file to OpenCV image format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Failed to load image. Ensure the file is a valid image.")
            st.stop()

        # Convert to HSV and create mask for business card detection
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, np.array([0, 0, 180]), np.array([180, 50, 255]))

        # Find contours and extract the largest one
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            st.error("No contours found; adjust the color range or upload a clearer image.")
            st.stop()

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the detected region
        cropped_image = image[y:y+h, x:x+w]
        pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        compressed_image = pil_image.resize((int(pil_image.width * 0.5), int(pil_image.height * 0.5)))

        # Display cropped image
        st.image(compressed_image, caption="Cropped Business Card", use_container_width=True)

        # OCR with EasyOCR
        reader = easyocr.Reader(['en', 'es'])
        result = reader.readtext(np.array(compressed_image))

        # Prepare extracted text for OpenAI prompt
        text_values = [detection[1] for detection in result]
        prompt = (
            "The following text was extracted from a business card. Convert it into valid JSON containing fields: "
            "first name, last name, position, email, phone number, country, and company name. Text: "
            + " ".join(text_values)
        )

        # Call OpenAI API to format data into JSON
        data = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": prompt,
            "temperature": 0.9,
            "max_tokens": 150
        }
        response = requests.post(
            "https://api.openai.com/v1/completions",
            headers=headers,
            data=json.dumps(data),
            verify=False  # Disable SSL verification
        )

        clean_response = response.json()['choices'][0]['text'].strip()

        # Validate JSON response
        while True:
            try:
                categorized_data = json.loads(clean_response)
                break
            except json.JSONDecodeError:
                prompt = "Convert this text into valid JSON: " + clean_response
                data["prompt"] = prompt
                response = requests.post(
                    "https://api.openai.com/v1/completions",
                    headers=headers,
                    data=json.dumps(data),
                    verify=False
                )
                clean_response = response.json()['choices'][0]['text'].strip()

        # Display editable fields for validation
        st.write("Please review and edit the extracted data below:")
        for key, value in categorized_data.items():
            categorized_data[key] = st.text_input(key, value)

        if st.button("Confirm Data"):
            st.success("Thank you! The data has been successfully reviewed.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
