import cv2
import pandas as pd
import torch
from PIL import Image
from transformers import ViTImageProcessor, AutoModelForImageClassification
import json
import google.generativeai as genai

class ImageProcessor:
    def __init__(self, color_csv_path, model_name, api_key):
        self.color_csv_path = color_csv_path
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.labels = self.model.config.id2label
        self.csv = pd.read_csv(self.color_csv_path, names=["color", "color_name", "hex", "R", "G", "B"])
        genai.configure(api_key=api_key)

    def get_color_name(self, R, G, B):
        min_distance = float('inf')
        color_name = ''
        for i in range(len(self.csv)):
            d = abs(R - int(self.csv.loc[i, "R"])) + abs(G - int(self.csv.loc[i, "G"])) + abs(B - int(self.csv.loc[i, "B"]))
            if d <= min_distance:
                min_distance = d
                color_name = self.csv.loc[i, "color_name"]
        return color_name

    def classify_clothing(self, image_path):
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        return self.labels[predicted_class_idx]

    def process_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None

        height, width, _ = img.shape
        center_x, center_y = width // 2, height // 2
        b, g, r = img[center_y, center_x]
        r, g, b = int(r), int(g), int(b)

        color_name = self.get_color_name(r, g, b)
        clothing_type = self.classify_clothing(img_path)

        return {
            "color_name": color_name,
            "color_rgb": {"R": r, "G": g, "B": b},
            "clothing_type": clothing_type
        }

    def generate_outfit_suggestion(self, results):
        model = genai.GenerativeModel(
            model_name="gemini-1.0-pro",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_output_tokens": 100,
                "response_mime_type": "text/plain",
            },
        )
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": ["hi\n"],
                },
                {
                    "role": "model",
                    "parts": ["welcome, i will help you with your fashion question and output will be short and sweet in one line perfect match\n"],
                },
            ]
        )
        user_prompt = f"Based on my current clothing type '{results['clothing_type']}' and the color '{results['color_name']}', can you suggest an outfit?"
        response = chat_session.send_message(user_prompt)
        return response.text
