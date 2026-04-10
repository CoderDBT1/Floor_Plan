import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

import json
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import re
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval()
    return model

model = load_model()

# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# -----------------------------
# Parse Dimensions
# -----------------------------
def parse_dimension(text):
    pattern = r"(\d+)\'(?:\s+(\d+)\")?\s*x\s*(\d+)\'(?:\s+(\d+)\")?"
    match = re.search(pattern, text)
    if match:
        ft1, inch1, ft2, inch2 = match.groups()
        width = float(ft1) + (float(inch1) if inch1 else 0) / 12.0
        height = float(ft2) + (float(inch2) if inch2 else 0) / 12.0
        return round(width, 2), round(height, 2)
    return None, None

# -----------------------------
# Orientation
# -----------------------------
def detect_orientation(bbox):
    x1, y1, x2, y2 = bbox
    return "horizontal" if (x2 - x1) > (y2 - y1) else "vertical"

# -----------------------------
# Mock Data
# -----------------------------
room_texts = [
    ("BALCONY", "11' x 6'"),
    ("KITCHEN", "9' 4\" x 7' 3\""),
    ("LIVING", "16' 7\" x 17' 5\""),
    ("BEDROOM", "12' 6\" x 11' 6\""),
    ("BEDROOM", "12' 0\" x 9' 11\""),
]

approximate_bboxes = [
    [50, 30, 180, 120],
    [200, 50, 350, 180],
    [400, 100, 650, 350],
    [50, 200, 250, 380],
    [280, 250, 450, 400],
]

# -----------------------------
# Generate JSON
# -----------------------------
def generate_json(image):
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        _ = model(img_tensor)

    floorplan = {
        "properties": {
            "rooms": [],
            "orientation": "landscape" if image.width > image.height else "portrait"
        }
    }

    for i, (name, dim) in enumerate(room_texts):
        w, h = parse_dimension(dim)
        bbox = approximate_bboxes[i]

        floorplan["properties"]["rooms"].append({
            "name": name,
            "dimensions": {
                "width_ft": w,
                "height_ft": h,
                "area_sqft": round(w*h, 2) if w and h else None
            },
            "orientation": detect_orientation(bbox)
        })

    return floorplan

# -----------------------------
# 3D Visualization
# -----------------------------
def visualize_3d(floorplan):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for room in floorplan["properties"]["rooms"]:
        if room["dimensions"]["width_ft"] and room["dimensions"]["height_ft"]:
            w = room["dimensions"]["width_ft"]
            h = room["dimensions"]["height_ft"]
            z = min(2, (w * h) / 200)

            x = [0, w, w, 0]
            y = [0, 0, h, h]

            for i in range(4):
                face = [
                    (x[i], y[i], 0),
                    (x[(i+1)%4], y[(i+1)%4], 0),
                    (x[(i+1)%4], y[(i+1)%4], z),
                    (x[i], y[i], z)
                ]
                ax.add_collection3d(Poly3DCollection([face], alpha=0.3))

    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_zlabel('Z')

    return fig

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🏠 Floor Plan Analyzer")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # FIX image format issue
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = Image.fromarray(np.array(image))

    st.image(image, caption="Uploaded Image")

    with st.spinner("Analyzing..."):
        result = generate_json(image)

    st.subheader("📊 JSON Output")
    st.json(result)

    st.subheader("🧱 3D Visualization")
    fig = visualize_3d(result)
    st.pyplot(fig)

    st.download_button(
        "Download JSON",
        data=json.dumps(result, indent=2),
        file_name="output.json"
    )
