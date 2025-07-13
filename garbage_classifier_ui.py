import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image
import json
import os

# Load class index map
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_indices))
model.load_state_dict(torch.load('garbage_classifier_resnet.pth', map_location=torch.device('cpu')))
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(image):
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        predicted_class = idx_to_class[pred.item()]
        confidence = torch.softmax(output, dim=1)[0, pred.item()].item() * 100
    return f"Predicted: {predicted_class} ({confidence:.2f}%)"

# Launch Gradio UI
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(),
    title="Smart Garbage Classifier",
    description="Upload an image of garbage to classify it into 7 waste categories."
)

if __name__ == '__main__':
    interface.launch()
# garbage_classifier_ui.py
# This script sets up a Gradio UI for the garbage classification model.