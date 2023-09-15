import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import os

# Initialize the model
num_classes = 4  # Replace with the number of classes you have
label_map = {1: 'basketball', 2: 'rim'}  # Map labels to their textual names

model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)

# Load the saved weights
model.load_state_dict(torch.load("epoch_1_checkpoint.pth"))
model.eval()

# Load an image
image_path = "data/train/img/zzpic11414_jpg.rf.523b241a65a762f6bcc5672f852aa16c.jpg"  # Replace with the path to your image
image = Image.open(image_path).convert("RGB")
image_tensor = F.to_tensor(image).unsqueeze(0)

# Run the image through the model
with torch.no_grad():
    prediction = model(image_tensor)

# The output is a list of dict, with keys 'boxes', 'labels', and 'scores'
boxes = prediction[0]['boxes']
labels = prediction[0]['labels']
scores = prediction[0]['scores']

# Let's only consider detections with a confidence score >= 0.5
for idx in range(boxes.shape[0]):
    print(scores[idx])
    if scores[idx] >= 0.24:

        box = boxes[idx].tolist()
        label = labels[idx].item()
        score = scores[idx].item()
        
        # Get the textual label from the label map
        label_name = label_map.get(label, f"Unknown {label}")

        # Draw bounding boxes and labels
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"{label_name}, Score: {score:.2f}", fill="red")

# Save or display the modified image
image.save("predicted_image.jpg")
image.show()
