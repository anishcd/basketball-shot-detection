import torch
import torch.utils.data
from torchvision.models.detection import ssdlite320_mobilenet_v3_large  # Example SSD model
from torchvision.models import mobilenet_v3_large
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from utils import load_annotations, coco_to_ssd_annotations
import yaml
from PIL import Image
import os

# Load the YAML configuration file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Access data paths and other settings from config.yaml
train_annotations_path = config['data']['train_annotations']
train_images_path = config['data']['train_images']
batch_size = config['hyperparameters']['batch_size']
num_classes = config['model']['num_classes']


# Initialize the dataset and dataloader from utils.py functions
train_annotations = load_annotations(train_annotations_path)  # Implement this function to load your JSON annotations
ssd_annotations_train = coco_to_ssd_annotations(train_annotations)  # Convert to SSD format


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, annotations, root_dir):
        self.annotations = annotations
        self.root_dir = root_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = list(self.annotations.keys())[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = F.to_tensor(Image.open(img_path).convert("RGB"))
        target = {}
        
        # Extract only the box coordinates and ensure they are in the format [N, 4]
        boxes = [ann[1:] for ann in self.annotations[img_name]]
        target["boxes"] = torch.FloatTensor(boxes)
        
        # Extract only the labels
        target["labels"] = torch.tensor([ann[0] for ann in self.annotations[img_name]], dtype=torch.int64)
        
        return image, target



# Initialize the dataset and dataloader
dataset = CustomDataset(ssd_annotations_train, train_images_path)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))


# Initialize the model - I am using this SSD model: https://pytorch.org/vision/main/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html
# Initialize the model with MobileNetV3 backbone
model = ssdlite320_mobilenet_v3_large(pretrained=False, num_classes=num_classes)

# Specify the training settings
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Training on MPS (M1 GPU)")
else:
    device = torch.device("cpu")
    print("Training on CPU")
model.to(device)
model.train()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=config['hyperparameters']['learning_rate'], momentum=0.9, weight_decay=0.0005)

print(f"Total Number of Training Samples: {len(dataset)}")

# Training loop
for epoch in range(config['hyperparameters']['epochs']):
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Print log
        print(f'Epoch [{epoch+1}/{config["hyperparameters"]["epochs"]}] Batch [{batch_idx+1}/{len(data_loader)}] Loss: {losses.item()}')
        
        # Backward pass and optimization
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

# Save the model
torch.save(model.state_dict(), config['save_model_path'])

