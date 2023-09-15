import torch
import torch.utils.data
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from utils import load_annotations, coco_to_ssd_annotations
import yaml
from PIL import Image
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import matplotlib.pyplot as plt

# Load the YAML configuration file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Access data paths and other settings from config.yaml
train_annotations_path = config['data']['train_annotations']
train_images_path = config['data']['train_images']
valid_annotations_path = config['data']['valid_annotations']
valid_images_path = config['data']['valid_images']
batch_size = config['hyperparameters']['batch_size']
num_classes = config['model']['num_classes']

# Initialize the dataset and dataloader from utils.py functions
train_annotations = load_annotations(train_annotations_path)
ssd_annotations_train = coco_to_ssd_annotations(train_annotations)

valid_annotations = load_annotations(valid_annotations_path)
ssd_annotations_valid = coco_to_ssd_annotations(valid_annotations)



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, annotations, root_dir):
        self.annotations = annotations
        self.root_dir = root_dir
        print(f"Number of annotations: {len(self.annotations)}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations[idx]['file_name']
        img_path = os.path.join(self.root_dir, img_name)
        image = F.to_tensor(Image.open(img_path).convert("RGB"))
        target = {}
        
        boxes = self.annotations[idx]['annotations']  # Assuming this contains the [x, y, width, height] annotations
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Convert to [x1, y1, x2, y2] format
        boxes[:, 2:] += boxes[:, :2]

        # Extract labels
        labels = self.annotations[idx]['labels']
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target["boxes"] = boxes
        target["labels"] = labels
        
        return image, target


# Initialize the dataset and dataloader
dataset = CustomDataset(ssd_annotations_train, train_images_path)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
# Initialize the validation dataset and dataloader
valid_dataset = CustomDataset(ssd_annotations_valid, valid_images_path)
valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Initialize the model with a ResNet-50 backbone and FPN
model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)

# Specify the training settings
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Training on MPS (M1 GPU)")
else:
    device = torch.device("cpu")
    print("Training on CPU")
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=config['hyperparameters']['learning_rate'], momentum=config['hyperparameters']['momentum'], weight_decay=config['hyperparameters']['weight_decay'])

# Initialize variables for early stopping
patience = 2  # Number of epochs to wait for improvement before stopping
best_valid_loss = float('inf')
counter = 0  # Counter to keep track of non-improving epochs
train_loss_over_epochs = []
valid_loss_over_epochs = []

for epoch in range(config['hyperparameters']['epochs']):
    # Training loop (same as before)
    train_losses = []
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        train_losses.append(losses.item())
        print(f'Epoch [{epoch+1}/{config["hyperparameters"]["epochs"]}] Batch [{batch_idx+1}/{len(data_loader)}] Loss: {losses.item()}')
    torch.save(model.state_dict(), f"epoch_{epoch+1}_checkpoint.pth")


    # Calculate average training loss
    avg_train_loss = sum(train_losses) / len(train_losses)
    
    model.eval()
    val_loss_total = 0
    val_batches = 0
    for batch_idx, (images, targets) in enumerate(valid_data_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            model.train()  # Temporarily set to train mode to get loss dict
            val_loss_dict = model(images, targets)
            model.eval()  # Set back to eval mode
            val_losses = sum(loss for loss in val_loss_dict.values())
            val_loss_total += val_losses.item()
            val_batches += 1
    avg_valid_loss = val_loss_total / val_batches

    train_loss_over_epochs.append(avg_train_loss)
    valid_loss_over_epochs.append(avg_valid_loss)
    
    
    print(f'Epoch [{epoch+1}/{config["hyperparameters"]["epochs"]}], Train Loss: {avg_train_loss}, Valid Loss: {avg_valid_loss}')
    
    # Check for early stopping
    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        counter = 0  # Reset counter
        torch.save(model.state_dict(), config['save_model_path'])
    else:
        counter += 1
        print(f'EarlyStopping counter: {counter} out of {patience}')
        if counter >= patience:
            print('Early stopping')
            break
    
    model.train()  # Reset the model to training mode for the next epoch

plt.plot(train_loss_over_epochs, label='Training loss')
plt.plot(valid_loss_over_epochs, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()

# Save the model
torch.save(model.state_dict(), config['save_model_path'])

