import torch

## A Py-torch based SSD Implementation requires the annotations to be in the following format for each image to be processed:
## [class_id, x_{min}, y_{min}, x_{max}, y_{max}], where (x_{min},y_{min}) and (x_{max},y_{max}) are the coordinates of the top-left and bottom-right corners of the bounding boxes, respectively. 
# Function to convert COCO annotations to SSD format
def coco_to_ssd_annotations(coco_annotations):
    ssd_annotations = {}
    
    # Create a mapping from image ID to file name
    image_id_to_file_name = {image['id']: image['file_name'] for image in coco_annotations['images']}
    
    # Loop through each annotation and convert it
    for annotation in coco_annotations['annotations']:
        image_id = annotation['image_id']
        file_name = image_id_to_file_name[image_id]
        
        # Extract bounding box and category ID
        bbox = annotation['bbox']
        category_id = annotation['category_id']
        
        # Convert COCO bbox format ([x_min, y_min, width, height]) to SSD format ([x_min, y_min, x_max, y_max])
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        
        # Create the SSD annotation for this object
        ssd_annotation = [category_id, x_min, y_min, x_max, y_max]
        
        # Add this annotation to the list of annotations for this image
        if file_name not in ssd_annotations:
            ssd_annotations[file_name] = []
        ssd_annotations[file_name].append(ssd_annotation)
    
    return ssd_annotations

import json

def load_annotations(json_path):
    """
    Load annotations from a COCO-formatted JSON file.
    
    Parameters:
        json_path (str): Path to the JSON file.
        
    Returns:
        dict: Loaded annotations.
    """
    with open(json_path, 'r') as file:
        annotations = json.load(file)
    return annotations

print(torch.cuda.is_available())
import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")