import json

def load_annotations(json_file):
    """
    Load annotations from a COCO-formatted JSON file.
    """
    with open(json_file, 'r') as f:
        annotations = json.load(f)
    return annotations

def coco_to_ssd_annotations(coco_annotations):
    """
    Convert COCO-format annotations to a list of dictionaries suitable for use with an SSD model.
    """
    image_dict = {}
    ssd_annotations = []
    
    for ann in coco_annotations['annotations']:
        image_id = ann['image_id']
        bbox = ann['bbox']
        category_id = ann['category_id']
        
        image_info = coco_annotations['images'][image_id]
        image_file_name = image_info['file_name']
        
        if image_file_name not in image_dict:
            image_dict[image_file_name] = {'annotations': [], 'labels': []}
        
        # Convert COCO bbox format [x,y,width,height] to [xmin, ymin, xmax, ymax]
        xmin, ymin, width, height = bbox
        xmax = xmin + width
        ymax = ymin + height
        ssd_bbox = [xmin, ymin, xmax, ymax]
        
        image_dict[image_file_name]['annotations'].append(ssd_bbox)
        image_dict[image_file_name]['labels'].append(category_id)
        
    for image_file_name, data in image_dict.items():
        ssd_annotations.append({
            'file_name': image_file_name,
            'annotations': data['annotations'],
            'labels': data['labels']
        })
        
    return ssd_annotations


