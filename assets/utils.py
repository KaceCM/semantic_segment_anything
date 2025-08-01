from datetime import datetime
import os
import json
import numpy as np
import torch
from PIL import Image

def printr(text):
    """Prints text with a timestamp."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - {text}")

def save_to_tensor(result, out_dir, file_name):
    resultarray = np.array(result["semantic_masks"])
    result["semantic_masks"] = torch.tensor(resultarray)
    torch.save(result["semantic_masks"], os.path.join(out_dir, file_name + '_semantic_masks.pt'))
    return True

def save_to_json(result, out_dir, file_name):
    with open(os.path.join(out_dir, file_name + '_info.json'), 'w') as f:
        result.pop('semantic_masks')
        result.pop('instance_masks')
        result.pop('class_names')
        json.dump(result, f, indent=4)

def img_load(data_path, filename):
    img = Image.open(os.path.join(data_path, filename)).convert('RGB')
    img_ndarray = np.array(img)
    return img_ndarray

def prepare_image(args):
    image_name = os.path.basename(args.img_path)
    image_name_no_ext = image_name.replace('.jpg', '').replace('.png', '')
    image_path = os.path.dirname(args.img_path)
    
    printr(f'Image name: {image_name}')
    printr(f'Image path: {image_path}')
    printr('[Image loading] Loading image...')
    
    img = img_load(image_path, image_name)
    printr('[Image loaded] Image is loaded.')
    return img, image_name_no_ext