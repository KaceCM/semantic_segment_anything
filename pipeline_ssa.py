import os
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import pycocotools.mask as maskUtils
import matplotlib.pyplot as plt
from assets.utils import printr
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation


def load_mask_generator(args, DEVICE):
    printr('[Model loading] Loading SAM model...')
    sam = sam_model_registry["vit_h"](checkpoint=args.ckpt_path).to(DEVICE)
    printr('[Model loaded] SAM model is loaded.')

    printr('[Model loading] Loading SAM mask branch...')
    mask_branch_model = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
    
        pred_iou_thresh=0.80,
        stability_score_thresh=0.85,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
        output_mode='coco_rle',
    )
    printr('[Model loaded] Mask branch (SAM) is loaded.')
    return mask_branch_model

def load_semantic_branch(args, DEVICE):
    cache_dir = os.path.dirname(args.ckpt_path)
    printr('[Model loading] Loading semantic branch model (oneformer)...')
    
    semantic_branch_processor = OneFormerProcessor.from_pretrained(
        "shi-labs/oneformer_ade20k_swin_large",
        cache_dir=cache_dir
        )
    semantic_branch_model = OneFormerForUniversalSegmentation.from_pretrained(
        "shi-labs/oneformer_ade20k_swin_large",
        cache_dir=cache_dir
        ).to(DEVICE)
    printr('[Model loaded] Semantic branch (your own segmentor) is loaded.')
    return semantic_branch_processor, semantic_branch_model


def oneformer_ade20k_segmentation(image, oneformer_ade20k_processor, oneformer_ade20k_model, rank):
    inputs = oneformer_ade20k_processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(rank)
    outputs = oneformer_ade20k_model(**inputs)
    predicted_semantic_map = oneformer_ade20k_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]])[0]
    return predicted_semantic_map


def generate_random_masks(mask_branch_model, img):
    printr('Generating masks using AutomaticMaskGeneration')
    anns = {'annotations': mask_branch_model.generate(img)}
    printr('Masks generated using AutomaticMaskGeneration')
    return anns


def generate_oneformer_masks(semantic_branch_processor, semantic_branch_model, img, DEVICE):
    printr('[Inference] Starting OneFormer/SegFormer inference...')
    class_ids = oneformer_ade20k_segmentation(Image.fromarray(img), semantic_branch_processor,semantic_branch_model, DEVICE)
    printr('[Inference done] OneFormer/SegFormer inference is done.')

    return class_ids


def save_raw_segformer_masks(filename, output_path, class_ids):
    printr('[Inference] Starting SegFormer inference...')
    num_classes = int(class_ids.max().item()) + 1
    cmap = plt.get_cmap('tab20', num_classes)
    class_ids_np = class_ids.cpu().numpy().astype(np.uint8)
    color_class_ids = cmap(class_ids_np / (num_classes if num_classes > 0 else 1))[:, :, :3]  # Drop alpha
    color_class_ids = (color_class_ids * 255).astype(np.uint8)
    out_path_raw = os.path.join(output_path, filename + '_oneformer_raw.png')
    Image.fromarray(color_class_ids).save(out_path_raw)
    printr(f'[Save] OneFormer/SegFormer raw output image saved to: {out_path_raw}')


def generate_semantic_masks(anns, class_ids, id2label):

    class_names = []
    masks_list = []  # List to store all masks

    printr('Starting for loop over annotations to generate semantic masks...')
    semantic_mask = class_ids.clone()
    anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)
    for ann in tqdm(anns['annotations']):
        valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
        masks_list.append(valid_mask)  # Store each mask
        # get the class ids of the valid pixels
        propose_classes_ids = class_ids[valid_mask]
        num_class_proposals = len(torch.unique(propose_classes_ids))
        if num_class_proposals == 1:
            semantic_mask[valid_mask] = propose_classes_ids[0]
            ann['class_name'] = id2label['id2label'][str(propose_classes_ids[0].item())]
            ann['class_proposals'] = id2label['id2label'][str(propose_classes_ids[0].item())]
            class_names.append(ann['class_name'])
            continue
        top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
        top_1_propose_class_names = [id2label['id2label'][str(class_id.item())] for class_id in top_1_propose_class_ids]

        semantic_mask[valid_mask] = top_1_propose_class_ids
        ann['class_name'] = top_1_propose_class_names[0]
        ann['class_proposals'] = top_1_propose_class_names[0]
        class_names.append(ann['class_name'])

        del valid_mask
        del propose_classes_ids
        del num_class_proposals
        del top_1_propose_class_ids
        del top_1_propose_class_names
    
    semantic_class_in_img = torch.unique(semantic_mask)
    semantic_bitmasks, semantic_class_names = [], []
    printr('Semantic masks generated...')
    return semantic_class_in_img, semantic_class_names, semantic_bitmasks, semantic_mask, class_names, masks_list


def generate_semantic_prediction(anns, semantic_class_in_img, semantic_mask, id2label, semantic_bitmasks, semantic_class_names):
    printr('[Inference] Starting semantic prediction...')
    anns['semantic_mask'] = {}
    for i in range(len(semantic_class_in_img)):
        class_name = id2label['id2label'][str(semantic_class_in_img[i].item())]
        class_mask = semantic_mask == semantic_class_in_img[i]
        class_mask = class_mask.cpu().numpy().astype(np.uint8)
        semantic_class_names.append(class_name)
        semantic_bitmasks.append(class_mask)
        anns['semantic_mask'][str(semantic_class_in_img[i].item())] = maskUtils.encode(np.array((semantic_mask == semantic_class_in_img[i]).cpu().numpy(), order='F', dtype=np.uint8))
        anns['semantic_mask'][str(semantic_class_in_img[i].item())]['counts'] = anns['semantic_mask'][str(semantic_class_in_img[i].item())]['counts'].decode('utf-8')
    printr('[Inference done] Semantic prediction is done.')
    return anns, semantic_class_names, semantic_bitmasks

def save_semantic_masks(output_path, filename, semantic_mask, semantic_class_names):
    num_classes = len(semantic_class_names)
    cmap = plt.get_cmap('tab20', num_classes)
    semantic_mask_np = semantic_mask.cpu().numpy().astype(np.uint8)
    color_mask = cmap(semantic_mask_np / (num_classes if num_classes > 0 else 1))[:, :, :3]  # Drop alpha
    color_mask = (color_mask * 255).astype(np.uint8)

    
    out_path = os.path.join(output_path, filename + '_semantic_mask.png')
    Image.fromarray(color_mask).save(out_path)
    printr(f'[Save] save SSA prediction: {out_path}')
    
