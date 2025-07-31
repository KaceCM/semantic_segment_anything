import os
import argparse
import torch
import json
import numpy as np
from assets.utils import printr
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

from pipeline_ssa import semantic_segment_anything_inference, img_load
from assets.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
printr(f"Using device: {DEVICE}")

def parse_args():
    parser = argparse.ArgumentParser(description='Semantically segment anything.')
    # parser.add_argument('--data_dir', help='specify the root path of images and masks')
    parser.add_argument('--img_path', help='specify the root path of images and masks')
    parser.add_argument('--save_masks', default=True, action='store_true', help='whether to save masks')
    parser.add_argument('--ckpt_path', default='ckp/sam_vit_h_4b8939.pth', help='specify the root path of SAM checkpoint')
    parser.add_argument('--out_dir', help='the dir to save semantic annotations')
    parser.add_argument('--save_img', default=False, action='store_true', help='whether to save annotated images')
    parser.add_argument('--world_size', type=int, default=0, help='number of nodes')
    parser.add_argument('--dataset', type=str, default='ade20k', choices=['ade20k', 'cityscapes', 'foggy_driving'], help='specify the set of class names')
    parser.add_argument('--eval', default=False, action='store_true', help='whether to execute evalution')
    parser.add_argument('--gt_path', default=None, help='specify the path to gt annotations')
    parser.add_argument('--model', type=str, default='segformer', choices=['oneformer', 'segformer'], help='specify the semantic branch model')
    args = parser.parse_args()
    return args
    
def main(rank, args):
    printr('STARTING MAIN FUNCTION')

    # printr('SETTING DIST INITPROCESS')
    # dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    # printr('DIST INIT DONE')


    printr('[Model loading] Loading SAM model...')
    sam = sam_model_registry["vit_h"](checkpoint=args.ckpt_path).to(rank)
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
    # yoo can add your own semantic branch here, and modify the following code

    printr('[Model loading] Loading semantic branch model (oneformer)...')
    cache_dir = os.path.dirname(args.ckpt_path)
    semantic_branch_processor = OneFormerProcessor.from_pretrained(
        "shi-labs/oneformer_ade20k_swin_large",
        cache_dir=cache_dir
        )
    semantic_branch_model = OneFormerForUniversalSegmentation.from_pretrained(
        "shi-labs/oneformer_ade20k_swin_large",
        cache_dir=cache_dir
        ).to(rank)
    printr('[Model loaded] Semantic branch (your own segmentor) is loaded.')
    
    
    image_name = os.path.basename(args.img_path)
    image_name_no_ext = image_name.replace('.jpg', '').replace('.png', '')
    image_path = os.path.dirname(args.img_path)
    printr(f'Image name: {image_name}')
    printr(f'Image path: {image_path}')

    
    printr('[Image loading] Loading image...')
    img = img_load(image_path, image_name)
    printr('[Image loaded] Image is loaded.')

    id2label = CONFIG_ADE20K_ID2LABEL

    printr('[Inference] Starting semantic segmentation inference...')
    with torch.no_grad():
        result = semantic_segment_anything_inference(image_name_no_ext, args.out_dir, rank, img=img, save_img=args.save_img,
                                semantic_branch_processor=semantic_branch_processor,
                                semantic_branch_model=semantic_branch_model,
                                mask_branch_model=mask_branch_model,
                                dataset=args.dataset,
                                id2label=id2label,
                                model=args.model)
        printr('[Inference done] Semantic segmentation inference is done.')
        
    
        if args.save_masks:
            printr('[Saving results] Saving results...')
            save_to_tensor(result, args.out_dir, image_name_no_ext)
            save_to_json(result, args.out_dir, image_name_no_ext)
            printr('[Saving results done] Results are saved.')
            del result
            

    printr('MAIN FUNCTION DONE')
    return True

def save_to_tensor(result, out_dir, file_name):
    resultarray = np.array(result["semantic_masks"])
    result["semantic_masks"] = torch.tensor(resultarray)
    torch.save(result["semantic_masks"], os.path.join(out_dir, file_name + '_semantic_masks.pt'))
    return True

def save_to_json(result, out_dir, file_name):
    with open(os.path.join(out_dir, file_name + '_info.json'), 'w') as f:
        result.pop('semantic_masks')
        result.pop('instance_masks')
        json.dump(result, f, indent=4)

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    results = main(DEVICE, args)