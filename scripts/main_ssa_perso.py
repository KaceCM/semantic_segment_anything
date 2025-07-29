import os
import argparse
import torch
import json
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

from pipeline_perso import semantic_segment_anything_inference, img_load
from configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL

import torch.distributed as dist
import torch.multiprocessing as mp
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12322'

def parse_args():
    parser = argparse.ArgumentParser(description='Semantically segment anything.')
    parser.add_argument('--data_dir', help='specify the root path of images and masks')
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
    print("SAVING MASKS" if args.save_masks else "NOT SAVING MASKS")
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    
    sam = sam_model_registry["vit_h"](checkpoint=args.ckpt_path).to(rank)

    mask_branch_model = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=128 if args.dataset == 'foggy_driving' else 64,
    
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
        output_mode='coco_rle',
    )
    print('[Model loaded] Mask branch (SAM) is loaded.')
    # yoo can add your own semantic branch here, and modify the following code

    semantic_branch_processor = OneFormerProcessor.from_pretrained(
        "shi-labs/oneformer_ade20k_swin_large")
    semantic_branch_model = OneFormerForUniversalSegmentation.from_pretrained(
        "shi-labs/oneformer_ade20k_swin_large").to(rank)
    print('[Model loaded] Semantic branch (your own segmentor) is loaded.')
    
    filenames = [fn_.replace('.jpg', '') for fn_ in os.listdir(args.data_dir) if '.jpg' in fn_]
    local_filenames = filenames[(len(filenames) // args.world_size + 1) * rank : (len(filenames) // args.world_size + 1) * (rank + 1)]
    print('[Image name loaded] get image filename list.')
    print('[SSA start] model inference starts.')


    for i, file_name in enumerate(local_filenames):
        print('[Runing] ', i, '/', len(local_filenames), ' ', file_name, ' on rank ', rank, '/', args.world_size)
        img = img_load(args.data_dir, file_name, args.dataset)

        id2label = CONFIG_ADE20K_ID2LABEL
        with torch.no_grad():
            result = semantic_segment_anything_inference(file_name, args.out_dir, rank, img=img, save_img=args.save_img,
                                   semantic_branch_processor=semantic_branch_processor,
                                   semantic_branch_model=semantic_branch_model,
                                   mask_branch_model=mask_branch_model,
                                   dataset=args.dataset,
                                   id2label=id2label,
                                   model=args.model)
            
            if args.save_masks:
                save_to_tensor(result, args.out_dir, file_name)
                save_to_json(result, args.out_dir, file_name)

                del result
                

    return True

def save_to_tensor(result, out_dir, file_name):
    result["semantic_masks"] = torch.tensor(result["semantic_masks"])
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
    results = main(0, args)