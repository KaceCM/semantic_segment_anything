import os
import argparse
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from pipeline import semantic_segment_anything_inference, eval_pipeline, img_load
from configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL
from configs.cityscapes_id2label import CONFIG as CONFIG_CITYSCAPES_ID2LABEL

import torch.distributed as dist
import torch.multiprocessing as mp
from dotenv import load_dotenv, find_dotenv
print(load_dotenv(find_dotenv(".env")))

SSA_DATASET = os.environ.get("SSA_DATASET")
SSA_MODEL = os.environ.get("SSA_MODEL")
SSA_WORLD_SIZE = int(os.environ.get("SSA_WORLD_SIZE"))

SSA_CKPT_PATH = os.environ.get("SSA_CKPT_PATH")
SSA_OUT_DIR = os.environ.get("SSA_OUT_DIR")
SSA_DATA_DIR = os.environ.get("SSA_DATA_DIR")
SSA_SAVE_IMAGE = bool(os.environ.get("SSA_SAVE_IMAGE"))

print('[Environment loaded] SSA_DATASET:', SSA_DATASET, type(SSA_DATASET))
print('[Environment loaded] SSA_MODEL:', SSA_MODEL, type(SSA_MODEL))
print('[Environment loaded] SSA_WORLD_SIZE:', SSA_WORLD_SIZE, type(SSA_WORLD_SIZE))
print('[Environment loaded] SSA_CKPT_PATH:', SSA_CKPT_PATH, type(SSA_CKPT_PATH))
print('[Environment loaded] SSA_OUT_DIR:', SSA_OUT_DIR, type(SSA_OUT_DIR))
print('[Environment loaded] SSA_DATA_DIR:', SSA_DATA_DIR, type(SSA_DATA_DIR))
print('[Environment loaded] SSA_SAVE_IMAGE:', SSA_SAVE_IMAGE, type(SSA_SAVE_IMAGE))





def main(rank):
    dist.init_process_group("nccl", rank=rank, world_size=SSA_WORLD_SIZE)
    
    sam = sam_model_registry["vit_h"](checkpoint=SSA_CKPT_PATH).to(rank)

    mask_branch_model = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=128 if SSA_DATASET == 'foggy_driving' else 64,
        # Foggy driving (zero-shot evaluate) is more challenging than other dataset, so we use a larger points_per_side
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
        output_mode='coco_rle',
    )
    print('[Model loaded] Mask branch (SAM) is loaded.')
    # yoo can add your own semantic branch here, and modify the following code
    if SSA_MODEL == 'oneformer':
        from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        if SSA_DATASET == 'ade20k':
            semantic_branch_processor = OneFormerProcessor.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_large")
            semantic_branch_model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_large").to(rank)
        else:
            raise NotImplementedError()
    elif SSA_MODEL == 'segformer':
        from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
        if SSA_DATASET == 'ade20k':
            semantic_branch_processor = SegformerFeatureExtractor.from_pretrained(
                "nvidia/segformer-b5-finetuned-ade-640-640")
            semantic_branch_model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b5-finetuned-ade-640-640").to(rank)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    print('[Model loaded] Semantic branch (your own segmentor) is loaded.')
    if SSA_DATASET == 'ade20k':
        filenames = [fn_.replace('.jpg', '') for fn_ in os.listdir(SSA_DATA_DIR) if '.jpg' in fn_]
    local_filenames = filenames[(len(filenames) // SSA_WORLD_SIZE + 1) * rank : (len(filenames) // SSA_WORLD_SIZE + 1) * (rank + 1)]
    print('[Image name loaded] get image filename list.')
    print('[SSA start] model inference starts.')

    for i, file_name in enumerate(local_filenames):
        print('[Runing] ', i, '/', len(local_filenames), ' ', file_name, ' on rank ', rank, '/', SSA_WORLD_SIZE)
        img = img_load(SSA_DATA_DIR, file_name, SSA_DATASET)
        if SSA_DATASET == 'ade20k':
            id2label = CONFIG_ADE20K_ID2LABEL
        else:
            raise NotImplementedError()
        with torch.no_grad():
            semantic_segment_anything_inference(file_name, SSA_OUT_DIR, rank, img=img, save_img=SSA_SAVE_IMAGE,
                                   semantic_branch_processor=semantic_branch_processor,
                                   semantic_branch_model=semantic_branch_model,
                                   mask_branch_model=mask_branch_model,
                                   dataset=SSA_DATASET,
                                   id2label=id2label,
                                   model=SSA_MODEL)
        # torch.cuda.empty_cache()

if __name__ == '__main__':
    main(0)