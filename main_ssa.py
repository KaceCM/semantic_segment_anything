import os
import argparse
import torch
from assets.utils import printr, save_to_tensor, save_to_json, prepare_image


from pipeline_ssa import (generate_random_masks,
                          generate_oneformer_masks,
                          save_raw_segformer_masks,
                          generate_semantic_masks,
                          generate_semantic_prediction,
                          save_semantic_masks,
                          load_mask_generator,
                          load_semantic_branch)


from assets.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
printr(f"Using device: {DEVICE}")

def parse_args():
    parser = argparse.ArgumentParser(description='Semantically segment anything.')
    parser.add_argument('--img_path', default=None, help='specify the root path of images and masks')
    parser.add_argument('--data_path', default=None, help='specify the root path of images and masks')
    parser.add_argument('--save_masks', default=True, action='store_true', help='whether to save masks')
    parser.add_argument('--ckpt_path', default='ckp/sam_vit_h_4b8939.pth', help='specify the root path of SAM checkpoint')
    parser.add_argument('--out_dir', help='the dir to save semantic annotations')
    parser.add_argument('--save_img', default=False, action='store_true', help='whether to save annotated images')
    args = parser.parse_args()
    return args
    







def main(args):
    printr('STARTING MAIN FUNCTION')
    
    
    printr(f'[Image path] Using single image: {args.img_path}')
    img, image_name_no_ext = prepare_image(args)
    id2label = CONFIG_ADE20K_ID2LABEL




    printr('[Inference] Starting semantic segmentation inference...')
    with torch.no_grad():
        
        mask_branch_model = load_mask_generator(args, DEVICE=DEVICE)
        printr('[Inference] Generating masks...')
        anns = generate_random_masks(mask_branch_model, img)
        del mask_branch_model  # Free memory after generating masks
        printr('[Inference done] Masks are generated.')

        semantic_branch_processor, semantic_branch_model = load_semantic_branch(args, DEVICE=DEVICE)
        printr('[Inference] Starting semantic segmentation inference...')
        class_ids = generate_oneformer_masks(semantic_branch_processor, semantic_branch_model, img, DEVICE)
        printr('[Inference done] Semantic segmentation inference is done.')
        del semantic_branch_model, semantic_branch_processor

        save_raw_segformer_masks(image_name_no_ext, args.out_dir, class_ids)

        semantic_class_in_img, semantic_class_names, semantic_bitmasks, semantic_mask, class_names, masks_list = generate_semantic_masks(anns, class_ids, id2label)
        anns, semantic_class_names, semantic_bitmasks = generate_semantic_prediction(anns, semantic_class_in_img, semantic_mask, id2label, semantic_bitmasks, semantic_class_names)

        save_semantic_masks(output_path=args.out_dir, filename=image_name_no_ext, semantic_mask=semantic_mask, semantic_class_names=semantic_class_names)
        
        printr('[Inference done] Semantic segmentation inference is done.')
        
        result = {
        'instance_masks': masks_list,
        'semantic_masks': semantic_bitmasks,
        'class_names': class_names,
        'semantic_class_names': semantic_class_names
    }

        if args.save_masks:
            printr('[Saving results] Saving results...')
            save_to_tensor(result, args.out_dir, image_name_no_ext)
            save_to_json(result, args.out_dir, image_name_no_ext)
            printr('[Saving results done] Results are saved.')
            del result

    del img
    del anns
    del class_ids
    del semantic_mask
    del class_names
    del semantic_bitmasks
    del semantic_class_names
    del semantic_class_in_img
    del masks_list
    del id2label

            

    printr('MAIN FUNCTION DONE')
    return True



if __name__ == '__main__':
    args = parse_args()
    if not args.img_path and not args.data_path:
        raise ValueError('Please specify either --img_path for a single image or --data_path for a directory of images.')
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    results = main(args)