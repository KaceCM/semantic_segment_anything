import os
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import pycocotools.mask as maskUtils
import matplotlib.pyplot as plt
from assets.utils import printr


def oneformer_ade20k_segmentation(image, oneformer_ade20k_processor, oneformer_ade20k_model, rank):
    inputs = oneformer_ade20k_processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(rank)
    outputs = oneformer_ade20k_model(**inputs)
    predicted_semantic_map = oneformer_ade20k_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]])[0]
    return predicted_semantic_map

def img_load(data_path, filename):
    img = Image.open(os.path.join(data_path, filename)).convert('RGB')
    img_ndarray = np.array(img)
    return img_ndarray

def semantic_segment_anything_inference(filename, output_path, rank, img=None, save_img=False,
                                 semantic_branch_processor=None,
                                 semantic_branch_model=None,
                                 mask_branch_model=None,
                                 id2label=None,):

    printr('[Inference] Starting semantic segmentation inference...')

    printr('Generating masks using AutomaticMaskGeneration')
    anns = {'annotations': mask_branch_model.generate(img)}
    printr('Masks generated using AutomaticMaskGeneration')
    
    h, w, _ = img.shape
    class_names = []
    masks_list = []  # List to store all masks

    printr('[Inference] Starting OneFormer/SegFormer inference...')
    class_ids = oneformer_ade20k_segmentation(Image.fromarray(img), semantic_branch_processor,semantic_branch_model, rank)
    printr('[Inference done] OneFormer/SegFormer inference is done.')


    printr('[Inference] Starting SegFormer inference...')
    num_classes = int(class_ids.max().item()) + 1
    cmap = plt.get_cmap('tab20', num_classes)
    class_ids_np = class_ids.cpu().numpy().astype(np.uint8)
    color_class_ids = cmap(class_ids_np / (num_classes if num_classes > 0 else 1))[:, :, :3]  # Drop alpha
    color_class_ids = (color_class_ids * 255).astype(np.uint8)
    out_path_raw = os.path.join(output_path, filename + '_oneformer_raw.png')
    Image.fromarray(color_class_ids).save(out_path_raw)
    printr(f'[Save] OneFormer/SegFormer raw output image saved to: {out_path_raw}')


    printr('Starting for loop over annotations to generate semantic masks...')
    semantc_mask = class_ids.clone()
    anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)
    for ann in tqdm(anns['annotations']):
        valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
        masks_list.append(valid_mask)  # Store each mask
        # get the class ids of the valid pixels
        propose_classes_ids = class_ids[valid_mask]
        num_class_proposals = len(torch.unique(propose_classes_ids))
        if num_class_proposals == 1:
            semantc_mask[valid_mask] = propose_classes_ids[0]
            ann['class_name'] = id2label['id2label'][str(propose_classes_ids[0].item())]
            ann['class_proposals'] = id2label['id2label'][str(propose_classes_ids[0].item())]
            class_names.append(ann['class_name'])
            continue
        top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
        top_1_propose_class_names = [id2label['id2label'][str(class_id.item())] for class_id in top_1_propose_class_ids]

        semantc_mask[valid_mask] = top_1_propose_class_ids
        ann['class_name'] = top_1_propose_class_names[0]
        ann['class_proposals'] = top_1_propose_class_names[0]
        class_names.append(ann['class_name'])

        del valid_mask
        del propose_classes_ids
        del num_class_proposals
        del top_1_propose_class_ids
        del top_1_propose_class_names
    
    sematic_class_in_img = torch.unique(semantc_mask)
    semantic_bitmasks, semantic_class_names = [], []
    printr('Semantic masks generated...')

    # semantic prediction
    printr('[Inference] Starting semantic prediction...')
    anns['semantic_mask'] = {}
    for i in range(len(sematic_class_in_img)):
        class_name = id2label['id2label'][str(sematic_class_in_img[i].item())]
        class_mask = semantc_mask == sematic_class_in_img[i]
        class_mask = class_mask.cpu().numpy().astype(np.uint8)
        semantic_class_names.append(class_name)
        semantic_bitmasks.append(class_mask)
        anns['semantic_mask'][str(sematic_class_in_img[i].item())] = maskUtils.encode(np.array((semantc_mask == sematic_class_in_img[i]).cpu().numpy(), order='F', dtype=np.uint8))
        anns['semantic_mask'][str(sematic_class_in_img[i].item())]['counts'] = anns['semantic_mask'][str(sematic_class_in_img[i].item())]['counts'].decode('utf-8')
    printr('[Inference done] Semantic prediction is done.')

    printr('[Save] Saving semantic masks and annotations...')
    if save_img:
        # Save the semantic mask as a color image
        
        # Create a color map for the number of classes
        num_classes = len(semantic_class_names)
        cmap = plt.get_cmap('tab20', num_classes)
        # Convert the semantic mask to a numpy array
        semantc_mask_np = semantc_mask.cpu().numpy().astype(np.uint8)
        # Create a color image using the class indices
        color_mask = cmap(semantc_mask_np / (num_classes if num_classes > 0 else 1))[:, :, :3]  # Drop alpha
        color_mask = (color_mask * 255).astype(np.uint8)
        # Save the color mask as PNG
        
        out_path = os.path.join(output_path, filename + '_semantic_mask.png')
        Image.fromarray(color_mask).save(out_path)
        print(f'[Save] SegFormer/OneFormer semantic mask image saved to: {out_path}')
        # imshow_det_bboxes(img,
        #                     bboxes=None,
        #                     labels=np.arange(len(sematic_class_in_img)),
        #                     segms=np.stack(semantic_bitmasks),
        #                     class_names=semantic_class_names,
        #                     font_size=25,
        #                     show=False,
        #                     out_file=os.path.join(output_path, filename + '_semantic.png'))
        print("IMSHOW DET BBOXES")
        print('[Save] save SSA prediction: ', os.path.join(output_path, filename + '_semantic.png'))
        # saving image :

    # Store the results before cleanup
    result = {
        'instance_masks': masks_list,
        'semantic_masks': semantic_bitmasks,
        'class_names': class_names,
        'semantic_class_names': semantic_class_names
    }

    del img
    del anns
    del class_ids
    del semantc_mask
    del class_names
    del semantic_bitmasks
    del semantic_class_names
    printr('[Inference done] Semantic segmentation inference is done.')
    return result  # Return the dictionary containing all masks and their associated information