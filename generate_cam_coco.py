import torch
import clip
from PIL import Image
from pytorch_grad_cam import GradCAM
import cv2
import argparse
from data.dataset import FSSDataset
import os

COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def get_cam_from_alldata(clip_model, preprocess, split='train', d0=None, d1=None, d2=None, d3=None,
                               datapath=None, campath=None, fold=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Extract metadata from dataloaders
    d0_metadata = d0.dataset.img_metadata_classwise
    d1_metadata = d1.dataset.img_metadata_classwise
    d2_metadata = d2.dataset.img_metadata_classwise
    d3_metadata = d3.dataset.img_metadata_classwise
    dd = [d0_metadata, d1_metadata, d2_metadata, d3_metadata]
    
    dataset_all = {}
    
    # Reorganize based on split
    if split == 'train':
        for ii in range(80):
            index = ii % 4 + 1
            if ii % 4 == 3:
                index = 0
            dataset_all[ii] = dd[index][ii]
    else:  # val split
        for ii in range(80):
            index = ii % 4
            dataset_all[ii] = dd[index][ii]
    
    del d0_metadata, d1_metadata, d2_metadata, d3_metadata, dd

    # Create split-specific folder
    split_folder = f"{split}_fold_{fold}"  # e.g., "train_fold_0"
    split_campath = os.path.join(campath, split_folder)
    os.makedirs(split_campath, exist_ok=True)
    
    print(f"Processing {split} fold {fold}")
    print(f"Saving CAM files to: {split_campath}")

    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in COCO_CLASSES]).to(device)
    
    total_images = sum(len(dataset_all[cls_id]) for cls_id in range(80))
    processed_count = 0
    
    for cls_id in range(80):
        L = len(dataset_all[cls_id])
        class_name = COCO_CLASSES[cls_id]
        
        print(f"Processing class {cls_id}: {class_name} ({L} images)")
        
        for ll in range(L):
            img_path = os.path.join(datapath, dataset_all[cls_id][ll])
            img = Image.open(img_path)
            img_input = preprocess(img).unsqueeze(0).to(device)

            # Generate CAM
            clip_model.get_text_features(text_inputs)
            target_layers = [clip_model.visual.layer4[-1]]
            input_tensor = img_input
            cam = GradCAM(model=clip_model, target_layers=target_layers, use_cuda=True)
            target_category = cls_id
            grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
            grayscale_cam = grayscale_cam[0, :]
            grayscale_cam = cv2.resize(grayscale_cam, (50, 50))
            grayscale_cam = torch.from_numpy(grayscale_cam)
            
            # Create filename from image path
            img_filename = os.path.basename(dataset_all[cls_id][ll])
            img_name = os.path.splitext(img_filename)[0]
            
            # Save in split-specific folder
            save_path = os.path.join(split_campath, f"{img_name}--{cls_id}.pt")
            torch.save(grayscale_cam, save_path)
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count}/{total_images} images")
    
    print(f"Completed {split} fold {fold} - Saved {total_images} CAM files to {split_campath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMR')
    parser.add_argument('--imgpath', type=str, default='/dataA/iedA/Shahroz/PICLIP/data/MSCOCO2014/')
    parser.add_argument('--campath', type=str, default='/dataA/iedA/Shahroz/IMR-HSNet/Datasets_HSN/CAM_COCO/')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'all'], default='all', 
                       help='Which split to process: train, val, or all')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_clip, preprocess = clip.load('RN50', device, jit=False)
    FSSDataset.initialize(img_size=400, datapath='/dataA/iedA/Shahroz/IMR-HSNet/Datasets_HSN/', use_original_imgsize=False)

    # Create base directory
    os.makedirs(args.campath, exist_ok=True)
    
    print("Starting CAM generation for COCO dataset...")
    
    if args.split in ['train', 'all']:
        print("\n" + "="*50)
        print("Processing COCO TRAIN split...")
        print("="*50)
        
        # Create dataloaders for train split
        dataloader_train_0 = FSSDataset.build_dataloader('coco', 1, 0, 0, 'train', 1)
        dataloader_train_1 = FSSDataset.build_dataloader('coco', 1, 0, 1, 'train', 1)
        dataloader_train_2 = FSSDataset.build_dataloader('coco', 1, 0, 2, 'train', 1)
        dataloader_train_3 = FSSDataset.build_dataloader('coco', 1, 0, 3, 'train', 1)
        
        # Process train split (COCO uses 4-fold cross-validation, so we process all folds together)
        get_cam_from_alldata(model_clip, preprocess, split='train',
                           d0=dataloader_train_0, d1=dataloader_train_1,
                           d2=dataloader_train_2, d3=dataloader_train_3,
                           datapath=args.imgpath, campath=args.campath, fold=0)
    
    if args.split in ['val', 'all']:
        print("\n" + "="*50)
        print("Processing COCO VALIDATION split...")
        print("="*50)
        
        # Create dataloaders for val split
        dataloader_val_0 = FSSDataset.build_dataloader('coco', 1, 0, 0, 'val', 1)
        dataloader_val_1 = FSSDataset.build_dataloader('coco', 1, 0, 1, 'val', 1)
        dataloader_val_2 = FSSDataset.build_dataloader('coco', 1, 0, 2, 'val', 1)
        dataloader_val_3 = FSSDataset.build_dataloader('coco', 1, 0, 3, 'val', 1)
        
        # Process val split
        get_cam_from_alldata(model_clip, preprocess, split='val',
                           d0=dataloader_val_0, d1=dataloader_val_1,
                           d2=dataloader_val_2, d3=dataloader_val_3,
                           datapath=args.imgpath, campath=args.campath, fold=0)

    print("\n" + "="*50)
    print("CAM generation completed successfully!")
    print("="*50)
    print(f"CAM files saved in: {args.campath}")
    print("\nFolder structure:")
    print("CAM_COCO/")
    if args.split in ['train', 'all']:
        print("├── train_fold_0/")
    if args.split in ['val', 'all']:
        print("├── val_fold_0/")
    print("\nEach folder contains CAM files named as: image_name--class_id.pt")