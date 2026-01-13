import torch
import clip
from PIL import Image
from pytorch_grad_cam import GradCAM
import cv2
import argparse
from data.dataset import FSSDataset
import os
import pdb

PASCAL_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']


def get_cam_from_alldata(clip_model, preprocess, d=None, datapath=None, campath=None, split_name="", fold=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_all = d.dataset.img_metadata
    L = len(dataset_all)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in PASCAL_CLASSES]).to(device)
    
    # Create organized folder structure
    split_folder = f"{split_name}_fold_{fold}"  # e.g., "train_fold_0"
    split_campath = os.path.join(campath, split_folder)
    os.makedirs(split_campath, exist_ok=True)
    
    print(f"Processing {split_name} fold {fold} - {L} images")
    print(f"Saving CAM files to: {split_campath}")
    
    for ll in range(L):
        img_path = os.path.join(datapath, dataset_all[ll][0] + '.jpg')
        img = Image.open(img_path)
        img_input = preprocess(img).unsqueeze(0).to(device)
        class_name_id = dataset_all[ll][1]
        
        # Get text features
        clip_model.get_text_features(text_inputs)
        
        # Setup GradCAM
        target_layers = [clip_model.visual.layer4[-1]]
        input_tensor = img_input
        cam = GradCAM(model=clip_model, target_layers=target_layers, use_cuda=True)
        target_category = class_name_id
        
        # Generate CAM
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        grayscale_cam = cv2.resize(grayscale_cam, (50, 50))
        grayscale_cam = torch.from_numpy(grayscale_cam)
        
        # Save in split-specific folder
        save_path = os.path.join(split_campath, dataset_all[ll][0] + '--' + str(class_name_id) + '.pt')
        torch.save(grayscale_cam, save_path)
        
        if ll % 100 == 0:  # Print progress every 100 images
            print(f'Processed {ll}/{L} images - CAM saved: {save_path}')
    
    print(f"Completed {split_name} fold {fold} - Saved {L} CAM files to {split_campath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMR')
    parser.add_argument('--imgpath', type=str, default='/dataA/iedA/Shahroz/PICLIP/data/VOCdevkit2012/VOC2012/JPEGImages/')
    parser.add_argument('--traincampath', type=str, default='/dataA/iedA/Shahroz/PICLIP/data/VOCdevkit2012/VOC2012/CAM_VOC_Train/')
    parser.add_argument('--valcampath', type=str, default='/dataA/iedA/Shahroz/PICLIP/data/VOCdevkit2012/VOC2012/CAM_VOC_Val/')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_clip, preprocess = clip.load('RN50', device, jit=False)
    FSSDataset.initialize(img_size=400, datapath='/dataA/iedA/Shahroz/PICLIP/data/VOCdevkit2012/VOC2012', use_original_imgsize=False)

    print("Starting CAM generation for Pascal VOC dataset...")
    
    # Create base directories
    os.makedirs(args.traincampath, exist_ok=True)
    os.makedirs(args.valcampath, exist_ok=True)

    # Train splits - 4 folds
    print("\n" + "="*50)
    print("Processing TRAIN splits...")
    print("="*50)
    
    dataloader_train_0 = FSSDataset.build_dataloader('pascal', 1, 0, 0, 'train', 1)
    dataloader_train_1 = FSSDataset.build_dataloader('pascal', 1, 0, 1, 'train', 1)
    dataloader_train_2 = FSSDataset.build_dataloader('pascal', 1, 0, 2, 'train', 1)
    dataloader_train_3 = FSSDataset.build_dataloader('pascal', 1, 0, 3, 'train', 1)

    get_cam_from_alldata(model_clip, preprocess, d=dataloader_train_0, 
                        datapath=args.imgpath, campath=args.traincampath, 
                        split_name="train", fold=0)
    get_cam_from_alldata(model_clip, preprocess, d=dataloader_train_1, 
                        datapath=args.imgpath, campath=args.traincampath, 
                        split_name="train", fold=1)
    get_cam_from_alldata(model_clip, preprocess, d=dataloader_train_2, 
                        datapath=args.imgpath, campath=args.traincampath, 
                        split_name="train", fold=2)
    get_cam_from_alldata(model_clip, preprocess, d=dataloader_train_3, 
                        datapath=args.imgpath, campath=args.traincampath, 
                        split_name="train", fold=3)

    # Val splits - 4 folds
    print("\n" + "="*50)
    print("Processing VALIDATION splits...")
    print("="*50)
    
    dataloader_val_0 = FSSDataset.build_dataloader('pascal', 1, 0, 0, 'val', 1)
    dataloader_val_1 = FSSDataset.build_dataloader('pascal', 1, 0, 1, 'val', 1)
    dataloader_val_2 = FSSDataset.build_dataloader('pascal', 1, 0, 2, 'val', 1)
    dataloader_val_3 = FSSDataset.build_dataloader('pascal', 1, 0, 3, 'val', 1)

    get_cam_from_alldata(model_clip, preprocess, d=dataloader_val_0, 
                        datapath=args.imgpath, campath=args.valcampath, 
                        split_name="val", fold=0)
    get_cam_from_alldata(model_clip, preprocess, d=dataloader_val_1, 
                        datapath=args.imgpath, campath=args.valcampath, 
                        split_name="val", fold=1)
    get_cam_from_alldata(model_clip, preprocess, d=dataloader_val_2, 
                        datapath=args.imgpath, campath=args.valcampath, 
                        split_name="val", fold=2)
    get_cam_from_alldata(model_clip, preprocess, d=dataloader_val_3, 
                        datapath=args.imgpath, campath=args.valcampath, 
                        split_name="val", fold=3)

    print("\n" + "="*50)
    print("CAM generation completed successfully!")
    print("="*50)
    print(f"Train CAMs saved in: {args.traincampath}")
    print(f"Val CAMs saved in: {args.valcampath}")
    print("\nFolder structure:")
    print("CAM_VOC_Train/")
    print("├── train_fold_0/")
    print("├── train_fold_1/")
    print("├── train_fold_2/")
    print("└── train_fold_3/")
    print("CAM_VOC_Val/")
    print("├── val_fold_0/")
    print("├── val_fold_1/")
    print("├── val_fold_2/")
    print("└── val_fold_3/")