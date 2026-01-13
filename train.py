""" Hypercorrelation Squeeze training (validation) code """
import argparse
import pdb

import torch.optim as optim
import torch.nn as nn
import torch
import os

from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from model.hsnet_imr import HypercorrSqueezeNetwork_imr

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["HF_HOME"] = "/dataA/iedA/.cache/huggingface"

def train(epoch, model, dataloader, optimizer, training, stage):
    r""" Train HSNet """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        # 1. Hypercorrelation Squeeze Networks forward pass
        batch = utils.to_cuda(batch)
        logit_mask_q, logit_mask_s, losses = model(
            query_img=batch['query_img'], support_img=batch['support_imgs'].squeeze(1),
            support_cam=batch['support_cams'].squeeze(1), query_cam=batch['query_cam'], stage=stage,
            query_mask=batch['query_mask'], support_mask=batch['support_masks'].squeeze(1))
        pred_mask_q = logit_mask_q.argmax(dim=1)

        # 2. Compute loss & update model parameters
        loss = losses.mean()
        if training:
            optimizer.zero_grad()
            loss.backward()
            # Add gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask_q, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Hypercorrelation Squeeze Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='/dataA/iedA/Shahroz/PICLIP/data/VOCdevkit2012')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logroot', type=str, default='logs')
    parser.add_argument('--bsz', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--niter', type=int, default=400)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--stage', type=int, default=2)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--traincampath', type=str, default='/dataA/iedA/Shahroz/PICLIP/data/VOCdevkit2012/VOC2012/CAM_VOC_Train/train_fold_0/')
    parser.add_argument('--valcampath', type=str, default='/dataA/iedA/Shahroz/PICLIP/data/VOCdevkit2012/VOC2012/CAM_VOC_Val/val_fold_0/')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for regularization')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--resume', action='store_true', help='Resume training from best.pt')
    
    args = parser.parse_args()
    
    # Create log directory based on dataset name and fold
    log_dir = os.path.join(args.logroot, f"{args.benchmark}_fold_{args.fold}")
    
    # Check if there's a file with the same name as our directory
    if os.path.exists(log_dir) and os.path.isfile(log_dir):
        # This is a file, not a directory - we need to handle this
        print(f"Warning: {log_dir} exists as a file, not a directory.")
        print(f"Renaming the file to {log_dir}.backup")
        os.rename(log_dir, log_dir + ".backup")
    
    # Now create the directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Set logfile path and logpath for Logger
    args.logfile = os.path.join(log_dir, f"{args.benchmark}_fold_{args.fold}.log")
    args.logpath = log_dir  # For model saving
    
    # Initialize logger with the logfile path
    Logger.initialize(args, training=True)
    
    assert args.bsz % torch.cuda.device_count() == 0

    # Model initialization
    model = HypercorrSqueezeNetwork_imr(args.backbone, False)
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Improved optimizer setup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        patience=5, 
        factor=0.5,
        verbose=True
    )

    # Resume functionality
    start_epoch = 0
    best_val_miou = float('-inf')
    best_epoch = 0
    
    if args.resume:
        # Define the checkpoint path based on dataset and fold
        checkpoint_path = os.path.join(log_dir, 'best.pt')
        if os.path.exists(checkpoint_path):
            Logger.info(f'Resuming training from {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if available
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            start_epoch = checkpoint['epoch'] + 1
            best_val_miou = checkpoint['best_val_miou']
            best_epoch = checkpoint['best_epoch']
            
            Logger.info(f'Resumed from epoch {start_epoch} with best_val_miou: {best_val_miou:.2f}')
        else:
            Logger.warning(f'Checkpoint not found at {checkpoint_path}. Starting from scratch.')
    else:
        Logger.info('Starting training from scratch.')

    Evaluator.initialize()

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn',
                                                 cam_train_path=args.traincampath, cam_val_path=args.valcampath)
    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val',
                                                 cam_train_path=args.traincampath, cam_val_path=args.valcampath)

    # Train HSNet with improved training loop
    no_improvement_count = 0
    
    Logger.info('Starting training with improved optimizer configuration...')
    Logger.info(f'Optimizer: AdamW, LR: {args.lr}, Weight Decay: {args.weight_decay}')
    Logger.info(f'Early stopping patience: {args.patience} epochs')

    for epoch in range(start_epoch, args.niter):
        # Training phase
        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True, stage=args.stage)
        
        # Validation phase
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False, stage=args.stage)
        
        # Update learning rate based on validation performance
        scheduler.step(val_miou)
        
        # Save best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_epoch = epoch
            no_improvement_count = 0
            
            # Save full checkpoint
            checkpoint_path = os.path.join(log_dir, 'best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_miou': best_val_miou,
                'best_epoch': best_epoch,
                'args': vars(args)
            }, checkpoint_path)
            
            Logger.info(f'*** New best model saved to {checkpoint_path} @epoch {epoch}: val_mIoU = {val_miou:.2f} ***')
        else:
            no_improvement_count += 1
            Logger.info(f'No improvement for {no_improvement_count} epochs (best: {best_val_miou:.2f} @epoch {best_epoch})')
        
        # Early stopping check
        if no_improvement_count >= args.patience:
            Logger.info(f'Early stopping triggered after {epoch + 1} epochs.')
            Logger.info(f'Best model: epoch {best_epoch} with val_mIoU: {best_val_miou:.2f}')
            break
        
        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.add_scalar('data/learning_rate', current_lr, epoch)
        Logger.tbd_writer.flush()
        
        # Print epoch summary
        Logger.info(f'Epoch {epoch:02d} | LR: {current_lr:.2e} | '
                   f'Trn: L={trn_loss:.3f}, mIoU={trn_miou:.2f} | '
                   f'Val: L={val_loss:.3f}, mIoU={val_miou:.2f}')

    Logger.tbd_writer.close()
    Logger.info('=' * 60)
    Logger.info(f'Training completed! Best validation mIoU: {best_val_miou:.2f} at epoch {best_epoch}')
    Logger.info('=' * 60)