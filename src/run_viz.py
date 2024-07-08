import os
import torch
import sys

from pytorch_lightning import seed_everything
import torch.nn.functional as F
import torchvision.transforms.functional as tF

from .general_ad import General_AD
from .load_data import prepare_loader

import random
import subprocess
import numpy as np
import csv
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt

def unnormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # reverses the normalization for visualization
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def viz_segmentation(args):

    # device
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit()

    device = torch.device("cuda:0")
    print("Device:", device)

    # data 
    _, test_loader = prepare_loader(image_size=args.image_size,
                                            path=args.data_dir,
                                            dataset_name=args.dataset_name, 
                                            class_name=args.normal_class, 
                                            batch_size=args.batch_size, 
                                            test_batch_size=args.test_batch_size, 
                                            num_workers=args.num_workers, 
                                            seed=args.seed,
                                            shots=args.shots)

    # model
    model = General_AD.load_from_checkpoint(args.checkpoint_dir).to(device)
    
    i = 0
    for images, _ in test_loader:
        scores = model(images.to(device))

        batch_size, num_patches = scores.shape
        image_size = images.shape[-1]
        patches_per_side = int(np.sqrt(num_patches))

        for b in range(batch_size):
            patch_scores = scores[b].reshape((patches_per_side, patches_per_side))
            scores_interpolated = F.interpolate(patch_scores.unsqueeze(0).unsqueeze(0),
                                                size=image_size,
                                                mode='bilinear',
                                                align_corners=False
                                                ).squeeze(0).squeeze(0)

            segmentations = gaussian_filter(scores_interpolated.cpu().detach().numpy(), sigma=args.smoothing_sigma, radius=args.smoothing_radius)  

            seg_min = segmentations.min()
            seg_max = segmentations.max()
            normalized_segmentations = (segmentations - seg_min) / (seg_max - seg_min)

            img_unnorm = unnormalize(images[b].clone())
            img_np = tF.to_pil_image(img_unnorm).convert("RGB")

            seg_img_rgba = np.zeros((image_size, image_size, 4), dtype=np.uint8)  
            seg_threshold = 0.5
            high_values_index = normalized_segmentations > seg_threshold  
            seg_img_rgba[..., 3] = (normalized_segmentations * 255).astype(np.uint8) 
            seg_img_rgba[high_values_index, :3] = plt.get_cmap('hot')((normalized_segmentations[high_values_index] - seg_threshold) / (1 - seg_threshold))[:, :3] * 255  # Apply colormap and threshold

            # Plotting
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(img_np)  
            ax.imshow(seg_img_rgba, cmap='hot', alpha=0.5)
            ax.axis('off')
            
            plt.savefig(os.path.join('images', f'seg{i}_{b}'), bbox_inches='tight', pad_inches=0)
            plt.close()

            # Plotting
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(img_np)  
            ax.axis('off')
            
            plt.savefig(os.path.join('images', f'img{i}_{b}'), bbox_inches='tight', pad_inches=0)
            plt.close()

        i = i + 1
        if i == 4:
            sys.exit()
    print("Visualization complete.")
    return