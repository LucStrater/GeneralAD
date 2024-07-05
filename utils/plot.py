import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def unnormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # reverses the normalization for visualization
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def plot_images(img, file_name, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # unnormalize
    img_unnorm = unnormalize(img.clone(), mean, std)

    # convert tensor to img
    img_np = tF.to_pil_image(img_unnorm).convert("RGB")

    # plot
    plt.imshow(img_np)
    plt.axis('off')  
    plt.grid(False)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_with_heatmap_and_original(images, scores, file_name, patch_size=16, image_size=224):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    for img, score in zip(images, scores):
        # unnormalize
        img_unnorm = unnormalize(img.clone(), mean, std)

        # numpy and grayscale for plotting
        img_np = tF.to_pil_image(img_unnorm).convert("RGB")
        img_gray_np = tF.to_pil_image(img_unnorm).convert("L")

        # reshape patches to 14x14
        patch_scores = score.reshape((image_size // patch_size, image_size // patch_size))

        # normalize scores
        normalized_scores = (patch_scores - patch_scores.min()) / (patch_scores.max() - patch_scores.min())

        # generate heatmap
        heatmap = F.interpolate(
            normalized_scores.unsqueeze(0).unsqueeze(0),
            size=image_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)

        # numpy
        heatmap_np = heatmap.cpu().numpy()

        # plotting
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img_np)
        axs[0].axis('off')
        axs[1].imshow(img_gray_np, cmap='gray')
        axs[1].imshow(heatmap_np, cmap='hot', alpha=0.8)
        axs[1].axis('off')

        # plot
        plt.savefig(file_name)
        plt.close()