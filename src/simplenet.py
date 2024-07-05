# basic
import numpy as np
import timm
import re
import copy
import sys
import math
import random
import os
import wget

# torch
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim

# image processing
from sklearn.metrics import roc_auc_score, auc
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from cv2 import getStructuringElement, MORPH_RECT, dilate

class FeatureExtractor(nn.Module):
    def __init__(self, hf_path, layer_indices, patch_size, image_size, device):
        super(FeatureExtractor, self).__init__()
        self.layer_indices = layer_indices
        self.patch_size = patch_size
        self.num_patches = (math.ceil(image_size / 2**(layer_indices[0]+1)))**2

        # get pretrained model
        self.pretrained_model = timm.create_model(hf_path, pretrained=True, num_classes=0).to(device)

        # freeze the pretrained model's parameters
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # embedding dimension
        feature_layer_size = {1: 256, 2: 512, 3: 1024, 4: 2048}
        self.embed_dim = 0
        for layer_idx in self.layer_indices:
            self.embed_dim += feature_layer_size[layer_idx]

    def forward(self, x):
        # initial input processing
        x = self.pretrained_model.conv1(x)
        x = self.pretrained_model.bn1(x)
        x = self.pretrained_model.act1(x)
        x = self.pretrained_model.maxpool(x)

        out = []
        layers = [self.pretrained_model.layer1, self.pretrained_model.layer2, self.pretrained_model.layer3, self.pretrained_model.layer4]

        # iterating through the layers up to last layer to extract from
        for idx, layer in enumerate(layers, start=1):
            # x of shape (batch, feature_size, channel_dim_h, channel_dim_w)
            x = layer(x)
            if idx == self.layer_indices[0]:
                patch_dim = x.shape[-1]
            if idx in self.layer_indices:
                # interpolate so features of different layers have same amount of patches
                if x.shape[-1] != patch_dim:
                    x = F.interpolate(x, size=(patch_dim, patch_dim), mode='bilinear', align_corners=False)

                padding = int((self.patch_size - 1) / 2)
                # unfold to extract patches 
                unfold = F.unfold(x, kernel_size=self.patch_size, stride=1, padding=padding, dilation=1)                # [batch, feature_size*patch_size*patch_size, num_patches]
                unfolded_features = unfold.reshape(*x.shape[:2], self.patch_size, self.patch_size, -1)                  # [batch, feature_size, patch_size, patch_size, num_patches]
                unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)                                            # [batch, num_patches, feature_size, patch_size, patch_size]
                
                # average pooling from (batch, num_patches, feature_size, patch_size, patch_size) to (batch, num_patches, feature_size)
                pooled_features = F.adaptive_avg_pool2d(unfolded_features, (1, 1))
                features_layer = pooled_features.view(pooled_features.size(0), pooled_features.size(1), -1)

                out.append(features_layer)
            if idx == max(self.layer_indices):
                break

        # cat the features of the layers to extract
        features = torch.cat(out, dim=-1)

        return features

class FeatureExtractor_ViT(nn.Module):
    def __init__(self, hf_path, layer_indices, image_size, device, pool_size=3):
        super(FeatureExtractor_ViT, self).__init__()
        self.layer_indices = layer_indices
        self.pool_size = pool_size

        # get pretrained model
        self.pretrained_model = timm.create_model(hf_path, pretrained=True, num_classes=0, img_size=image_size).to(device)

        # freeze the pretrained model's parameters
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # get the embedding dimension
        self.embed_dim = len(self.layer_indices) * self.pretrained_model.embed_dim
        self.num_patches = (image_size // self.pretrained_model.patch_embed.patch_size[0])**2

        # remove indexes cls + registers
        pattern = r'reg(\d+)'
        match = re.search(pattern, hf_path)
        if match:
            self.start_index = int(match.group(1)) + 1
        else:
            self.start_index = 1

    def forward(self, x):
        # initial input processing
        x = self.pretrained_model.patch_embed(x)
        x = self.pretrained_model._pos_embed(x)
        x = self.pretrained_model.patch_drop(x)
        x = self.pretrained_model.norm_pre(x)

        out = []

        # iterating through the layers up to last layer to extract from
        for idx, layer in enumerate(self.pretrained_model.blocks, start=1):
            # x of shape (batch, num_patches, feature_size)
            x = layer(x)
            if idx in self.layer_indices:
                features_layer = self.pretrained_model.norm(x[:, self.start_index:, :])
                out.append(features_layer)
            if idx == max(self.layer_indices):
                break

        # cat the features of the layers to extract
        features = torch.cat(out, dim=-1)

        return features
    
class FeatureAdaptor(nn.Module):
    def __init__(self, embed_dim):
        super(FeatureAdaptor, self).__init__()
        self.linear = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape
        x = x.contiguous().view(-1, embed_dim)
        x = self.linear(x)
        x = x.view(batch_size, num_patches, -1)
        return x

class Discriminator(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers):
        super(Discriminator, self).__init__()

        # Initialize an empty Sequential container
        self.layers = nn.Sequential()

        # Loop to add all layers
        for layer_idx in range(num_layers - 1):
            if layer_idx == 0:
                # First layer - transforms input from embed_dim to hidden_dim
                self.layers.add_module(f"linear{layer_idx}", nn.Linear(embed_dim, hidden_dim))
            else:
                # Subsequent layers - maintain hidden_dim size
                self.layers.add_module(f"linear{layer_idx}", nn.Linear(hidden_dim, hidden_dim))

            # Add BatchNorm and LeakyReLU for each layer
            self.layers.add_module(f"batch_norm{layer_idx}", nn.BatchNorm1d(hidden_dim))
            self.layers.add_module(f"leaky_relu{layer_idx}", nn.LeakyReLU(0.2))

        # Add the final output layer
        self.layers.add_module("final_linear", nn.Linear(hidden_dim, 1, bias=False))

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape
        x = x.view(-1, embed_dim)
        x = self.layers(x)
        x = x.view(batch_size, num_patches)
        return x

class SimpleNet(pl.LightningModule):

    def __init__(self, lr, lr_adaptor, hf_path, layers_to_extract_from, hidden_dim, wd, epochs, noise_std, dsc_layers, pool_size, image_size, log_pixel_metrics, smoothing_sigma, smoothing_radius):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # feature extractor depends on the backbone type
        if hf_path.startswith("vit"):
            self.feature_extractor = FeatureExtractor_ViT(hf_path, layers_to_extract_from, image_size, self.device, pool_size)
        else:
            self.feature_extractor = FeatureExtractor(hf_path, layers_to_extract_from, pool_size, image_size, self.device)

        # get embedding dimension from the feature extractor
        embed_dim = self.feature_extractor.embed_dim
        num_patches = self.feature_extractor.num_patches
        self.patches_per_side = int(np.sqrt(num_patches))
            
        # models        
        self.feature_adaptor = FeatureAdaptor(embed_dim).to(self.device)
        self.discriminator = Discriminator(embed_dim, hidden_dim, dsc_layers).to(self.device)

        # init for evaluation
        self.val_scores = []
        self.val_labels = []
        self.val_masks = []
        self.test_scores = []
        self.test_labels = []
        self.test_masks = []

    def forward(self, x):
        scores = self._step(x)
        return scores

    def configure_optimizers(self):
        optimizer_dsc = optim.AdamW(self.discriminator.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        optimizer_fa = optim.AdamW(self.feature_adaptor.parameters(), lr=self.hparams.lr_adaptor)
        lr_scheduler_dsc = optim.lr_scheduler.CosineAnnealingLR(optimizer_dsc, self.hparams.epochs, self.hparams.lr*.4)
        lr_scheduler_fa = optim.lr_scheduler.LambdaLR(optimizer_fa, lr_lambda=lambda epoch: 1)
        return [optimizer_dsc, optimizer_fa], [lr_scheduler_dsc, lr_scheduler_fa]

    def _step(self, images):
        features = self.feature_extractor(images)
        adapted_features = self.feature_adaptor(features)
        scores = self.discriminator(adapted_features)
        return scores

    def training_step(self, batch, batch_idx):
        images = batch[0]
        features = self.feature_extractor(images)
        adapted_features = self.feature_adaptor(features)

        noise = torch.normal(0, self.hparams.noise_std * 1.1, adapted_features.shape).to(self.device)
        adapted_features_fake = adapted_features + noise

        scores = self.discriminator(torch.cat([adapted_features, adapted_features_fake]))
        true_scores = scores[:len(adapted_features)]
        fake_scores = scores[len(adapted_features):]

        true_loss = torch.clip(true_scores + 0.5, min=0)
        fake_loss = torch.clip(-fake_scores + 0.5, min=0)
        loss = true_loss.mean() + fake_loss.mean()

        self.log('train_loss', loss)

        # Manual optimization
        loss.backward()
        opt_dsc, opt_fa = self.optimizers()
        opt_dsc.step()
        opt_fa.step()
        opt_fa.zero_grad()
        opt_dsc.zero_grad()
        
        return loss

    def on_train_epoch_end(self, unused=None):
        lr_schedulers = self.lr_schedulers()
        for lr_scheduler in lr_schedulers:
            lr_scheduler.step()

    def validation_step(self, batch, batch_idx):
        images = batch[0]
        labels = batch[1]
        scores = self._step(images)
        self.val_scores.append(scores)
        self.val_labels.append(labels)

        if self.hparams.log_pixel_metrics:
            masks = batch[2]
            self.val_masks.append(masks)

    def test_step(self, batch, batch_idx):
        images = batch[0]
        labels = batch[1]
        scores = self._step(images)
        self.test_scores.append(scores)
        self.test_labels.append(labels)

        if self.hparams.log_pixel_metrics:
            masks = batch[2]
            self.test_masks.append(masks)

    def on_validation_epoch_end(self):
        scores = torch.cat(self.val_scores, dim=0)
        image_scores = torch.max(scores, dim=1)[0]
        image_labels = torch.cat(self.val_labels, dim=0)

        # calculate I-AUROC
        image_auroc = roc_auc_score(image_labels.view(-1).cpu().numpy(), image_scores.view(-1).cpu().numpy())
        self.log('val_image_auroc', round(image_auroc, 3), on_epoch=True)

        if self.hparams.log_pixel_metrics:
            masks = torch.cat(self.val_masks, dim=0)
            patch_scores = scores.reshape(-1, self.patches_per_side, self.patches_per_side)
            pixel_scores = F.interpolate(patch_scores.unsqueeze(1), size=(masks.shape[-1], masks.shape[-1]), mode='bilinear', align_corners=False)
            segmentations = gaussian_filter(pixel_scores.squeeze(1).cpu().detach().numpy(), sigma=self.hparams.smoothing_sigma, radius=self.hparams.smoothing_radius, axes=(1,2))

            # calculate P-AUROC    
            pixel_auroc = roc_auc_score(masks.view(-1).cpu().numpy(), segmentations.reshape(-1))
            self.log('val_pixel_auroc', round(pixel_auroc, 3), on_epoch=True)

        self.val_scores = []
        self.val_labels = []
        self.val_masks = []

    def _compute_pro(self, masks, segmentations, num_th=200):
        binary_segmentations = np.zeros_like(segmentations, dtype=bool)

        min_th = segmentations.min()
        max_th = segmentations.max()
        delta = (max_th - min_th) / num_th

        patch = getStructuringElement(MORPH_RECT, (5, 5))

        pro_data = []
        fpr_data = []

        for th in np.arange(min_th, max_th, delta):
            binary_segmentations[segmentations <= th] = 0
            binary_segmentations[segmentations > th] = 1

            pros = []
            for binary_segmentation, mask in zip(binary_segmentations, masks):
                binary_segmentation = dilate(binary_segmentation.astype(np.uint8), patch)
                for region in regionprops(label(mask)):
                    x_idx = region.coords[:, 0]
                    y_idx = region.coords[:, 1]
                    tp_pixels = binary_segmentation[x_idx, y_idx].sum()
                    pros.append(tp_pixels / region.area)
            
            inverse_masks = 1 - masks
            fp_pixels = np.logical_and(inverse_masks, binary_segmentations).sum()
            fpr = fp_pixels / inverse_masks.sum()

            pro_data.append(np.mean(pros))
            fpr_data.append(fpr)

        fpr_data = np.array(fpr_data)
        pro_data = np.array(pro_data)

        valid_indices = fpr_data < 0.3
        fpr_data = fpr_data[valid_indices]
        pro_data = pro_data[valid_indices]

        fpr_data = fpr_data / fpr_data.max()

        aupro = auc(fpr_data, pro_data)
        return aupro

    def on_test_epoch_end(self):
        scores = torch.cat(self.test_scores, dim=0)
        image_scores = torch.max(scores, dim=1)[0]
        image_labels = torch.cat(self.test_labels, dim=0)

        # calculate I-AUROC
        image_auroc = roc_auc_score(image_labels.view(-1).cpu().numpy(), image_scores.view(-1).cpu().numpy())
        self.log('test_image_auroc', round(image_auroc, 3), on_epoch=True)

        if self.hparams.log_pixel_metrics:
            masks = torch.cat(self.test_masks, dim=0)
            patch_scores = scores.reshape(-1, self.patches_per_side, self.patches_per_side)
            pixel_scores = F.interpolate(patch_scores.unsqueeze(1), size=(masks.shape[-1], masks.shape[-1]), mode='bilinear', align_corners=False)
            segmentations = gaussian_filter(pixel_scores.squeeze(1).cpu().detach().numpy(), sigma=self.hparams.smoothing_sigma, radius=self.hparams.smoothing_radius, axes=(1,2))

            # calculate P-AUROC    
            pixel_auroc = roc_auc_score(masks.view(-1).cpu().numpy(), segmentations.reshape(-1))
            self.log('test_pixel_auroc', round(pixel_auroc, 3), on_epoch=True)

            # calculate PRO-score    
            pro_score = self._compute_pro(masks.cpu().numpy(), segmentations)
            self.log('test_pro_score', round(pro_score, 3), on_epoch=True)

        self.test_scores = []
        self.test_labels = []
        self.test_masks = []









