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

def my_forward_wrapper(attn_obj):
    def my_forward(x):
        B, N, C = x.shape
        qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   

        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x
    return my_forward

class FeatureExtractor_ViT(nn.Module):
    def __init__(self, hf_path, layer_indices, image_size, device):
        super(FeatureExtractor_ViT, self).__init__()
        self.layer_indices = layer_indices

        # get pretrained model
        if hf_path.endswith("_ibot"):
            self.pretrained_model = timm.create_model(hf_path[:-5], pretrained=False, num_classes=0, img_size=image_size).to(device)

            if hf_path.startswith('vit_small'):
                model_url = 'https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint_teacher.pth'
            elif hf_path.startswith('vit_base'):
                model_url = 'https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth'
            elif hf_path.startswith('vit_large'):
                model_url = 'https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16/checkpoint_teacher.pth'
            else:
                print("This model does not have an IBOT version")
                sys.exit()

            model_path = os.path.join('data', 'model.pth')
            wget.download(model_url, model_path)
            state_dict = torch.load(model_path)['state_dict']
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.pretrained_model.load_state_dict(state_dict, strict=False)
        else:
            self.pretrained_model = timm.create_model(hf_path, pretrained=True, num_classes=0, img_size=image_size).to(device)

        # freeze the pretrained model's parameters
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # store shapes
        self.embed_dim = len(layer_indices) * self.pretrained_model.embed_dim
        self.patch_size = self.pretrained_model.patch_embed.patch_size[0]
        self.num_patches = (image_size // self.patch_size)**2

        # remove indexes cls + registers
        pattern = r'reg(\d+)'
        match = re.search(pattern, hf_path)
        self.start_index = int(match.group(1)) + 1 if match else 1

        # wrapper to be able to extract the attention map
        self.pretrained_model.blocks[layer_indices[-1]-1].attn.forward = my_forward_wrapper(self.pretrained_model.blocks[layer_indices[-1]-1].attn)

    def forward(self, x, output_attn=False):
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

        if output_attn:
            attn_map = self.pretrained_model.blocks[self.layer_indices[-1]-1].attn.attn_map
            attn_map_cls = attn_map[:, :, 0, self.start_index:]

        return (features, attn_map_cls) if output_attn else features

class FeatureExtractor_EVA(nn.Module):
    def __init__(self, hf_path, layer_indices, image_size, device):
        super(FeatureExtractor_EVA, self).__init__()
        self.layer_indices = layer_indices

        # get pretrained model
        self.pretrained_model = timm.create_model(hf_path, pretrained=True, num_classes=0, img_size=image_size).to(device)

        # freeze the pretrained model's parameters
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # store shapes
        self.embed_dim = len(layer_indices) * self.pretrained_model.embed_dim
        self.patch_size = self.pretrained_model.patch_embed.patch_size[0]
        self.num_patches = (image_size // self.patch_size)**2

    def forward(self, x, output_attn=False):
        # initial input processing
        x = self.pretrained_model.patch_embed(x)
        x, rot_pos_embed = self.pretrained_model._pos_embed(x)

        out = []

        # iterating through the layers up to last layer to extract from
        for idx, layer in enumerate(self.pretrained_model.blocks, start=1):
            # x of shape (batch, num_patches, feature_size)
            x = layer(x, rope=rot_pos_embed)
            if idx in self.layer_indices:
                features_layer = self.pretrained_model.norm(x[:, 1:, :])
                out.append(features_layer)
            if idx == max(self.layer_indices):
                break

        # cat the features of the layers to extract
        features = torch.cat(out, dim=-1)

        return features

class AttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.layer_norm_1(x + self.dropout1(attn_output))
        x = x + self.dropout2(self.linear(x))
        return x


class PPT_Discriminator(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_patches, num_layers=1, num_heads=12, dropout_rate=0):
        super(PPT_Discriminator, self).__init__()

        # Transformer encoder layer
        self.transformer_encoder = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout_rate) for _ in range(num_layers)])

        # Output layer
        self.output_layer = nn.Linear(embed_dim, 1, bias=False)

        # Learnable positional encodings
        self.positional_encodings = nn.Parameter(torch.randn(num_patches, embed_dim))

    def forward(self, x):
        # pos emb
        x = x + self.positional_encodings.unsqueeze(0)
        x = self.transformer_encoder(x)
        x = self.output_layer(x).squeeze(-1)
        return x

class General_AD(pl.LightningModule):

    def __init__(self, lr, lr_decay_factor, hf_path, layers_to_extract_from, hidden_dim, wd, epochs, noise_std, dsc_layers, dsc_heads, dsc_dropout, pool_size, image_size, num_fake_patches, fake_feature_type, top_k, log_pixel_metrics, smoothing_sigma, smoothing_radius):
        super().__init__()
        self.save_hyperparameters()

        # feature extractor depends on the backbone type
        if hf_path.startswith("vit"):
            self.feature_extractor = FeatureExtractor_ViT(hf_path, layers_to_extract_from, image_size, self.device)
            self.attn_output = True
        elif hf_path.startswith("eva"):
            self.feature_extractor = FeatureExtractor_EVA(hf_path, layers_to_extract_from, image_size, self.device)
            self.attn_output = False
        else:
            self.feature_extractor = FeatureExtractor(hf_path, layers_to_extract_from, pool_size, image_size, self.device)
            self.attn_output = False

        # get feature extractor parameters
        embed_dim = self.feature_extractor.embed_dim
        num_patches = self.feature_extractor.num_patches
        self.patch_size = self.feature_extractor.patch_size
        self.patches_per_side = int(np.sqrt(num_patches))

        if top_k < 0 or top_k > num_patches:
            self.top_k = num_patches
        else:
            self.top_k = top_k

        # make sure the number of fake patches doesnt crash the code
        if num_fake_patches < 0 or num_fake_patches > num_patches:
            self.num_fake_patches = num_patches
        else:
            self.num_fake_patches = num_fake_patches
            
        # models  
        self.discriminator = PPT_Discriminator(embed_dim, hidden_dim, num_patches, dsc_layers, dsc_heads, dsc_dropout).to(self.device)

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
        lr_scheduler_dsc = optim.lr_scheduler.CosineAnnealingLR(optimizer_dsc, self.hparams.epochs, self.hparams.lr*self.hparams.lr_decay_factor)
        return {"optimizer": optimizer_dsc, "lr_scheduler": {"scheduler": lr_scheduler_dsc, "interval": "epoch"}}

    def _step(self, images):
        features = self.feature_extractor(images)
        scores = self.discriminator(features)
        return scores

    def _add_random_noise(self, features):
        # clone features to avoid in-place modification
        fake_features = features.clone()

        # create a noise tensor
        noise = torch.normal(0, self.hparams.noise_std, features.shape).to(self.device)

        batch_size, num_patches, feature_dim = features.shape 
        masks = torch.zeros((batch_size, num_patches), dtype=torch.bool)  

        for i in range(batch_size):         
            num_fake = random.randint(1, num_patches)
            random_indices = torch.randperm(num_patches)[:num_fake]
            masks[i, random_indices] = True
            fake_features[i, random_indices, :] += noise[i, random_indices, :]

        return fake_features, masks

    def _add_attn_noise(self, features, attn_map):
        # clone features to avoid in-place modification
        fake_features = features.clone()

        # create a noise tensor
        noise = torch.normal(0, self.hparams.noise_std, features.shape).to(self.device)

        batch_size, num_heads, num_patches = attn_map.shape 
        masks = torch.zeros((batch_size, num_patches), dtype=torch.bool)  

        for i in range(batch_size):         
            head = random.randint(0, num_heads-1)
            num_fake = random.randint(1, num_patches)
            max_attn_indices = torch.tensor(list(torch.topk(attn_map[i, head, :], num_fake)[1]))
            masks[i, max_attn_indices] = True
            fake_features[i, max_attn_indices, :] += noise[i, max_attn_indices, :]

        return fake_features, masks

    def _add_attn_copy_out(self, features, attn_map):
        # clone features to avoid in-place modification
        fake_features = features.clone()

        batch_size, num_heads, num_patches = attn_map.shape 
        masks = torch.zeros((batch_size, num_patches), dtype=torch.bool)  

        for i in range(batch_size):
            # take the number of patches with the highest attention score from a random chosen head
            head = random.randint(0, num_heads-1)
            num_fake = random.randint(1, self.num_fake_patches)
            max_attn_indices = torch.tensor(list(torch.topk(attn_map[i, head, :], num_fake)[1]))
            random_indices = torch.randperm(num_patches)[:num_fake]
            masks[i, max_attn_indices] = True
            fake_features[i, max_attn_indices, :] = features[i, random_indices, :]

        return fake_features, masks   

    def _add_attn_shuffle(self, features, attn_map):
        # clone features to avoid in-place modification
        fake_features = features.clone()

        batch_size, num_heads, num_patches = attn_map.shape 
        masks = torch.zeros((batch_size, num_patches), dtype=torch.bool)  

        for i in range(batch_size):
            # take the number of patches with the highest attention score from a random chosen head
            head = random.randint(0, num_heads-1)
            num_fake = random.randint(1, self.num_fake_patches)
            max_attn_indices = torch.tensor(list(torch.topk(attn_map[i, head, :], num_fake)[1]))
            masks[i, max_attn_indices] = True

            # shuffle
            shuffled_patches = fake_features[i, max_attn_indices].clone()
            shuffled_patches = shuffled_patches[torch.randperm(num_fake)]
            fake_features[i, max_attn_indices, :] = shuffled_patches

        return fake_features, masks  

    def _add_random_shuffle(self, features):
        # clone features to avoid in-place modification
        fake_features = features.clone()

        batch_size, num_patches, feature_dim = features.shape 
        masks = torch.zeros((batch_size, num_patches), dtype=torch.bool)  

        for i in range(batch_size):
            # take the number of patches with the highest attention score from a random chosen head
            num_fake = random.randint(1, self.num_fake_patches)
            random_indices = torch.randperm(num_patches)[:num_fake]
            masks[i, random_indices] = True

            # shuffle
            shuffled_patches = fake_features[i, random_indices].clone()
            shuffled_patches = shuffled_patches[torch.randperm(num_fake)]
            fake_features[i, random_indices, :] = shuffled_patches

        return fake_features, masks  

    def _add_noise_all(self, features):
        # clone features to avoid in-place modification
        fake_features = features.clone()

        # create a noise tensor
        noise = torch.normal(0, self.hparams.noise_std, features.shape).to(self.device)
        fake_features = fake_features + noise
        masks = torch.ones(features.shape[0]*features.shape[1]).to(self.device)

        return fake_features, masks

    def training_step(self, batch, batch_idx):
        images = batch[0]

        if self.attn_output:
            features, attn_map = self.feature_extractor(images, output_attn=True)
        else:
            features = self.feature_extractor(images)

        # stack loss
        loss = 0

        # true
        scores_true = self.discriminator(features).flatten()
        masks_true = torch.zeros(features.shape[0]*features.shape[1]).to(self.device)
        loss += F.binary_cross_entropy_with_logits(scores_true, masks_true)

        # fake
        fake_features, masks_fake = self._add_noise_all(features)
        scores_fake = self.discriminator(fake_features)
        loss += F.binary_cross_entropy_with_logits(scores_fake.flatten(), masks_fake.flatten())

        if self.hparams.fake_feature_type in ['random', 'copy_out_and_random', 'shuffle_and_random', 'randshuffle_and_random']:
            # add noise to random subset of the patches
            random_features, masks_random = self._add_random_noise(features)
            scores_random = self.discriminator(random_features).flatten()
            masks_random = masks_random.flatten().to(self.device)
            loss += F.binary_cross_entropy_with_logits(scores_random[~masks_random], masks_random[~masks_random].float())
            loss += F.binary_cross_entropy_with_logits(scores_random[masks_random], masks_random[masks_random].float())
        elif self.hparams.fake_feature_type in ['attn', 'copy_out_and_attn', 'shuffle_and_attn', 'randshuffle_and_attn']:
            # add noise to patches with highest attention in the feature extractor 
            attn_features, masks_attn = self._add_attn_noise(features, attn_map)
            scores_attn = self.discriminator(attn_features).flatten()
            masks_attn = masks_attn.flatten().to(self.device)
            loss += F.binary_cross_entropy_with_logits(scores_attn[~masks_attn], masks_attn[~masks_attn].float())
            loss += F.binary_cross_entropy_with_logits(scores_attn[masks_attn], masks_attn[masks_attn].float())
  
        if self.hparams.fake_feature_type in ['copy_out', 'copy_out_and_random', 'copy_out_and_attn']:
            # perform cutpaste in the feature space based on the patches with the highest attention value
            copy_features, masks_copy = self._add_attn_copy_out(features, attn_map)
            scores_copy = self.discriminator(copy_features).flatten()
            masks_copy = masks_copy.flatten().to(self.device)
            loss += F.binary_cross_entropy_with_logits(scores_copy[~masks_copy], masks_copy[~masks_copy].float())
            loss += F.binary_cross_entropy_with_logits(scores_copy[masks_copy], masks_copy[masks_copy].float())
        elif self.hparams.fake_feature_type in ['shuffle', 'shuffle_and_random', 'shuffle_and_attn']:
            # shuffle the patches with the highest attention values in the feature extractor
            shuffle_features, masks_shuffle = self._add_attn_shuffle(features, attn_map)
            scores_shuffle = self.discriminator(shuffle_features).flatten()
            masks_shuffle = masks_shuffle.flatten().to(self.device)
            loss += F.binary_cross_entropy_with_logits(scores_shuffle[~masks_shuffle], masks_shuffle[~masks_shuffle].float())
            loss += F.binary_cross_entropy_with_logits(scores_shuffle[masks_shuffle], masks_shuffle[masks_shuffle].float())
        elif self.hparams.fake_feature_type in ['randshuffle', 'randshuffle_and_random', 'randshuffle_and_attn']:
            # shuffle a random subset of the patches
            randshuffle_features, masks_randshuffle = self._add_random_shuffle(features)
            scores_randshuffle = self.discriminator(randshuffle_features).flatten()
            masks_randshuffle = masks_randshuffle.flatten().to(self.device)
            loss += F.binary_cross_entropy_with_logits(scores_randshuffle[~masks_randshuffle], masks_randshuffle[~masks_randshuffle].float())
            loss += F.binary_cross_entropy_with_logits(scores_randshuffle[masks_randshuffle], masks_randshuffle[masks_randshuffle].float())

        self.log('train_loss', loss)
        
        return loss

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

        topk_values, _ = torch.topk(scores, self.top_k, dim=1)
        image_scores = torch.mean(topk_values, dim=1)
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

        patch = getStructuringElement(MORPH_RECT, (self.patch_size, self.patch_size))

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

        topk_values, _ = torch.topk(scores, self.top_k, dim=1)
        image_scores = torch.mean(topk_values, dim=1)
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




