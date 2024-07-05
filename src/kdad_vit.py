import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
import pytorch_lightning as pl
from transformers import ViTImageProcessor, ViTForImageClassification, AutoImageProcessor, AutoModel
import timm
import re

def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return x

class AttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in MLP
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the MLP
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x
    
class VisionTransformer(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, patch_size, num_patches, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels*(patch_size**2), embed_dim)
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,1+num_patches,embed_dim))


    def forward(self, x):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # output
        last_hidden_state = x.transpose(0, 1)[:, 1:, :]

        return last_hidden_state


class MLP(nn.Module):
    def __init__(self, patch_size, num_patches, num_channels, embed_dim, hidden_dim, dropout=0.0):
        """
        Inputs:
            patch_size - Size of the patches each image is divided into
            num_patches - Number of patches aka sequence length
            num_channels - Number of channels in the input image
            embed_dim - Dimensionality of the embedding layer
            hidden_dim - Dimensionality of the hidden layer in the MLP
        """
        super().__init__()

        self.patch_size = patch_size
        self.linear = nn.Sequential(nn.Linear(num_channels*(patch_size**2), hidden_dim),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(hidden_dim, embed_dim)
                                    )

    def forward(self, x):
        """
        Inputs:
            x - Input images as tensors, shape x: (batch, channels, height, width)
        """
        x = img_to_patch(x, self.patch_size)    # shape x: (batch, num_patches, num_channels*(patch_size**2))
        x = self.linear(x)                      # shape x: (batch, num_patches, embed_dim)
        return x
    
    
class AD_ViT(pl.LightningModule):

    def __init__(self, embed_dim, hidden_dim, num_heads, num_layers, patch_size, num_channels, num_patches, dropout, lr, hf_path, milestones, gamma, model_type):
        super().__init__()
        self.save_hyperparameters()

        # decide on type of modelto train
        if model_type == 'MLP':
            self.model = MLP(patch_size, num_patches, num_channels, embed_dim, hidden_dim, dropout)
        else:
            self.model = VisionTransformer(embed_dim, hidden_dim, num_channels, num_heads, num_layers, patch_size, num_patches, dropout)

        # get pretrained ViT
        self.vit_pretrained = timm.create_model(hf_path, pretrained=True, num_classes=0).to(self.device)

        # freeze the pretrained model's parameters
        for param in self.vit_pretrained.parameters():
            param.requires_grad = False

        # loss function
        self.criterion = nn.MSELoss()

        # init for evaluation
        self.val_scores = []
        self.val_labels = []
        self.test_scores = []
        self.test_labels = []

        # remove indexes cls + registers
        pattern = r'reg(\d+)'
        match = re.search(pattern, hf_path)
        if match:
            self.start_index = int(match.group(1)) + 1
        else:
            self.start_index = 1

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.milestones, gamma=self.hparams.gamma)
        return [optimizer], [lr_scheduler]

    def _step(self, batch, mode="train"):
        images = batch[0].to(self.device)

        # inference
        output_pred = self.model(images)
        output_real = self.vit_pretrained.forward_features(images)[:, self.start_index:, :]

        # loss
        y_pred_norm = F.normalize(output_pred, p=2, dim=2)
        y_real_norm = F.normalize(output_real, p=2, dim=2)
        loss = self.criterion(y_pred_norm, y_real_norm)
        self.log(f'{mode}_loss', loss)

        return loss, y_pred_norm, y_real_norm

    def _eval_step(self, batch, mode="val"):
        loss, y_pred_norm, y_real_norm = self._step(batch, mode=mode)

        labels = batch[1]
        scores = torch.mean((y_pred_norm - y_real_norm) ** 2, dim=(1, 2))

        return scores, labels

    def _eval_epoch_end(self, scores, labels, mode="val"):
        all_scores = torch.cat(scores, dim=0)
        all_labels = torch.cat(labels, dim=0)

        # numpy
        scores_np = all_scores.view(-1).cpu().numpy()
        labels_np = all_labels.view(-1).cpu().numpy()

        # calculate I-AUROC
        image_auroc = roc_auc_score(labels_np, scores_np)
        self.log(f'{mode}_image_auroc', round(image_auroc, 3), on_epoch=True)

        return image_auroc

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        scores, labels = self._eval_step(batch, mode="val")
        self.val_scores.append(scores)
        self.val_labels.append(labels)

    def test_step(self, batch, batch_idx):
        scores, labels = self._eval_step(batch, mode="test")
        self.test_scores.append(scores)
        self.test_labels.append(labels)

    def on_validation_epoch_end(self):
        val_result = self._eval_epoch_end(self.val_scores, self.val_labels, mode="val")
        self.val_scores = []
        self.val_labels = []

    def on_test_epoch_end(self):
        test_result = self._eval_epoch_end(self.test_scores, self.test_labels, mode="test")
        self.test_scores = []
        self.test_labels = []
