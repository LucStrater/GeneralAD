import argparse
import os

# import functions
from utils.test import test_dataloader
from src.run_model import run
from src.run_viz import viz_segmentation


def main(args):
    if args.run_type == 'kdad' or args.run_type == 'simplenet' or args.run_type == 'general_ad':
        run(args)        
    elif args.run_type == 'test_data':
        test_dataloader(args)
    elif args.run_type == 'viz_segmentation':
        viz_segmentation(args)

if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(description="Train an Anomaly Detection algorithm")

    # Add arguments
    parser.add_argument("--normal_class", type=str, default=0, help="Normal class for training")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generators")
    parser.add_argument("--dataset_name", default="cifar10", choices=['cifar10', 'mvtec-loco-ad', 'mvtec-ad', 'fgvc-aircraft', 'cifar100', 'stanford-cars', 'fmnist', 'catsvdogs', 'view', 'mpdd', 'visa'], help="Name of the dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--log_every_n_steps", type=int, default=5, help="Log frequency")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=128, help="Batch size for testing")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--patch_size", type=int, default=14, help="Size of each patch")
    parser.add_argument("--num_channels", type=int, default=3, help="Number of channels")
    parser.add_argument("--num_patches", type=int, default=256, help="Number of patches")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--lr_decay_factor", type=float, default=0.1, help="Learning rate decay factor for cosine annealing")
    parser.add_argument("--lr_adaptor", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--hf_path", type=str, default='vit_base_patch14_dinov2.lvd142m', help="Huggingface model path")
    parser.add_argument("--milestones", type=str, default="5", help="Scheduler milestones as a comma-separated string")
    parser.add_argument("--gamma", type=float, default=0.2, help="Scheduler gamma value")
    parser.add_argument("--wandb_entity", type=str, default="private", help="WandB entity")
    parser.add_argument("--wandb_api_key", type=str, default="private", help="WandB API key")
    parser.add_argument("--wandb_name", type=str, default="default", help="WandB run name for logging")
    parser.add_argument('--data_dir', type=str, default='data/', help='Data directory where to store/find the dataset.')
    parser.add_argument("--run_type", default="kdad", choices=['kdad', 'test_data', 'simplenet', 'viz_attn', 'general_ad', 'viz_segmentation'], help="The files that have to be run.")
    parser.add_argument("--model_type", default="ViT", choices=['ViT', 'MLP'], help="The type of model to be trained for KDAD.")
    parser.add_argument("--image_size", type=int, default=224, help="Input size of ViT images")
    parser.add_argument("--layers_to_extract_from", type=str, default="2,3", help="Layers to extract from as a comma-separated string")
    parser.add_argument("--wd", type=float, default=0.00001, help="Weight decay for the discriminator")
    parser.add_argument("--dsc_layers", type=int, default=1, help="Number of layers for the discriminator")
    parser.add_argument("--dsc_heads", type=int, default=12, help="Number of heads for the discriminator")
    parser.add_argument("--dsc_dropout", type=float, default=0.0, help="Dropout rate for the discriminator")
    parser.add_argument("--noise_std", type=float, default=0.015, help="Standard deviation of the noise to create fake samples for the discriminator")
    parser.add_argument("--dsc_type", default="mlp", choices=['mlp', 'transformer'], help="The type of model you want for the discriminator.")
    parser.add_argument('--no_avg_pooling', action='store_false', help='Set to disable average pooling. Defaults to True.')
    parser.add_argument("--pool_size", type=int, default=3, help="Size of local neighboorhood to aggregate over.")
    parser.add_argument("--num_fake_patches", type=int, default=-1, help="Number of fake patches for the transformer discriminator")
    parser.add_argument('--load_checkpoint', action='store_true', help='Load the model from a checkpoint instead of training from scratch. Defaults to False.')
    parser.add_argument("--checkpoint_dir", type=str, default="lightning_logs/dir", help="The directory in which the model checkpoints are stored, printed after a training run.")
    parser.add_argument("--blob_size_factor", type=float, default=1.0, help="Size of the blob")
    parser.add_argument("--sigma_blob_noise", type=float, default=0.4, help="magnitude of the standard deviation for the probability distribution over the grid for creating the starting patch of the blob.")
    parser.add_argument("--fake_feature_type", type=str, default="noise_all", help="The type of fake featuers to create for general ad.")
    parser.add_argument("--top_k", type=int, default=10, help="number of patches to use to determine if an image is anomalous.")
    parser.add_argument("--smoothing_sigma", type=float, default=4, help="Standard deviation of the smoothing to create the segmentation map.")
    parser.add_argument("--smoothing_radius", type=int, default=2, help="Standard deviation of the smoothing to create the segmentation map.")
    parser.add_argument("--shots", type=int, default=-1, help="number of shots for few-shot setting.")
    parser.add_argument("--val_monitor", default="image_auroc", choices=['image_auroc', 'pixel_auroc'], help="Validate based on image level score or pixel level score.")
    parser.add_argument("--log_pixel_metrics", type=int, default=0, choices=[0, 1], help="If the dataset includes segmentation masks than 1 else 0.")

    # Parse arguments
    args = parser.parse_args()
    args.milestones = [int(x) for x in args.milestones.split(',')]
    args.layers_to_extract_from = [int(x) for x in args.layers_to_extract_from.split(',')]

    # Run the main function
    main(args)
