from src.load_data import prepare_loader
from .plot import plot_images, plot_with_heatmap_and_original
from torchvision.utils import make_grid

import os
import torch
import matplotlib.pyplot as plt

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from src.general_ad import General_AD

import torch
import torchvision.transforms.functional as tF
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os

def test_dataloader(args):
    train_loader, test_loader = prepare_loader(image_size=args.image_size,
                                                        path=args.data_dir,
                                                        dataset_name=args.dataset_name, 
                                                        class_name=args.normal_class, 
                                                        batch_size=args.batch_size, 
                                                        test_batch_size=args.test_batch_size,
                                                        num_workers=args.num_workers, 
                                                        seed=args.seed,
                                                        shots=args.shots)

    if args.dataset_name == 'mvtec-ad' or args.dataset_name == 'mvtec-loco-ad' or args.dataset_name == 'mpdd' or args.dataset_name == 'visa':   
        dataiter = iter(train_loader)
        images, labels, masks = next(dataiter)

        plot_images(make_grid(images, padding=20, pad_value=0), os.path.join('slurm', 'cache', f'{args.dataset_name}_train_loader_test'))
        print(labels)

        dataiter = iter(test_loader)
        images, labels, masks = next(dataiter)

        plot_images(make_grid(images, padding=20, pad_value=0), os.path.join('slurm', 'cache', f'{args.dataset_name}_test_loader_test'))
        print(labels)

        first_mask = masks[0].cpu().numpy()  

        plt.figure(figsize=(6, 6))
        plt.imshow(first_mask, cmap='gray') 
        plt.title('First Mask in the Batch')
        plt.axis('off')  
        plt.savefig(os.path.join('slurm', 'cache', f'{args.dataset_name}_test_pixel_mask'))
        plt.close()
    else:
        dataiter = iter(train_loader)
        images, labels = next(dataiter)

        plot_images(make_grid(images, padding=20, pad_value=0), os.path.join('slurm', 'cache', f'{args.dataset_name}_train_loader_test'))
        print(labels)

        dataiter = iter(test_loader)
        images, labels = next(dataiter)

        plot_images(make_grid(images, padding=20, pad_value=0), os.path.join('slurm', 'cache', f'{args.dataset_name}_test_loader_test'))
        print(labels)

    return

