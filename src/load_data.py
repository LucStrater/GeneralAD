# basic
import os
import random
import sys
from PIL import Image
import scipy.io as sio
import numpy as np

# torch
import torch
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, CIFAR100, FGVCAircraft, FashionMNIST

class RandomSubsetSampler(data.Sampler):
    def __init__(self, data_source, subset_size):
        self.data_source = data_source
        self.subset_size = subset_size

    def __iter__(self):
        return iter(np.random.choice(len(self.data_source), self.subset_size, replace=False))

    def __len__(self):
        return self.subset_size

class MVTec(data.Dataset):
    def __init__(self, dataset_name, path, class_name, transform=None, mask_transform=None, seed=0, split='train'):
        self.transform = transform
        self.mask_transform = mask_transform
        self.data = []

        if dataset_name == 'mvtec-loco-ad':
            path = os.path.join(path, "mvtec_loco_ad", class_name)
            mv_str = '/000.'
        elif dataset_name == 'mvtec-ad':
            path = os.path.join(path, "mvtec_ad", class_name)
            mv_str = '_mask.'
        else:
            path = os.path.join(path, "MPDD", class_name)
            mv_str = '_mask.'

        # normall folders
        normal_dir = os.path.join(path, split, "good")

        # normal samples
        for img_file in os.listdir(normal_dir):
            image_dir = os.path.join(normal_dir, img_file)
            self.data.append((image_dir, None))
        
        if split == 'test':
            # anomaly folder
            test_dir = os.path.join(path, "test")
            test_anomaly_dirs = []
            for entry in os.listdir(test_dir):
                full_path = os.path.join(test_dir, entry)

                # check if the entry is a directory and not the non-anomaly one
                if os.path.isdir(full_path) and full_path != normal_dir:
                    test_anomaly_dirs.append(full_path)

            # anomaly samples
            for dir in test_anomaly_dirs:
                for img_file in os.listdir(dir):
                    image_dir = os.path.join(dir, img_file)
                    mask_dir = image_dir.replace("test", "ground_truth")
                    parts = mask_dir.rsplit('.', 1)
                    mask_dir = parts[0] + mv_str + parts[1]
                    self.data.append((image_dir, mask_dir))

            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')  

        image = self.transform(image)          

        if mask_path:
            mask = Image.open(mask_path).convert('RGB')
            mask = self.mask_transform(mask)
            mask = 1.0 - torch.all(mask == 0, dim=0).float()
            label = 1
        else:
            C, W, H = image.shape
            mask = torch.zeros((H, W))
            label = 0
            
        return image, label, mask

class VisA(data.Dataset):
    def __init__(self, path, class_name, transform=None, mask_transform=None, seed=0, split='train'):
        self.path_normal = os.path.join(path, "visa", class_name, "Data", "Images", "Normal")
        self.path_anomaly = os.path.join(path, "visa", class_name, "Data", "Images", "Anomaly")
        self.class_name = class_name
        self.transform = transform
        self.mask_transform = mask_transform
        self.data = []
        img_count = 0

        for filename in os.listdir(self.path_normal):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                img_count += 1

        if split == 'train':
            i = 0
            for img_path in os.listdir(self.path_normal):
                if i < int(0.8*img_count):
                    self.data.append((os.path.join(self.path_normal, img_path), None)) 
                i += 1
        elif split == 'test':
            i = 0
            for img_path in os.listdir(self.path_normal):
                if i >= int(0.8*img_count):
                    self.data.append((os.path.join(self.path_normal, img_path), None)) 
                i += 1

            for img_path in os.listdir(self.path_anomaly):
                image_dir = os.path.join(self.path_anomaly, img_path)
                mask_dir = image_dir.replace("Images", "Masks")[:-3] + "png"
                self.data.append((image_dir, mask_dir)) 

            random.seed(seed)
            random.shuffle(self.data)            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')  

        image = self.transform(image)          

        if mask_path:
            mask = Image.open(mask_path).convert('RGB')
            mask = self.mask_transform(mask)
            mask = 1.0 - torch.all(mask == 0, dim=0).float()
            label = 1
        else:
            C, W, H = image.shape
            mask = torch.zeros((H, W))
            label = 0
            
        return image, label, mask

class View(data.Dataset):
    def __init__(self, path, class_name, transform=None, seed=0, split='train'):
        self.transform = transform
        self.data = []
        normal_dir = os.path.join(path, "view", split, class_name)

        # normal samples
        for img_file in os.listdir(normal_dir):
            image_dir = os.path.join(normal_dir, img_file)
            self.data.append((image_dir, 0))
        
        if split == 'test':
            # anomaly folder
            test_dir = os.path.join(path, "view", "test")
            test_anomaly_dirs = []
            for entry in os.listdir(test_dir):
                full_path = os.path.join(test_dir, entry)

                # check if the entry is a directory and not the non-anomaly one
                if os.path.isdir(full_path) and full_path != normal_dir:
                    test_anomaly_dirs.append(full_path)

            # anomaly samples
            for dir in test_anomaly_dirs:
                for img_file in os.listdir(dir):
                    image_dir = os.path.join(dir, img_file)
                    self.data.append((image_dir, 1))

            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')  

        image = self.transform(image)
            
        return image, label

class StanfordCars(data.Dataset):
    def __init__(self, path, class_name, transform, seed=0, split='train'):
        self.transform = transform
        path_base = os.path.join(path, "stanford_cars")
        class_name = int(class_name)

        if split == 'train':
            path_images = os.path.join(path_base, "cars_train")
            path_classes = os.path.join(path_base, "devkit", "cars_train_annos.mat")

            self.data = [(os.path.join(path_images, annotation["fname"]), 0)
                        for annotation in sio.loadmat(path_classes, squeeze_me=True)["annotations"]
                        if int(annotation["class"]) == class_name]
        elif split == 'test':
            path_images = os.path.join(path_base, "cars_test")
            path_classes = os.path.join(path_base, "cars_test_annos_withlabels.mat")

            test_set_0 = [(os.path.join(path_images, annotation["fname"]), 0)
                          for annotation in sio.loadmat(path_classes, squeeze_me=True)["annotations"]
                          if int(annotation["class"]) == class_name]
            test_set_1 = [(os.path.join(path_images, annotation["fname"]), 1)
                          for annotation in sio.loadmat(path_classes, squeeze_me=True)["annotations"]
                          if int(annotation["class"]) != class_name]

            num_zeros = len(test_set_0)
            num_ones = len(test_set_1)
            spacing = num_ones // (num_zeros + 1)

            # final test set with equally spaced 0's
            self.data = []
            index = 0
            for i in range(num_zeros):
                self.data.append(test_set_0[i])
                self.data.extend(test_set_1[index:index+spacing])
                index += spacing
            self.data.extend(test_set_1[index:])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')  
        if self.transform:
            image = self.transform(image)
        return image, label

class CatsVsDogs(data.Dataset):
    def __init__(self, path, class_name, transform=None):
        self.path = os.path.join(path, "catsvdogs")
        self.class_name = class_name
        self.transform = transform
        self.data = []

        self.load_dataset()

    def load_dataset(self):
        classes = ['Cat', 'Dog']
        
        for cls in classes:
            cls_path = os.path.join(self.path, cls)
            label = 0 if cls == self.class_name else 1
            for img_name in os.listdir(cls_path):
                if img_name.endswith('.jpg'):  
                    img_path = os.path.join(cls_path, img_name)
                    self.data.append((img_path, label))  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')  
        if self.transform:
            image = self.transform(image)
        return image, label
    
def prepare_loader(image_size, path, dataset_name, class_name, batch_size, test_batch_size, num_workers, seed, shots):
    transform = transforms.Compose([transforms.Resize((image_size, image_size), Image.LANCZOS),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                    ])
    transform_fmnist = transforms.Compose([transforms.Resize((image_size, image_size), Image.LANCZOS),
                                            transforms.Grayscale(num_output_channels=3),  
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                            ])
    mask_transform = transforms.Compose([transforms.Resize((image_size, image_size), Image.LANCZOS),
                                        transforms.ToTensor()  
                                        ])
    if dataset_name == 'mvtec-loco-ad' or dataset_name == 'mvtec-ad' or dataset_name == 'mpdd':
        train_set = MVTec(dataset_name, path, class_name, transform=transform, mask_transform=mask_transform, seed=seed, split='train')
        test_set = MVTec(dataset_name, path, class_name, transform=transform, mask_transform=mask_transform, seed=seed, split='test')
    elif dataset_name == 'visa':
        train_set = VisA(path, class_name, transform=transform, mask_transform=mask_transform, seed=seed, split='train')
        test_set = VisA(path, class_name, transform=transform, mask_transform=mask_transform, seed=seed, split='test')
    elif dataset_name == 'cifar10':
        train_dataset = CIFAR10(root=path, train=True, transform=transform, download=True)
        test_set = CIFAR10(root=path, train=False, transform=transform, download=True)

        # set target to anomaly or not
        for dataset in [train_dataset, test_set]:
            dataset.targets = [0 if target == int(class_name) else 1 for target in dataset.targets]

        # create subsets
        filtered_indices = [idx for idx, target in enumerate(train_dataset.targets) if target == 0]
        train_set = torch.utils.data.Subset(train_dataset, filtered_indices)
    elif dataset_name == 'cifar100':
        train_dataset = CIFAR100(root=path, train=True, transform=transform, download=True)
        test_set = CIFAR100(root=path, train=False, transform=transform, download=True)

        coarse_labels = [4, 1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13]

        # set target to anomaly or not
        for dataset in [train_dataset, test_set]:
            dataset.targets = [0 if coarse_labels[target] == int(class_name) else 1 for target in dataset.targets]

        # create subsets
        filtered_indices = [idx for idx, target in enumerate(train_dataset.targets) if target == 0]
        train_set = torch.utils.data.Subset(train_dataset, filtered_indices)
    elif dataset_name == 'fmnist':
        train_dataset = FashionMNIST(root=path, train=True, transform=transform_fmnist, download=True)
        test_set = FashionMNIST(root=path, train=False, transform=transform_fmnist, download=True)

        # set target to anomaly or not
        for dataset in [train_dataset, test_set]:
            dataset.targets = [0 if target == int(class_name) else 1 for target in dataset.targets]

        # create subsets
        filtered_indices = [idx for idx, target in enumerate(train_dataset.targets) if target == 0]
        train_set = torch.utils.data.Subset(train_dataset, filtered_indices)
    elif dataset_name == 'view':
        train_set = View(path, class_name, transform=transform, seed=seed, split='train')
        test_set = View(path, class_name, transform=transform, seed=seed, split='test')
    elif dataset_name == 'fgvc-aircraft':
        train_dataset = FGVCAircraft(root=path, split='train', annotation_level='variant', transform=transform, download=True)
        test_dataset = FGVCAircraft(root=path, split='test', annotation_level='variant', transform=transform, download=True)

        desired_labels = [91, 96, 59, 19, 37, 45, 90, 68, 74, 89]

        train_set = [(data, 0) for (data, target) in train_dataset if target == int(class_name)]
        test_set_0 = [(data, 0) for (data, target) in test_dataset if target == int(class_name)]
        test_set_1 = [(data, 1) for (data, target) in test_dataset if target in desired_labels and target != int(class_name)]

        num_zeros = len(test_set_0)
        num_ones = len(test_set_1)
        spacing = num_ones // (num_zeros + 1)

        # final test set with equally spaced 0's
        test_set = []
        index = 0
        for i in range(num_zeros):
            test_set.append(test_set_0[i])
            test_set.extend(test_set_1[index:index+spacing])
            index += spacing
        test_set.extend(test_set_1[index:])
    elif dataset_name == 'stanford-cars':
        train_set = StanfordCars(path, class_name, transform, seed=seed, split='train')
        test_set = StanfordCars(path, class_name, transform, seed=seed, split='test')
    elif dataset_name == 'catsvdogs':
        dataset = CatsVsDogs(path, class_name, transform)

        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        test_size = total_size - train_size 

        train_dataset, test_set = data.random_split(dataset, [train_size, test_size])
        filtered_indices = [idx for idx in train_dataset.indices if dataset.data[idx][1] == 0]
        train_set = torch.utils.data.Subset(dataset, filtered_indices)
    else:
        sys.exit("This is not a valid dataset name")

    if shots > 0 and shots < len(train_set):
        indices = list(range(shots))
        indices_seeded = [x + seed for x in indices]  
        train_subset = data.Subset(train_set, indices_seeded)
        train_loader = data.DataLoader(train_subset, batch_size=min(shots, batch_size), shuffle=True, drop_last=True, pin_memory=True)
    elif dataset_name == 'cifar10' or dataset_name == 'cifar100' or dataset_name == 'fmnist' or dataset_name == 'view' or dataset_name == 'catsvdogs':
        sampler_train = RandomSubsetSampler(train_set, 250)
        train_loader = data.DataLoader(train_set, batch_size=batch_size, sampler=sampler_train, drop_last=True, pin_memory=True, num_workers=num_workers)
    else:
        train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=num_workers)
    
    test_loader = data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    return train_loader, test_loader