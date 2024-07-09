"""
Author: Haoran Chen
Date: 2024.07.07
"""

import os
from torch.utils.data import Dataset
from continuum.datasets import CIFAR100, ImageNet1000, OfficeHome, DomainNet, ImageFolderDataset, TinyImageNet200, Core50, CUB200
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from utils import imagenet1k_labels, imagenetr_labels, tinyimagenet_labels
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from torchvision.transforms import functional as Fv
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


try:
    interpolation = Fv.InterpolationMode.BICUBIC
except:
    interpolation = 3

n_px = 224


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def build_transform(is_train, args):
    resize_im = True
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * 224)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(224))
    t.append(transforms.ToTensor())
    
    return transforms.Compose(t)



def gen_dataset(args):
    if args["dataset"] == 'IMAGENET1K':
        train_dataset = ImageNet1000(data_path=args["data_root"], download=True, train=True)
        test_dataset = ImageNet1000(data_path=args["data_root"], download=True, train=False)

        label_path = os.path.join(args["file_root"], 'imagenet1000_clsidx_to_labels.txt')
        classnames = imagenet1k_labels(label_path)

    elif args["dataset"] == 'IMAGENET100':
        train_dataset = ImageNet100(data_path=args["data_root"], download=True, train=True)
        test_dataset = ImageNet100(data_path=args["data_root"], download=True, train=False)

        label_path = os.path.join(args["file_root"], 'imagenet1000_clsidx_to_labels.txt')
        classnames = imagenet1k_labels(label_path)
        classnames = classnames[:100]

    elif args["dataset"] == 'Cifar':
        train_dataset = CIFAR100(data_path=args["data_root"], download=True, train=True)
        test_dataset = CIFAR100(data_path=args["data_root"], download=True, train=False)

        classnames = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                      'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                      'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'cra', 'crocodile', 'cup', 'dinosaur',
                      'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
                      'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
                      'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
                      'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
                      'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
                      'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
                      'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    
    elif args["dataset"] == 'OfficeHome':
        train_dataset = OfficeHome(data_path=args["data_root"], download=True, train=True)
        test_dataset = OfficeHome(data_path=args["data_root"], download=True, train=False)

        data_file = os.path.join(args["data_root"], 'OfficeHomeDataset_10072016')
        domain_name = os.listdir(data_file)
        classnames = os.listdir(os.path.join(data_file, domain_name[0]))
    
    elif args["dataset"] == 'DomainNet':
        train_dataset = DomainNet(data_path=args["data_root"], download=True, train=True)
        test_dataset = DomainNet(data_path=args["data_root"], download=True, train=False)

        
        domain_name = os.listdir(args["data_root"])
        classnames = os.listdir(os.path.join(args["data_root"], domain_name[0]))

    elif args["dataset"] == 'DomainNetR':
        train_path = os.path.join(args["data_root"], 'train')
        test_path = os.path.join(args["data_root"], 'val')
        train_dataset = ImageFolderDataset(data_path=train_path, train=True)
        test_dataset = ImageFolderDataset(data_path=test_path, train=False)

        
        domain_name = os.listdir(args["data_root"])
        classnames = os.listdir(os.path.join(args["data_root"], domain_name[0]))

    elif args["dataset"] == 'CUB200':
        train_dataset = CUB200(data_path=args["data_root"], train=True)
        test_dataset = CUB200(data_path=args["data_root"], train=False)

        classnames_path = os.path.join(args["data_root"], 'CUB_200_2011', 'classes.txt')
        file = open(classnames_path, "r")
        content=file.readlines()
        classnames = []
        for i in range(len(content)):
            item = content[i]
            idx = item.index('.')
            item = item[idx+1:]
            item = item.replace('_', ' ')
            item = item.strip()
            classnames.append(item)

    elif args["dataset"] == 'ImagenetR':
        train_path = os.path.join(args["data_root"], 'train')
        test_path = os.path.join(args["data_root"], 'test')
        train_dataset = ImageFolderDataset(data_path=train_path, train=True)
        test_dataset = ImageFolderDataset(data_path=test_path, train=False)

        label_path = os.path.join(args["data_root"], 'README.txt')
        classnames = imagenetr_labels(label_path)

    elif args["dataset"] == 'TINYIMAGENET':
        train_dataset = TinyImageNet200(data_path=args["data_root"], download=True, train=True)
        test_dataset = TinyImageNet200(data_path=args["data_root"], download=True, train=False)

        id_path = os.path.join(args["data_root"], 'tiny-imagenet-200', 'wnids.txt')
        word_path = os.path.join(args["data_root"], 'tiny-imagenet-200','words.txt')

        classnames = tinyimagenet_labels(id_path, word_path)

    elif args["dataset"] == "Core50":
        train_dataset = Core50(data_path=args["data_root"], scenario='domains', download=False, train=True)
        test_dataset = Core50(data_path=args["data_root"],  scenario='domains', download=False, train=False)
        
        classnames = ['plug_adapter1', 'plug_adapter2', 'plug_adapter3', 'plug_adapter4', 'plug_adapter5',
                      'mobile_phone1', 'mobile_phone2', 'mobile_phone3', 'mobile_phone4', 'mobile_phone5',
                      'scissor1', 'scissor2', 'scissor3', 'scissor4', 'scissor5',
                      'light_bulb1', 'light_bulb2', 'light_bulb3', 'light_bulb4', 'light_bulb5',
                      'can1', 'can2', 'can3', 'can4', 'can5',
                      'glass1', 'glass2', 'glass3', 'glass4', 'glass5',
                      'ball1', 'ball2', 'ball3', 'ball4', 'ball5', 
                      'marker1', 'marker2', 'marker3', 'marker4', 'marker5',
                      'cup1', 'cup2', 'cup3', 'cup4', 'cup5',
                      'remote_control1', 'remote_control2', 'remote_control3', 'remote_control4', 'remote_control5']
    else:
        raise Exception("Custom dataset not implemented")

    args["num_classes"] = len(classnames)
    
    transform = [Resize(n_px, interpolation=BICUBIC), CenterCrop(n_px), _convert_image_to_rgb, ToTensor(),
                      Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))]

    return train_dataset, test_dataset, classnames, transform, transform