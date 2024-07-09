"""
Author: Haoran Chen
Date: 2024.07.07
"""

import argparse
import os
import json
import sys
import numpy as np
import random
import logging
import time
from datetime import datetime

import torch
import torch.nn as nn
import clip
from dataset import gen_dataset
from continuum import ClassIncremental, rehearsal, ContinualScenario
from timm.models import create_model
from model import Clip_PF, Clip_PFLite
from train_pf import train_pf
from train_pflite import train_pflite
import utils

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param


def set_log():
    args["output_folder"] = "{}/{}/{}/{}/".format(args["file_root"], args["model_type"], args["dataset"], args["backbone"])

    if not os.path.exists(args["output_folder"]):
        os.makedirs(args["output_folder"])

    now = datetime.now()
    log_filename = now.strftime("%Y-%m-%d_%H:%M:%S")

    logfilename = "{}/{}/{}/{}/{}".format(
        args["file_root"],
        args["model_type"],
        args["dataset"],
        args["backbone"],
        log_filename
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def get_clip_model(args):
    clip_model_path = args["file_root"] + '/' + args["backbone"] + '.pt'

    if os.path.exists(clip_model_path):
        clip_model, _ = clip.load(args["backbone"], device=args["device"], model_path=clip_model_path)
    else:
        raise Exception("Model doesn't exist! Please manually download it!")
    
    utils.convert_models_to_fp32(clip_model)

    for name, param in clip_model.named_parameters():
        param.requires_grad_(False)

    return clip_model

def set_models(clip_model, args):
    vpt_model = create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes=args["num_classes"],
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=None,
        prompt_length=args["vpt_prompt_length"],   
        prompt_init='uniform'
    )

    vpt_model = vpt_model.to(args["device"])

    freeze = ['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed']
    for n, p in vpt_model.named_parameters():
        if n.startswith(tuple(freeze)):
            p.requires_grad = False
            
    vpt_model = nn.DataParallel(vpt_model)

    if args["model_type"] == 'PF':
        custom_clip_model = Clip_PF(clip_model, args).to(args["device"])
    elif args["model_type"] == 'PF_Lite':
        custom_clip_model = Clip_PFLite(clip_model, args).to(args["device"])
    else:
        raise Exception("Model type doesn't exist!")

    custom_clip_model = nn.DataParallel(custom_clip_model)
    custom_clip_model = custom_clip_model.module

    for name, param in custom_clip_model.named_parameters():
        if (not 'prompt' in name) and (not 'alpha' in name) and (not 'beta' in name) and (not 'lambda_' in name) and (not 'gumbel' in name):
            param.requires_grad_(False)

    return vpt_model, custom_clip_model

def main(args):
    train_dataset, test_dataset, classnames, transform_train, transform_test = gen_dataset(args)

    args["increment"] = int(args["num_classes"] / args["step"])

    class_mask = list()
    labels = [i for i in range(len(classnames))]

    for _ in range(args["step"]):
        scope = labels[:args["increment"]]
        labels = labels[args["increment"]:]
        class_mask.append(scope)

    scenario_train = ClassIncremental(train_dataset, increment=args["increment"], transformations=transform_train)
    scenario_test = ClassIncremental(test_dataset, increment=args["increment"], transformations=transform_test)

    memory = rehearsal.RehearsalMemory(memory_size=args["memory_size"], herding_method=args["herding_method"])
    
    clip_model = get_clip_model(args)
    vpt_model, custom_clip_model = set_models(clip_model, args)

    t = time.time()

    if args["model_type"] == "PF":
        acc_table = train_pf(clip_model, custom_clip_model, vpt_model, scenario_train, scenario_test, classnames, memory, class_mask, args)
    elif args["model_type"] == "PF_Lite":
        acc_table = train_pflite(clip_model, custom_clip_model, vpt_model, scenario_train, scenario_test, classnames, memory, class_mask, args)
    else:
        raise Exception("Model type doesn't exist!")

    forgetting = np.mean((np.max(acc_table, axis=1) - acc_table[:, args["step"] - 1]))
    logging.info("Forgetting: {:.3f}".format(forgetting))
    logging.info(f'Cost:{time.time() - t:.4f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and Evaluation Script')
    parser.add_argument('--config', type=str, default='./config/pf_cifar.json', help='Json file of settings.')
    args = parser.parse_args()

    param = load_json(args.config)

    args = vars(args) 
    args.update(param) 

    set_log()

    for key, value in args.items():
        logging.info("{}: {}".format(key, value))

    main(args)
