"""
Author: Haoran Chen
Date: 2024.07.07
"""

import os
import math
import numpy as np
import logging
from tqdm import tqdm
from PIL import Image

import clip
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from prompt import clip_text_prompt_genarator
from utils import extract_features


def test(task_id, test_loader, prompt, tokenized_prompt, custom_clip_model, vpt_model, args):
    tot_acc = 0
    tot = 0

    custom_clip_model.eval()
    vpt_model.eval()
    scale = custom_clip_model.logit_scale.exp()

    with torch.no_grad():
        for data, label, _ in test_loader:
            tot += data.size()[0]
            data = data.to(args["device"])
            label = label.to(args["device"])

            img_features, txt_features = custom_clip_model(data, prompt, tokenized_prompt)
            logits_clip = scale * img_features @ txt_features.t()

            logits_vpt, _ = vpt_model(data)
            logits_vpt = logits_vpt[:, :logits_clip.shape[1]]

            logits = (1 - nn.Sigmoid()(custom_clip_model.lambda_)) * logits_clip + nn.Sigmoid()(custom_clip_model.lambda_) * logits_vpt

            softmaxed_logits = nn.Softmax(dim=1)(logits)
            output = torch.argmax(softmaxed_logits, dim=1)

            tot_acc += (output == label).sum().item()

    return format(tot_acc / tot, '.3f')



def train_pf(clip_model, custom_clip_model, vpt_model, scenario_train, scenario_test, classnames, memory, class_mask, args):
    acc_table = np.zeros([args["step"], args["step"]])

    for task_id, dataset_train in enumerate(scenario_train):
        logging.info("Start Training Task {}".format(task_id + 1))

        custom_clip_model.train()
        vpt_model.train()

        task_classnames = classnames[task_id * args["increment"] : (task_id + 1) * args["increment"]]
        custom_clip_model.update_parameters(task_id, task_classnames, args)
       
        optimizer = torch.optim.AdamW(list(custom_clip_model.parameters()) + list(vpt_model.parameters()), lr=args["learning_rate"], weight_decay=args["weight_decay"])

        clip_parameters = sum(p.numel() for p in custom_clip_model.parameters() if p.requires_grad)
        vpt_parameters = sum(p.numel() for p in vpt_model.parameters() if p.requires_grad)
        logging.info('number of trainable params: {}'.format(clip_parameters + vpt_parameters))

        tot_clip_parameters = sum(p.numel() for p in custom_clip_model.parameters())
        tot_vpt_parameters = sum(p.numel() for p in vpt_model.parameters())
        logging.info('total number of params: {}'.format(tot_clip_parameters + tot_vpt_parameters))

        scheduler = CosineAnnealingLR(optimizer, T_max=args["epoch"])

        if task_id > 0:
            mem_x, mem_y, mem_t = memory.get()
            dataset_train.add_samples(mem_x, mem_y, mem_t)

        train_loader = DataLoader(dataset_train, batch_size=args["batch_size"], drop_last=True, num_workers=4 * args["ngpus"], shuffle=True, pin_memory=True) 

        prog_bar = tqdm(range(args["epoch"]))

        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            for data, label, _ in train_loader:
                data = data.to(args["device"])
                label = label.to(args["device"])
                label = label.to(torch.int64)

                task_prompt, task_tokenized_prompt = clip_text_prompt_genarator(task_classnames, clip_model, custom_clip_model.task_raw_prompt)
                if task_id > 0:
                    prompt = torch.cat([learned_prompt, task_prompt], dim=0)
                    tokenized_prompt = torch.cat([learned_tokenized_prompt, task_tokenized_prompt], dim=0)
                else:
                    prompt = task_prompt
                    tokenized_prompt = task_tokenized_prompt

                scale = custom_clip_model.logit_scale.exp()
                image_feature, text_feature = custom_clip_model(data, prompt, tokenized_prompt)
                logits_clip = scale * image_feature @ text_feature.t()
                logits_vpt, _ = vpt_model(data)
                logits_vpt = logits_vpt[:, :logits_clip.shape[1]]

                logits = (1 - nn.Sigmoid()(custom_clip_model.lambda_)) * logits_clip + nn.Sigmoid()(custom_clip_model.lambda_) * logits_vpt

                softmaxed_logits = F.log_softmax(logits, 1)

                if task_id > 0:      
                    weight = custom_clip_model.generate_weight_mask()
                    softmaxed_logits = softmaxed_logits * weight

                loss = F.nll_loss(softmaxed_logits, label)
                losses += loss.item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(custom_clip_model.parameters()) + list(vpt_model.parameters()), 1.0)
                optimizer.step()
            scheduler.step()

            info = "Task {}, Epoch {}/{} => Loss {:.3f}".format(
                    task_id + 1,
                    epoch + 1,
                    args["epoch"],
                    losses / len(train_loader)
                )
            prog_bar.set_description(info)
           
        for task_id_test, dataset_test_ in enumerate(scenario_test):
            if task_id_test > task_id:
                break
            else:
                test_loader_ = DataLoader(dataset_test_, batch_size=1, drop_last=True,
                                        num_workers=4 * args["ngpus"], shuffle=False)
                acc = test(task_id, test_loader_, prompt, tokenized_prompt, custom_clip_model, vpt_model, args)
                acc_table[task_id_test][task_id] = acc

        cur_acc = acc_table[:, task_id]
        logging.info("Task {} Average Accuracy is {:.3f}".format(task_id + 1, np.sum(cur_acc) / (task_id + 1)))
        logging.info(acc_table)

        learned_prompt = prompt
        learned_tokenized_prompt = tokenized_prompt

        if args["herding_method"] == 'barycenter':
            features = extract_features(dataset_train, clip_model, args)
            memory.add(*scenario_train[task_id].get_raw_samples(), features)
        elif args["herding_method"] == 'closest':
            pass
        elif args["herding_method"] == 'random':
            memory.add(*scenario_train[task_id].get_raw_samples(), None)
        else:
            raise Exception("Herding method not implemented")
    
    return acc_table
        