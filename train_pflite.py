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
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

from utils import extract_features, sample_gumbel, gumbel_softmax, gumbel_softmax_sample
from prompt import clip_text_prompt_genarator


def test(task_id, test_loader, prompt, tokenized_prompt, custom_clip_model, vpt_model, args):
    tot_acc = 0
    tot = 0
    vpt_frequency = 0

    scale = custom_clip_model.logit_scale.exp()

    custom_clip_model.eval()
    vpt_model.eval()
    
    with torch.no_grad():
        for data, label, _ in test_loader:
            data = data.to(args["device"])
            label = label.to(args["device"])

            img_features, txt_features = custom_clip_model(data, prompt, tokenized_prompt)
            logits_clip = scale * img_features @ txt_features.t()

            gumbel = custom_clip_model.gumbel_fc1(img_features)
            gumbel = custom_clip_model.gumbel_fc2(custom_clip_model.activation(gumbel))
            gumbel = gumbel_softmax(gumbel, temperature=args["gumbel_temp"])


            b, _, _, _ = data.shape
            gumbel_mask = gumbel[:, 0]
            index = (gumbel_mask == 1).nonzero(as_tuple=True)[0]
            data = data[index, :, :, :]
 
            logits_vpt_temp, _ = vpt_model(data)
            logits_vpt_temp = logits_vpt_temp[:, :logits_clip.shape[1]]

            _, f = logits_vpt_temp.shape
            logits_vpt = torch.zeros([b, f]).to(args["device"])
            logits_vpt[index, :] = logits_vpt_temp

            logits =  logits_vpt + logits_clip

            vpt_frequency += torch.sum(gumbel[:, 0])

            softmaxed_logits = logits.softmax(dim=-1)

            output = torch.argmax(softmaxed_logits, dim=1)

            tot += b
            tot_acc += (output == label).sum().item()

    return format(tot_acc / tot, '.3f'), vpt_frequency / tot



def train_pflite(clip_model, custom_clip_model, vpt_model, scenario_train, scenario_test, classnames, memory, class_mask, args):
    acc_table = np.zeros([args["step"], args["step"]])

    for task_id, dataset_train in enumerate(scenario_train):
        logging.info("Start Training Task {}".format(task_id + 1))

        custom_clip_model.train()
        vpt_model.train()

        task_classnames = classnames[task_id * args["increment"] : (task_id + 1) * args["increment"]]
        custom_clip_model.update_parameters(task_id, task_classnames, args)
                
        clip_parameters = sum(p.numel() for p in custom_clip_model.parameters() if p.requires_grad)
        vpt_parameters = sum(p.numel() for p in vpt_model.parameters() if p.requires_grad)
        logging.info('number of trainable params: {}'.format(clip_parameters + vpt_parameters))

        tot_clip_parameters = sum(p.numel() for p in custom_clip_model.parameters())
        tot_vpt_parameters = sum(p.numel() for p in vpt_model.parameters())
        logging.info('total number of params: {}'.format(tot_clip_parameters + tot_vpt_parameters))

        optimizer = torch.optim.AdamW(list(custom_clip_model.parameters()) + list(vpt_model.parameters()), lr=args["learning_rate"], weight_decay=args["weight_decay"])
        
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
                
                logits_vpt, vpt_feature = vpt_model(data)
                logits_vpt = logits_vpt[:, :text_feature.shape[0]]

                gumbel_logits = custom_clip_model.gumbel_fc1(image_feature)
                gumbel_logits = custom_clip_model.gumbel_fc2(custom_clip_model.activation(gumbel_logits))
                gumbel = gumbel_softmax(gumbel_logits, temperature=args["gumbel_temp"])
                    

                logits = (logits_vpt + logits_clip) * gumbel[:, :1] +  logits_clip * gumbel[:, 1:]
                softmaxed_logits = F.log_softmax(logits, 1)

                KD_loss = 0
                if task_id > 0:      
                    weight = custom_clip_model.generate_weight_mask()
                    softmaxed_logits = softmaxed_logits * weight

                    gumbel_copy_logits = gumbel_fc1_copy(image_feature)
                    gumbel_copy_logits = gumbel_fc2_copy(gumbel_copy_logits)

                    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(gumbel_logits/2, dim=1), F.softmax(gumbel_copy_logits/2, dim=1))

                cls_loss = F.nll_loss(softmaxed_logits, label)
                
                activation_loss = (torch.sum(gumbel[:, 0]) / gumbel.shape[0] - args["gamma"])**2

                loss = cls_loss + args["Lambda"] * activation_loss + args["Delta"] * KD_loss
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
        
        vpt_frequency_dict = {}
        tot_freq = 0.0
        for task_id_test, dataset_test_ in enumerate(scenario_test):
            if task_id_test > task_id:
                break
            else:
                test_loader_ = DataLoader(dataset_test_, batch_size=args["batch_size"], drop_last=True,
                                        num_workers=4 * args["ngpus"], shuffle=False)
                acc, frequency = test(task_id, test_loader_, prompt, tokenized_prompt, custom_clip_model, vpt_model, args)
                acc_table[task_id_test][task_id] = acc
                tot_freq += frequency
                vpt_frequency_dict["Task {}".format(task_id_test + 1)] = format(frequency,'.2f')
        avg_frequency = (tot_freq / (task_id + 1)).item()
        vpt_frequency_dict["Avg"] = format(avg_frequency, '.2f')
        cur_acc = acc_table[:, task_id]
        logging.info("Task {} Average Accuracy is {:.3f}".format(task_id + 1, np.sum(cur_acc) / (task_id + 1)))
        logging.info(acc_table)
        logging.info("VPT frequency is {}".format(vpt_frequency_dict))

        learned_prompt = prompt
        learned_tokenized_prompt = tokenized_prompt

        gumbel_fc1_copy = copy.deepcopy(custom_clip_model.gumbel_fc1)
        gumbel_fc2_copy = copy.deepcopy(custom_clip_model.gumbel_fc2)

        print("start generating memory")
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