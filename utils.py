"""
Author: Haoran Chen
Date: 2024.07.07
"""

import clip
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def clip_text_prompt(classnames, clip_model, prompt):

    length = prompt.shape[1]
    prompt_prefix = " ".join(["X"] * (length * 2 ))

    classnames = [name.replace("_", " ") for name in classnames]
    prompt_with_cls = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompt_with_cls]).to('cuda')

    with torch.no_grad():
        embedding = clip_model.token_embedding(tokenized_prompts).float()

    prefix = embedding[:, :1, :]
    suffix = embedding[:, 1 + length:, :]

    prompt_full = torch.cat(
        [prefix,  # (n_cls, 1, dim)
         prompt,  # (n_cls, M1, dim)
         suffix,  # (n_cls, *, dim)
         ],
        dim=1)

    return prompt_full, tokenized_prompts


def extract_features(dataset, clip_model, args):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        num_workers=4 * args["ngpus"],
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )

    features, targets = [], []

    with torch.no_grad():
        for x, y, _ in loader:
            if hasattr(clip_model, 'module'):
                feats = clip_model.module.encode_image(x.cuda())
            else:
                feats = clip_model.encode_image(x.cuda())

            feats = feats.cpu().numpy()
            features.append(feats)

    features = np.concatenate(features)
    return features


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=True):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard

def imagenet1k_labels(path):
    with open(path) as f:
        class_map = f.readlines()

    classnames = []
    for map_ in class_map:
        label, sem_label, *_ = map_.split(',')
        classnames.append(sem_label[1:])

    return classnames

def imagenetr_labels(path):
    with open(path) as f:
        class_map = f.readlines()

    classnames = []
    for map_ in class_map:
        label, name = map_.split()
        classnames.append(name)
    return classnames

def tinyimagenet_labels(id_path, word_path):
    with open(id_path) as f:
        lines = f.readlines()

    wnids = []
    for item in lines:
        wnids.append(item.replace("\n", ""))
    wnids.sort()

    with open(word_path) as f:
        lines = f.readlines()

    words = {}
    for item in lines:
        item = item.replace("\n", "")
        item = item.replace("\t", " ")
        if "," in item:
            index = item.index(',')
            item = item[:index]
        id = item[:9]
        label = item[10:]
        words[id] = label

    classnames = []
    for item in wnids:
        label = words[item]
        classnames.append(label)

    return classnames


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

