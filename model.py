"""
Author: Haoran Chen
Date: 2024.07.07
"""

import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from timm.models.registry import register_model
from vision_transformer import _create_vision_transformer

__all__ = [
    'vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224',
]

@register_model
def vit_tiny_patch16_224(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        prompts = prompts.type(self.dtype)
        # prompts = prompts.float()
        x = prompts + self.positional_embedding.type(self.dtype)
        # x = prompts + self.positional_embedding.float()
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        # x = self.ln_final(x).float()

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class VisionEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.proj = clip_model.visual.proj

    def forward(self, x, prompt):
        batch = x.shape[0]
        x = self.conv1(x)
        x = x.reshape(batch, x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(batch, 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        cls_token = x[:, 0, :].expand(1, -1, -1)
        cls_token = cls_token.permute(1, 0, 2)
        x = x[:, 1:, :]

        prompt = prompt.repeat(batch, 1, 1)
        prompt = prompt.to(x.device)
        x = torch.cat([cls_token, prompt, x], dim=1)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = x[:, 1 : (1+prompt.shape[1]), :]
        x = x.mean(dim=1)
        # x = x[:, 0]

        x = self.ln_post(x)

        x = x @ self.proj

        return x


# custom clip with learned prompts
class Clip_PF(nn.Module):
    def __init__(self, clip_model, args):
        super().__init__()
        self.image_encoder = VisionEncoder(clip_model)
        # self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.image_prompt = nn.Parameter(torch.randn((1, args["clip_image_prompt_length"], 768)))
        self.alpha = nn.Parameter(torch.ones(args["batch_size"], args["increment"]))

        self.lambda_ = nn.Parameter(torch.ones([1]))

    def update_parameters(self, task_id, task_classnames, args):
        task_raw_prompt = torch.empty(args["increment"], args["M"], 512, requires_grad=True, device=args["device"])
        
        nn.init.normal_(task_raw_prompt, std=0.01)

        self.task_raw_prompt = nn.Parameter(task_raw_prompt)

        beta = torch.ones(args["batch_size"], task_id * args["increment"], requires_grad=True, device=args["device"])
        self.beta = nn.Parameter(beta)

        self.task_id = task_id
        self.task_classnames = task_classnames

        self.theta = task_id / 2

    def generate_weight_mask(self):
        weight_for_previous_task = self.theta / nn.Sigmoid()(self.beta)
        weight_for_current_task = nn.Sigmoid()(self.alpha) 
        weight = torch.cat([weight_for_previous_task, weight_for_current_task], dim=1)

        return weight

    def forward(self, image, prompt, tokenized_prompts):
        image_features = self.image_encoder(image, self.image_prompt)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self.text_encoder(prompt, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features

# custom clip with learned prompts
class Clip_PFLite(nn.Module):
    def __init__(self, clip_model, args):
        super().__init__()
        self.image_encoder = VisionEncoder(clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.image_prompt = nn.Parameter(torch.randn((1, args["clip_image_prompt_length"], 768)))
        self.alpha = nn.Parameter(torch.ones(args["batch_size"], args["increment"]))

        self.gumbel_fc1 = nn.Linear(512, 100)
        self.gumbel_fc2 = nn.Linear(100, 2)

        self.lambda_ = nn.Parameter(torch.ones([1]))

        if args["activation"] == 'relu':
            self.activation =  nn.ReLU
        elif args["activation"] == 'tanh':
            self.activation = torch.tanh


    def update_parameters(self, task_id, task_classnames, args):
        task_raw_prompt = torch.empty(args["increment"], args["M"], 512, requires_grad=True, device=args["device"])
        nn.init.normal_(task_raw_prompt, std=0.01)

        self.task_raw_prompt = nn.Parameter(task_raw_prompt)

        beta = torch.ones(args["batch_size"], task_id * args["increment"], requires_grad=True, device=args["device"])
        self.beta = nn.Parameter(beta)

        self.task_id = task_id
        self.task_classnames = task_classnames
        self.theta = task_id / 2

    def generate_weight_mask(self):
        weight_for_previous_task = self.theta / nn.Sigmoid()(self.beta)
        weight_for_current_task = nn.Sigmoid()(self.alpha) 
        weight = torch.cat([weight_for_previous_task, weight_for_current_task], dim=1)

        return weight

    def forward(self, image, prompt, tokenized_prompts):
        image_features = self.image_encoder(image, self.image_prompt)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self.text_encoder(prompt, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features



